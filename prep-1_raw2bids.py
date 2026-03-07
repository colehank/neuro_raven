# %%
from pathlib import Path
from dotenv import load_dotenv
import os
from mne_bids import BIDSPath, write_raw_bids
import mne

# from tqdm_joblib import tqdm_joblib
# from joblib import Parallel, delayed
from tqdm.auto import tqdm
import warnings
import pandas as pd

warnings.filterwarnings(
    "ignore", message=".*Encountered unsupported non-voltage units.*"
)
load_dotenv()
DATA_DIR = Path(os.getenv("DATA_DIR"))
raw_dir = DATA_DIR / "raw"
bids_dir = DATA_DIR / "bids"
bids_dir.mkdir(exist_ok=True)

EEG_RENAME_MAP = {
    "FP1": "Fp1",
    "FP2": "Fp2",
    "FPZ": "Fpz",
    "FZ": "Fz",
    "FCZ": "FCz",
    "CZ": "Cz",
    "CPZ": "CPz",
    "PZ": "Pz",
    "POZ": "POz",
    "OZ": "Oz",
}

# 规则编码（3水平）：con(恒定), di3(三分布), pro(递增)
# 刺激编码（4水平）：edg(符号-图形边数)、qty(符号-图形个数), ara(非符号-阿拉伯), chi(非符号-中文)
# s1(on)off: 第1屏呈现/结束
# res(on)off: 反应开始/结束

EVENT_ID = {
    "begin": 1,
    # 1. 恒定规则
    "con_edg_s1on": 2,
    "con_edg_s1off": 3,
    "con_edg_s2on": 4,
    "con_edg_s2off": 5,
    "con_edg_reson": 6,
    "con_edg_resoff": 7,
    "con_qty_s1on": 8,
    "con_qty_s1off": 9,
    "con_qty_s2on": 10,
    "con_qty_s2off": 11,
    "con_qty_reson": 12,
    "con_qty_resoff": 13,
    "con_ara_s1on": 38,
    "con_ara_s1off": 39,
    "con_ara_s2on": 40,
    "con_ara_s2off": 41,
    "con_ara_reson": 42,
    "con_ara_resoff": 43,
    "con_chi_s1on": 56,
    "con_chi_s1off": 57,
    "con_chi_s2on": 58,
    "con_chi_s2off": 59,
    "con_chi_reson": 60,
    "con_chi_resoff": 61,
    # 2. 三分布规则
    "di3_qty_s1on": 14,
    "di3_qty_s1off": 15,
    "di3_qty_s2on": 16,
    "di3_qty_s2off": 17,
    "di3_qty_reson": 18,
    "di3_qty_resoff": 19,
    "di3_edg_s1on": 26,
    "di3_edg_s1off": 27,
    "di3_edg_s2on": 28,
    "di3_edg_s2off": 29,
    "di3_edg_reson": 30,
    "di3_edg_resoff": 31,
    "di3_ara_s1on": 44,
    "di3_ara_s1off": 45,
    "di3_ara_s2on": 46,
    "di3_ara_s2off": 47,
    "di3_ara_reson": 48,
    "di3_ara_resoff": 49,
    "di3_chi_s1on": 62,
    "di3_chi_s1off": 63,
    "di3_chi_s2on": 64,
    "di3_chi_s2off": 65,
    "di3_chi_reson": 66,
    "di3_chi_resoff": 67,
    # 3. 递增规则
    "pro_qty_s1on": 20,
    "pro_qty_s1off": 21,
    "pro_qty_s2on": 22,
    "pro_qty_s2off": 23,
    "pro_qty_reson": 24,
    "pro_qty_resoff": 25,
    "pro_edg_s1on": 32,
    "pro_edg_s1off": 33,
    "pro_edg_s2on": 34,
    "pro_edg_s2off": 35,
    "pro_edg_reson": 36,
    "pro_edg_resoff": 37,
    "pro_ara_s1on": 50,
    "pro_ara_s1off": 51,
    "pro_ara_s2on": 52,
    "pro_ara_s2off": 53,
    "pro_ara_reson": 54,
    "pro_ara_resoff": 55,
    "pro_chi_s1on": 68,
    "pro_chi_s1off": 69,
    "pro_chi_s2on": 70,
    "pro_chi_s2off": 71,
    "pro_chi_reson": 72,
    "pro_chi_resoff": 73,
}


def set_annos(raw):
    event_id = {str(v): k for k, v in EVENT_ID.items()}
    _, ori_id = mne.events_from_annotations(raw)
    uniq_ev = set(ori_id.keys())
    share_id = {str(k): event_id[k] for k in uniq_ev if k in event_id}
    raw.annotations.rename(share_id)


def fit_standard_montage(raw):
    raw.set_channel_types({"HEO": "eog", "VEO": "eog"})
    raw.rename_channels(EEG_RENAME_MAP)
    if "CB1" in raw.ch_names:
        raw.drop_channels(["CB1", "CB2"])  # not satisfied with the 1020
    if "EKG" in raw.ch_names:
        raw.set_channel_types({"EKG": "ecg"})
    if "EMG" in raw.ch_names:
        raw.set_channel_types({"EMG": "emg"})
    montage = mne.channels.make_standard_montage("standard_1020")
    raw.set_montage(montage)
    return raw


def get_all_raw_fps():
    raw_fps = {}
    sub_dirs = raw_dir.glob("sub*")
    for sub_dir in sub_dirs:
        for fp in sub_dir.glob("*.dat"):
            subid = fp.stem.split("sub")[1].split("_")[0]
            runid = fp.stem.split("run")[1].split(".")[0]
            raw_fps[(subid, runid)] = fp
            # Ensure subid and runid are zero-padded to 2 digits
            subid = fp.stem.split("sub")[1].split("_")[0].zfill(2)
            runid = fp.stem.split("run")[1].split(".")[0].zfill(2)
    return raw_fps


def process_one(subid, runid, fp):
    raw = mne.io.read_raw_curry(fp, verbose=False)
    raw = fit_standard_montage(raw)
    runid = runid.zfill(2)
    bids_path = BIDSPath(
        subject=subid,
        session="01",  # only 1 session per sub
        run=runid,
        task="raven",
        root=bids_dir,
    )
    temp_fp = f"{subid}_run{runid}_temp_raw.fif"
    temp_raw = raw.copy()
    temp_raw.save(temp_fp, overwrite=True)
    raw = mne.io.read_raw_fif(temp_fp, verbose=False)
    set_annos(raw)
    ev, evid = mne.events_from_annotations(raw)
    write_raw_bids(
        raw,
        bids_path,
        overwrite=True,
        format="auto",
        event_id=evid,
        events=ev,
    )

    temp_fp = Path(temp_fp)
    if temp_fp.exists():
        temp_fp.unlink()


# %%
if __name__ == "__main__":
    raw_fps = get_all_raw_fps()

    # with tqdm_joblib(desc="cooking", total=len(raw_fps)) as progress_bar:
    #     Parallel(n_jobs=10)(
    #         delayed(process_one)(subid, runid, fp)
    #         for (subid, runid), fp in raw_fps.items()
    #     )# 多jobs会导致sidecars文件冲突，不能完整保存整个bids

    for (subid, runid), fp in tqdm(raw_fps.items(), desc="raw2bids"):
        process_one(subid, runid, fp)
    # %%

    df = pd.DataFrame(EVENT_ID.items(), columns=["event_name", "marker"])
    rules_map = {"con": "constant", "di3": "distribution", "pro": "progression"}
    type_map = {
        "edg": "symbol",
        "qty": "symbol",
        "ara": "non-symbol",
        "chi": "non-symbol",
    }
    level_map = {"edg": "edges", "qty": "quantity", "ara": "arabic", "chi": "chinese"}
    status_map = {
        "s1on": "screen1_on",
        "s1off": "screen1_off",
        "s2on": "screen2_on",
        "s2off": "screen2_off",
        "reson": "response_start",
        "resoff": "response_end",
    }
    split_df = df["event_name"].str.split("_", expand=True)
    df["status"] = split_df[2].map(status_map)
    df["rule"] = split_df[0].map(rules_map)
    df["stim_type"] = split_df[1].map(type_map)
    df["stim_level"] = split_df[1].map(level_map)
    df.loc[
        df["event_name"] == "begin", ["rule", "stim_type", "stim_level", "status"]
    ] = "run_start"
    df.to_csv(bids_dir / "derivatives" / "events_metadata.tsv", sep="\t", index=False)
