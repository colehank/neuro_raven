# %%
import mne
import os
from dotenv import load_dotenv
from pathlib import Path
import pandas as pd
from tqdm.auto import tqdm
import numpy as np

load_dotenv()
DATA_DIR = Path(os.getenv("DATA_DIR"))
PREP_DIR = DATA_DIR / "bids" / "derivatives" / "preprocessed"
EPO_DIR = DATA_DIR / "bids" / "derivatives" / "epoched"
EPO_DIR.mkdir(exist_ok=True)


def epoit(fp):
    info = fp.stem
    subid = info.split("_")[0].replace("sub-", "")
    session = info.split("_")[1].replace("ses-", "")
    runid = info.split("_")[3].replace("run-", "")
    meta = pd.read_csv(PREP_DIR.parent / "events_metadata.tsv", sep="\t")
    new_event_id = {k:v for k,v in zip(meta.event_name, meta.marker)}
    raw = mne.io.read_raw(fp)
    ev, evid = mne.events_from_annotations(raw)
    
    new_event_id = {k:v for k,v in new_event_id.items() if k in evid}
    evid_ = {v: k for k, v in evid.items()}
    ev_names = [evid_[i] for i in ev[:, 2]]
    meta = pd.DataFrame(ev_names, columns=["event_name"]).merge(meta, on="event_name", how="left")
    new_events = np.empty(ev.shape, dtype=ev.dtype)
    new_events[:, 0] = ev[:, 0]
    new_events[:, 1] = ev[:, 1]
    new_events[:, 2] = meta["marker"].values
    meta["sub"] = subid
    meta["ses"] = session
    meta["run"] = runid
    epo = mne.Epochs(
        raw=raw,
        events=new_events,
        event_id=new_event_id,
        tmin=-0.2,
        tmax=2,
        metadata=meta,
        baseline=(-0.2, 0),
        event_repeated="drop"
    )
    return epo
# %%
all_preped_fps = list(PREP_DIR.rglob("*.fif"))
sub_fps = {}
for fp in all_preped_fps:
    info = fp.stem
    subid = info.split("_")[0].replace("sub-", "")
    if subid not in sub_fps:
        sub_fps[subid] = []
    sub_fps[subid].append(fp)
sub_fps = {k: sorted(v, key=lambda x: x.stem) for k, v in sub_fps.items()}
# %%
for sub, fps in tqdm(sub_fps.items(), desc="cooking subs", position=0):
    subepos = []
    for fp in tqdm(fps, desc="cooking files", leave=False, position = 1):
        epo = epoit(fp)
        subepos.append(epo)
    subepo = mne.concatenate_epochs(subepos)
    subepo.save(EPO_DIR / f"sub-{sub}_epo.fif", overwrite = True)
# %%
