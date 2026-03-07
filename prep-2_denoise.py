# %%
from mne_bids import find_matching_paths
from meeg_utils import BatchPreprocessingPipeline as PrepPipe
from meeg_utils import setup_logging
from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()
DATA_DIR = Path(os.getenv("DATA_DIR"))
bids_dir = DATA_DIR / "bids"
DER_DIR = bids_dir / "derivatives"
DER_DIR.mkdir(exist_ok=True)
setup_logging(log_filename=".logs/preprocessing.log")
all_bids_paths = find_matching_paths(bids_dir, extensions=".vhdr")
# %%
pipe = PrepPipe(
    input_paths=all_bids_paths,
    output_dir=DER_DIR / "preprocessed",
    use_cuda=True,
    n_jobs=8,
)
pipe.run(
    filter_params={"highpass": 0.1, "lowpass": 100.0, "sfreq": 250.0},
    detect_bad_channels=True,
    remove_line_noise=True,
    apply_ica=True,
    ica_params={"n_components": 20, "method": "infomax", "regress": True},
    skip_existing=True,
)
# %%
