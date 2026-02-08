from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent

PROJECT_ROOT = PROJECT_DIR / "results" / "hcp_emotion"

RAW_DATA = Path("/data/hcp_emotion/raw")

DERIVATIVES = PROJECT_ROOT / "derivatives"
FIRST_LEVEL = DERIVATIVES / "first_level"
GROUP_LEVEL = DERIVATIVES / "group_level"
LME_VOXELWISE = DERIVATIVES / "lme_voxelwise"
FIGURES = PROJECT_ROOT / "figures"
LOGS = PROJECT_ROOT / "logs"

for p in [DERIVATIVES, FIRST_LEVEL, GROUP_LEVEL, LME_VOXELWISE, FIGURES, LOGS]:
    p.mkdir(parents=True, exist_ok=True)

def get_subject_dirs():
    return sorted([d for d in RAW_DATA.iterdir() if d.is_dir()])

def get_subject_ids():
    return [d.name for d in get_subject_dirs()]
