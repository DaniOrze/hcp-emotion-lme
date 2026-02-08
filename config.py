# fMRI parameters
TR = 0.72
HRF_MODEL = "spm"
HIGH_PASS = 1 / 128
NOISE_MODEL = "ar1"

# Thresholds
Z_THRESHOLD = 3.1          # p < 0.001 uncorrected
CLUSTER_THRESHOLD = 50     # minimum cluster size in voxels
SMOOTH_FWHM = 5.0          # group level smoothing

# Computation
N_JOBS = 6          # GLM paralelo (6 × ~4GB ≈ 24GB)
LME_N_JOBS = 3      # LME paralelo (3 processos bem pesados)
CHUNK_SIZE = 5000   # ideal para 64GB


# LME settings
LME_METHOD = "powell"
LME_REML = True
LME_MAX_ITER = 100

# Contrast of interest
CONTRAST_NAME = "fear_minus_neut"
