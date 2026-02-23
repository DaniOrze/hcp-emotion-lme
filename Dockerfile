# ==================================================================
# Dockerfile for HCP Emotion LME Analysis
#
# Provides a fully reproducible environment with:
#   - Python 3.11 + scientific stack
#   - R 4.3 + lme4, lmerTest (gold-standard LME implementation)
#   - pymer4 (Python interface to R lme4)
#   - All neuroimaging packages (nilearn, nibabel)
#
# Build:
#   docker build -t hcp-emotion-lme .
#
# Run (mount data and results directories):
#   docker run -v /path/to/hcp/data:/data/hcp_emotion/raw \
#              -v /path/to/results:/results/hcp_emotion \
#              hcp-emotion-lme bash run_full_analysis_v2.sh
#
# Run interactively:
#   docker run -it -v /path/to/data:/data/hcp_emotion/raw \
#              hcp-emotion-lme bash
# ==================================================================

FROM python:3.11-slim

LABEL maintainer="HCP Emotion LME Analysis"
LABEL description="Reproducible environment for GLM vs LME comparison"

# Install system dependencies + R
RUN apt-get update && apt-get install -y --no-install-recommends \
    r-base \
    r-base-dev \
    libcurl4-openssl-dev \
    libssl-dev \
    libxml2-dev \
    libgit2-dev \
    libfontconfig1-dev \
    libharfbuzz-dev \
    libfribidi-dev \
    libfreetype6-dev \
    libpng-dev \
    libtiff5-dev \
    libjpeg-dev \
    gfortran \
    && rm -rf /var/lib/apt/lists/*

# Install R packages (lme4 + lmerTest for Satterthwaite dof)
RUN R -e 'install.packages(c("lme4", "lmerTest", "Matrix", "nloptr"), \
          repos="https://cloud.r-project.org/")'

# Install Python packages
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy analysis code
COPY . /app/

# Create output directories
RUN mkdir -p /results/hcp_emotion/derivatives \
             /results/hcp_emotion/figures \
             /results/hcp_emotion/logs

# Verify R lme4 and pymer4 are functional
RUN python -c "from pymer4.models import Lmer; print('pymer4 + R lme4 OK')"
RUN R -e 'library(lme4); library(lmerTest); cat("R lme4 + lmerTest OK\n")'

# Default command
CMD ["bash", "run_full_analysis_v2.sh"]
