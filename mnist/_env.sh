#!/bin/bash
# Shared cluster environment for w1w2_bayes on Amarel (Rutgers).
#
# Source this at the top of any job script:
#     source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"
#
# All cluster-specific settings live here, so moving to another cluster
# means editing this file only.

PROJECT_ROOT=/cache/home/bj394/w1w2_bayes
CONDA_ENV=w1w2_bayes

# Amarel's community software tree is not on the default module path.
source /etc/profile.d/lmod.sh 2>/dev/null || \
    source /opt/ohpc/admin/lmod/lmod/init/bash 2>/dev/null || true
module use /projects/community/modulefiles
module load miniforge/26.3.2-ez82

# `conda activate` needs shell hooks that aren't set up in batch jobs.
source activate "${CONDA_ENV}"

cd "${PROJECT_ROOT}"
export PYTHONUNBUFFERED=1
mkdir -p logs
