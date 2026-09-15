# Jango shared production environment defaults for <SLURM_NODE>.
# This file is sourced automatically by `conda activate jango`.

export JANGO_SOURCE=${JANGO_SOURCE:-/path/to/jango}
export JANGO_WORK_ROOT=${JANGO_WORK_ROOT:-/path/to/jango_runs/work}
export JANGO_OUTPUTS_ROOT=${JANGO_OUTPUTS_ROOT:-/path/to/jango_runs/outputs}
export JANGO_DATA_ROOT=${JANGO_DATA_ROOT:-/path/to/jango_runs/data}
export JANGO_MPNN_ROOT=${JANGO_MPNN_ROOT:-${JANGO_DATA_ROOT}/mpnn}
export JANGO_FOLD_ROOT=${JANGO_FOLD_ROOT:-${JANGO_DATA_ROOT}/folds}
export JANGO_SOURCE_LABEL=${JANGO_SOURCE_LABEL:-sabdab2}
export JANGO_RAW_TARBALLS=${JANGO_RAW_TARBALLS:-/path/to/jango/data/tarballs}
export JANGO_PATHS_CONFIG=${JANGO_PATHS_CONFIG:-/path/to/jango/configs/production/paths.env}
export JANGO_SHARED_DATA_ROOT=${JANGO_SHARED_DATA_ROOT:-/path/to/jango/data}

# Prefer the curated shared MSA cache for the shared installation. Users may
# override MSA_CACHE_ROOT before activation or source a backend env afterward.
export MSA_CACHE_ROOT=${MSA_CACHE_ROOT:-/path/to/jango/data/msa}

# The default interactive backend is ESMFold2. Backend-specific commands may
# source configs/production/opendde.env or configs/production/boltz2.env later.
if [ -f /path/to/jango/configs/production/esmfold2.env ]; then
  # shellcheck source=/path/to/jango/configs/production/esmfold2.env
  source /path/to/jango/configs/production/esmfold2.env
fi

mkdir -p "${JANGO_WORK_ROOT}" "${JANGO_OUTPUTS_ROOT}" "${JANGO_DATA_ROOT}" /path/to/jango_runs/cache /path/to/jango_runs/tmp 2>/dev/null || true

jango-use-run() {
  if [ "$#" -ne 1 ] || [ -z "$1" ]; then
    echo "usage: jango-use-run <run-id>" >&2
    return 2
  fi
  case "$1" in
    */*|*..*|.*|*' '*|*':'*)
      echo "invalid run id: $1" >&2
      return 2
      ;;
  esac
  export JANGO_RUN_NAME="$1"
  export JANGO_RUN_ROOT="${JANGO_WORK_ROOT}/${JANGO_RUN_NAME}"
  export JANGO_OUTPUT_ROOT="${JANGO_RUN_ROOT}"
  mkdir -p "${JANGO_RUN_ROOT}" 2>/dev/null || true
  echo "JANGO_RUN_ROOT=${JANGO_RUN_ROOT}"
}

jango-use-backend() {
  if [ "$#" -ne 1 ]; then
    echo "usage: jango-use-backend <esmfold2|opendde|boltz2>" >&2
    return 2
  fi
  case "$1" in
    esmfold2) source /path/to/jango/configs/production/esmfold2.env ;;
    opendde) source /path/to/jango/configs/production/opendde.env ;;
    boltz2) source /path/to/jango/configs/production/boltz2.env ;;
    *) echo "unknown backend: $1" >&2; return 2 ;;
  esac
  echo "FOLDING_BACKEND=${FOLDING_BACKEND}"
}
