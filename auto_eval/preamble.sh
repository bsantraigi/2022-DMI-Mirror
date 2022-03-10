# HELPER FUNCTIONS
RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m' # No Color

error(){
  printf "${RED}ERROR${NC} $1\n"
}

status(){
  printf "${GREEN}STATUS${NC} $1\n"
}

# REQUIRED VARIABLES
# 1. CKPT_LINK
# 2. MODEL_NAME_PATH
# 3. IS_ROB

confirm() {
  VAR_NAME="${1}"
  if [[ -z "${!VAR_NAME}" ]]; then
    error "set the environment variable ${VAR_NAME}"
    exit
  else
    status "Recieved value: ${VAR_NAME} = ${!VAR_NAME}"
  fi
}

confirm CKPT_LINK
confirm MODEL_NAME_PATH

# DOWNLOAD MODEL CHECKPOINTS
mkdir "checkpoints/$(dirname ${MODEL_NAME_PATH})"
wget -q -c "${CKPT_LINK}" -O "checkpoints/${MODEL_NAME_PATH}"