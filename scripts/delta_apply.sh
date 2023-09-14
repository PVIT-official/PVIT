BASE_PATH=$(cd "$(dirname "$0")"; pwd)
BASE_PATH=${BASE_PATH%%/scripts*}

cd $BASE_PATH

python3 -m pvit.model.apply_delta \
    --base $BASE_PATH/model_weights/llama-7b \
    --target $BASE_PATH/model_weights/pvit \
    --delta $BASE_PATH/model_weights/pvit-delta
