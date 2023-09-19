BASE_PATH=$(cd "$(dirname "$0")"; pwd)
BASE_PATH=${BASE_PATH%%/scripts*}

cd $BASE_PATH

export https_proxy= http_proxy= all_proxy=

# launch a controller
python -m pvit.serve.controller --host 0.0.0.0 --port $CONTROLLER_PORT &

# launch a worker
python -m pvit.serve.model_worker \
    --host 0.0.0.0 \
    --controller http://0.0.0.0:$CONTROLLER_PORT \
    --port $WORKER_PORT \
    --worker http://0.0.0.0:$WORKER_PORT \
    --model-path $MODEL_PATH \
    --multi-modal \
    --num-gpus 1 &
