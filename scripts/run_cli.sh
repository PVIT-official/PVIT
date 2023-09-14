BASE_PATH=$(cd "$(dirname "$0")"; pwd)
BASE_PATH=${BASE_PATH%%/scripts*}

cd $BASE_PATH

python pvit/run_cli.py \
    --query "<image>\nWhat is the animal in <Region><L600><L600><L900><L1000></Region>?" \
    --image_path "./images/example.jpg"
