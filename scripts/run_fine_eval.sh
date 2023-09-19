BASE_PATH=$(cd "$(dirname "$0")"; pwd)
BASE_PATH=${BASE_PATH%%/scripts*}

cd $BASE_PATH

python pvit/run_fine_eval.py \
    --image_path "./fine_eval/images" \
    --input_data "./fine_eval/instructions.jsonl" \
    --output_data "./fine_eval/pvit_answer.jsonl"
