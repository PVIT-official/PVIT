import argparse, requests, re, os, json, tqdm, base64
from io import BytesIO
from PIL import Image
import conversation as conversation_lib

REGION_PATTERN = r'<Region>(\s*<L(\d{1,4})>\s*<L(\d{1,4})>\s*<L(\d{1,4})>\s*<L(\d{1,4})>\s*)</Region>'


def convert_from_prompt_tokens(s_with_region_tokens):
    boxes = []
    matched_strs = []
    boxes_str = re.findall(REGION_PATTERN, s_with_region_tokens)
    for boxes_str_i in boxes_str:
        matched_str_i, boxes_str_i = boxes_str_i[0], boxes_str_i[1:]
        boxes.append(str([int(s)/1000 for s in boxes_str_i]).replace(" ", ""))
        matched_strs.append("<Region>" + matched_str_i + "</Region>")
    return matched_strs, boxes


def convert_to_plain_text_from_prompt_tokens(s_with_region_tokens):
    matched_strs, boxes = convert_from_prompt_tokens(s_with_region_tokens)
    assert len(matched_strs) == len(boxes)
    for i in range(len(matched_strs)):
        s_with_region_tokens = s_with_region_tokens.replace(matched_strs[i], boxes[i])
    return s_with_region_tokens


def encode_img(img):
    if isinstance(img, str):
        img = Image.open(img).convert('RGB')
    im_file = BytesIO()
    img.save(im_file, format="JPEG")
    # im_bytes: image in binary format
    im_bytes = im_file.getvalue()
    im_b64 = base64.b64encode(im_bytes).decode()
    return im_b64


def send_one_message(query, image_path, region_style="token", worker_addr=None, max_new_tokens=32):
    conv = conversation_lib.default_conversation.copy()
    if isinstance(query, str):
        query = {
            'conversations': [{'value': query}]
        }
    # TODO: only one conversation here
    if region_style == "token":
        value = query['conversations'][0]["value"]
    elif region_style == "text":
        # convert prompt tokens in value to plain texts
        value = convert_to_plain_text_from_prompt_tokens(query['conversations'][0]["value"])
    else:
        raise Exception("Wrong Region Style. ")
    conv.append_message(conv.roles[0], value)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    headers = {"User-Agent": "PVIT Client"}
    pload = {
        "prompt": prompt,
        "images": [encode_img(image_path)],
        "max_new_tokens": max_new_tokens,
        "temperature": 0.0,
        "stop": conv.sep2,
    }
    response = requests.post(worker_addr + "/worker_generate_stream", headers=headers, json=pload, stream=True)
    for chunk in response.iter_lines(chunk_size=8192, decode_unicode=False, delimiter=b"\0"):
        if chunk:
            data_t = json.loads(chunk.decode("utf-8"))
            output = data_t["text"].split(conv.roles[1]+':')[-1]
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", default="./fine_eval/images")
    parser.add_argument("--input_data", default="./fine_eval/instructions.jsonl")
    parser.add_argument("--output_data", default="./fine_eval/pvit_answer.jsonl")
    args = parser.parse_args()
    with open(args.input_data, "r") as fp_in:
        with open(args.output_data, "w") as fp_out:
            for line in tqdm.tqdm(fp_in.readlines()):
                entry = json.loads(line)
                entry["output"] = send_one_message(
                    "<image>\n" + entry["instruction"],
                    os.path.join(args.image_path, entry["image"]),
                    worker_addr=os.getenv('MODEL_ADDR'),
                )
                fp_out.write(json.dumps(entry) + "\n")
