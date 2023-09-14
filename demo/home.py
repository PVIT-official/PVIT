import re, os
import copy
import json
import yaml
import random
import streamlit as st
from PIL import Image
import requests
import base64
from io import BytesIO
import seaborn as sns

import json

import requests

import streamlit as st
from streamlit_chat import message as st_message
from streamlit_drawable_canvas import st_canvas

st.set_page_config(page_title="Model Chat", page_icon="🌍", layout="wide", initial_sidebar_state="collapsed")

col_img, col_chat = st.columns([1, 1])
with col_chat:
    with st.container():
        input_area = st.container()
        chatbox = st.container()


# ==================== Conversation =================== #
import dataclasses, re
from enum import auto, Enum
from typing import List


class SeparatorStyle(Enum):
    """Different separator style."""
    SINGLE = auto()
    TWO = auto()


def convert_region_tags(text):
    pattern = r'<Region>(.*?)<\/Region>'
    replaced_text = re.sub(pattern, lambda m: '&lt;Region&gt;' + m.group(1).replace('<', '&lt;').replace('>', '&gt;') + '&lt;/Region&gt;', text)
    return replaced_text


@dataclasses.dataclass
class Conversation:
    """A class that keeps all conversation history."""
    system: str
    roles: List[str]
    messages: List[List[str]]
    offset: int
    sep_style: SeparatorStyle = SeparatorStyle.SINGLE
    sep: str = "###"
    sep2: str = None
    version: str = "Unknown"

    skip_next: bool = False

    def get_prompt(self):
        if self.sep_style == SeparatorStyle.SINGLE:
            ret = self.system + self.sep
            for role, message in self.messages:
                if message:
                    if type(message) is tuple:
                        message, _, _ = message
                    ret += role + ": " + message + self.sep
                else:
                    ret += role + ":"
            return ret
        elif self.sep_style == SeparatorStyle.TWO:
            seps = [self.sep, self.sep2]
            ret = self.system + seps[0]
            for i, (role, message) in enumerate(self.messages):
                if message:
                    if type(message) is tuple:
                        message, _, _ = message
                    ret += role + ": " + message + seps[i % 2]
                else:
                    ret += role + ":"
            return ret
        else:
            raise ValueError(f"Invalid style: {self.sep_style}")

    def append_message(self, role, message):
        self.messages.append([role, message])

    def get_images(self, return_pil=False):
        images = []
        for i, (role, msg) in enumerate(self.messages[self.offset:]):
            if i % 2 == 0:
                if type(msg) is tuple:
                    import base64
                    from io import BytesIO
                    from PIL import Image
                    msg, image, image_process_mode = msg
                    if image_process_mode == "Pad":
                        def expand2square(pil_img, background_color=(122, 116, 104)):
                            width, height = pil_img.size
                            if width == height:
                                return pil_img
                            elif width > height:
                                result = Image.new(pil_img.mode, (width, width), background_color)
                                result.paste(pil_img, (0, (width - height) // 2))
                                return result
                            else:
                                result = Image.new(pil_img.mode, (height, height), background_color)
                                result.paste(pil_img, ((height - width) // 2, 0))
                                return result
                        image = expand2square(image)
                    elif image_process_mode == "Crop":
                        pass
                    elif image_process_mode == "Resize":
                        image = image.resize((224, 224))
                    else:
                        raise ValueError(f"Invalid image_process_mode: {image_process_mode}")
                    max_hw, min_hw = max(image.size), min(image.size)
                    aspect_ratio = max_hw / min_hw
                    max_len, min_len = 800, 400
                    shortest_edge = int(min(max_len / aspect_ratio, min_len, min_hw))
                    longest_edge = int(shortest_edge * aspect_ratio)
                    W, H = image.size
                    if H > W:
                        H, W = longest_edge, shortest_edge
                    else:
                        H, W = shortest_edge, longest_edge
                    image = image.resize((W, H))
                    if return_pil:
                        images.append(image)
                    else:
                        buffered = BytesIO()
                        image.save(buffered, format="JPEG")
                        img_b64_str = base64.b64encode(buffered.getvalue()).decode()
                        images.append(img_b64_str)
        return images

    def to_gradio_chatbot(self):
        ret = []
        for i, (role, msg) in enumerate(self.messages[self.offset:]):
            if i % 2 == 0:
                if type(msg) is tuple:
                    import base64
                    from io import BytesIO
                    msg, image, image_process_mode = msg
                    msg = convert_region_tags(msg)
                    max_hw, min_hw = max(image.size), min(image.size)
                    aspect_ratio = max_hw / min_hw
                    max_len, min_len = 800, 400
                    shortest_edge = int(min(max_len / aspect_ratio, min_len, min_hw))
                    longest_edge = int(shortest_edge * aspect_ratio)
                    W, H = image.size
                    if H > W:
                        H, W = longest_edge, shortest_edge
                    else:
                        H, W = shortest_edge, longest_edge
                    image = image.resize((W, H))
                    buffered = BytesIO()
                    image.save(buffered, format="JPEG")
                    img_b64_str = base64.b64encode(buffered.getvalue()).decode()
                    img_str = f'<img src="data:image/png;base64,{img_b64_str}" alt="user upload image" />'
                    msg = msg.replace('<image>', img_str)
                else:
                    msg = convert_region_tags(msg)
                ret.append([msg, None])
            else:
                if isinstance(msg, str):
                    msg = convert_region_tags(msg)
                ret[-1][-1] = msg
        return ret

    def copy(self):
        return Conversation(
            system=self.system,
            roles=self.roles,
            messages=[[x, y] for x, y in self.messages],
            offset=self.offset,
            sep_style=self.sep_style,
            sep=self.sep,
            sep2=self.sep2)

    def dict(self):
        if len(self.get_images()) > 0:
            return {
                "system": self.system,
                "roles": self.roles,
                "messages": [[x, y[0] if type(y) is tuple else y] for x, y in self.messages],
                "offset": self.offset,
                "sep": self.sep,
                "sep2": self.sep2,
            }
        return {
            "system": self.system,
            "roles": self.roles,
            "messages": self.messages,
            "offset": self.offset,
            "sep": self.sep,
            "sep2": self.sep2,
        }

conv_vicuna_v1_1 = Conversation(
    system="A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions.",
    roles=("USER", "ASSISTANT"),
    version="v1",
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.TWO,
    sep=" ",
    sep2="</s>",
)

default_conversation = conv_vicuna_v1_1

# ==================== Chat =================== #


def convert_bbox_to_region(bbox_xywh, image_width, image_height):
    bbox_x, bbox_y, bbox_w, bbox_h = bbox_xywh
    x1 = bbox_x
    y1 = bbox_y
    x2 = bbox_x + bbox_w
    y2 = bbox_y + bbox_h

    x1_normalized = x1 / image_width
    y1_normalized = y1 / image_height
    x2_normalized = x2 / image_width
    y2_normalized = y2 / image_height

    x1_norm = int(x1_normalized * 1000)
    y1_norm = int(y1_normalized * 1000)
    x2_norm = int(x2_normalized * 1000)
    y2_norm = int(y2_normalized * 1000)

    region_format = "<Region><L{}><L{}><L{}><L{}></Region>".format(x1_norm, y1_norm, x2_norm, y2_norm)
    return region_format


def load_config(config_fn, field='chat'):
    config = yaml.load(open(config_fn), Loader=yaml.Loader)
    return config[field]


def get_model_list():
    return ['PVIT_v1.0']


def change_model(model_name):
    if model_name != st.session_state.get('model_name', ''):
        st.session_state['model_name'] = 'PVIT_v1.0'
        st.session_state['model_addr'] = os.getenv('MODEL_ADDR')
        st.session_state['messages'] = []


def init_chat(image=None):
    st.session_state['image'] = image
    if 'input_message' not in st.session_state:
        st.session_state['input_message'] = ''
    if 'messages' not in st.session_state:
        st.session_state['messages'] = []


def clear_messages():
    st.session_state['messages'] = []
    st.session_state['input_message'] = ''


def encode_img(img):
    if isinstance(img, str):
        img = Image.open(img).convert('RGB')
        im_file = BytesIO()
        img.save(im_file, format="JPEG")
    elif isinstance(img, Image.Image):
        im_file = BytesIO()
        img.save(im_file, format="JPEG")
    else:
        im_file = img
    im_bytes = im_file.getvalue()
    im_b64 = base64.b64encode(im_bytes).decode()
    return im_b64


def send_one_message(message, max_new_tokens=32, temperature=0.0): 
    conv = default_conversation.copy()
    if len(st.session_state['messages']) == 0:
        if '<image>' not in message:
            message = '<image>\n' + message
    st.session_state['messages'].append([conv.roles[0], message])
    conv.messages = copy.deepcopy(st.session_state['messages'])
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    
    if 'canvas_result' in st.session_state:
        objects = st.session_state['canvas_result'].get('objects', [])
        for i, obj in enumerate(objects):
            prompt = prompt.replace(f'[REGION-{i}]', obj['bbox_label'])
    
    headers = {"User-Agent": "LLaVA Client"}
    pload = {
        "prompt": prompt,
        "images": [st.session_state['image']],
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "stop": conv.sep2,
    }
    print(prompt)
    response = requests.post(st.session_state['model_addr'] + "/worker_generate_stream", headers=headers,
            json=pload, stream=True)
    result = ""
    for chunk in response.iter_lines(chunk_size=8192, decode_unicode=False, delimiter=b"\0"):
        if chunk:
            data_t = json.loads(chunk.decode("utf-8"))
            output = data_t["text"].split(conv.roles[1]+':')[-1]
    result = output
    
    st.session_state['messages'].append([conv.roles[1], result])


# Customize Streamlit UI using CSS  # background-color: #eb5424;
st.markdown("""
<style>
div.stButton > button:first-child {
    background-color: #eb5424;
    color: white;
    font-size: 20px;
    font-weight: bold;
    border-radius: 0.5rem;
    padding: 0.5rem 1rem;
    border: none;
    box-shadow: 0 0.5rem 1rem rgba(0,0,0,0.15);
    width: 300 px;
    height: 42px;
    transition: all 0.2s ease-in-out;
} 
div.stButton > button:first-child:hover {
    transform: translateY(-3px);
    box-shadow: 0 1rem 2rem rgba(0,0,0,0.15);
}
div.stButton > button:first-child:active {
    transform: translateY(-1px);
    box-shadow: 0 0.5rem 1rem rgba(0,0,0,0.15);
}
div.stButton > button:focus:not(:focus-visible) {
    color: #FFFFFF;
}
@media only screen and (min-width: 768px) {
  /* For desktop: */
  div.stButton > button:first-child {
      background-color: #eb5424;
      color: white;
      font-size: 20px;
      font-weight: bold;
      border-radius: 0.5rem;
      padding: 0.5rem 1rem;
      border: none;
      box-shadow: 0 0.5rem 1rem rgba(0,0,0,0.15);
      width: 300 px;
      height: 42px;
      transition: all 0.2s ease-in-out;
      position: relative;
      bottom: -32px;
      right: 0px;
  } 
  div.stButton > button:first-child:hover {
      transform: translateY(-3px);
      box-shadow: 0 1rem 2rem rgba(0,0,0,0.15);
  }
  div.stButton > button:first-child:active {
      transform: translateY(-1px);
      box-shadow: 0 0.5rem 1rem rgba(0,0,0,0.15);
  }
  div.stButton > button:focus:not(:focus-visible) {
      color: #FFFFFF;
  }
  input {
      border-radius: 0.5rem;
      padding: 0.5rem 1rem;
      border: none;
      box-shadow: 0 0.5rem 1rem rgba(0,0,0,0.15);
      transition: all 0.2s ease-in-out;
      height: 40px;
  }
}
</style>
""", unsafe_allow_html=True)

# ==================== Draw Bounding Boxes =================== #

COLORS = sns.color_palette("Paired", n_colors=100).as_hex()
random.Random(32).shuffle(COLORS)

def update_annotation_states(canvas_result, ratio, img_size):
    for obj in canvas_result['objects']:
        top = obj["top"] * ratio
        left = obj["left"] * ratio
        width = obj["width"] * ratio
        height = obj["height"] * ratio
        obj['bbox_label'] = convert_bbox_to_region([left, top, width, height], img_size[0], img_size[1])
    st.session_state['canvas_result'] = canvas_result
    st.session_state['label_color'] = COLORS[len(st.session_state['canvas_result']['objects'])+1]
    
def init_canvas():
    if 'canvas_result' not in st.session_state:
        st.session_state['canvas_result'] = None
    if 'label_color' not in st.session_state:
        st.session_state['label_color'] = COLORS[0]

def input_message(msg):
    st.session_state['input_message'] = msg
    

def get_objects():
    canvas_result = st.session_state.get('canvas_result', {})
    if canvas_result is not None:
        objects = canvas_result.get('objects', [])
    else:
        objects = []
    return objects
    
def format_object_str(input_str):
    if 'canvas_result' in st.session_state:
        objects = st.session_state['canvas_result'].get('objects', [])
        for i, obj in enumerate(objects):
            input_str = input_str.replace(f'[REGION-{i}]', obj['bbox_label'])
    return input_str
        
# select model
model_list = get_model_list()
with col_img:
    model_name = st.selectbox(
        'Choose a model to chat with',
        model_list
    )
change_model(model_name)

css = ''
# upload image
with col_img:
    image = st.file_uploader("Chat with Image", type=["png", "jpg", "jpeg"], on_change=clear_messages)
    img_fn = image.name if image is not None else None
if image:
    init_chat(encode_img(image))
    init_canvas()
    
    img = Image.open(image).convert('RGB')
    
    width = 700
    height = round(width * img.size[1] * 1.0 / img.size[0])
    ratio = img.size[0] / width
    
    with st.sidebar:
        max_new_tokens = st.number_input('max_new_tokens', min_value=1, max_value=1024, value=128)
        temperature = st.number_input('temperature', min_value=0.0, max_value=1.0, value=0.0)
        drawing_mode = st.selectbox("Drawing tool:", ("rect", "point", "line", "circle"))
        drawing_mode = "transform" if st.checkbox("Move ROIs", False) else drawing_mode
        stroke_width = st.slider("Stroke width: ", 1, 25, 3)
            
    with col_img:
        canvas_result = st_canvas(
            fill_color=st.session_state['label_color'] + "77",
            stroke_width=stroke_width,
            stroke_color=st.session_state['label_color'] + "77",
            background_color="#eee",
            background_image=Image.open(image) if image else None,
            update_streamlit=True,
            width=width,
            height=height,
            drawing_mode=drawing_mode,
            point_display_radius=3 if drawing_mode == 'point' else 0,
            key="canvas",
        )
    
    if canvas_result.json_data is not None:
        update_annotation_states(canvas_result.json_data, ratio, img.size)
    
    with input_area:    
        col3, col4, col5 = st.columns([5, 1, 1])
        
        with col3: 
            message = st.text_input('User', key="input_message")
        
        with col4:
            submit_btn = st.button(label='submit')
        with col5:
            clear_btn = st.button(label='clear', on_click=clear_messages)
        
        objects = get_objects()
        
        if len(objects):
            bbox_cols = st.columns([1 for _ in range(len(objects))])
            
            def on_bbox_button_click(str):
                def f():
                    st.session_state['input_message'] += str
                return f
            
            for i, (obj, bbox_col) in enumerate(zip(objects, bbox_cols)):
                with bbox_col:
                    st.button(label=f'Region-{i}', on_click=on_bbox_button_click(f'[REGION-{i}]'))
                    css += f"#root > div:nth-child(1) > div.withScreencast > div > div > div > section.main.css-uf99v8.ea3mdgi5 > div.block-container.css-z5fcl4.ea3mdgi4 > div:nth-child(1) > div > div.css-ocqkz7.e1f1d6gn3 > div:nth-child(2) > div:nth-child(1) > div > div:nth-child(1) > div > div:nth-child(1) > div > div:nth-child(2) > div:nth-child({i+1}) > div:nth-child(1) > div > div > div > button {{background-color:{obj['stroke'][:7]}; bottom: 0px}} \n" + '\n'
    if submit_btn:
        send_one_message(message, max_new_tokens=max_new_tokens, temperature=temperature)
    
    for i, (role, msg) in enumerate(st.session_state['messages']):
        with chatbox:
            st_message(msg.lstrip('<image>\n'), is_user=(role==default_conversation.roles[0]), key=f'{i}-{msg}')

st.markdown("<style>\n" + css + "</style>", unsafe_allow_html=True)
