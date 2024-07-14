import argparse
import os
import random
import sys

import gradio as gr
import numpy as np
import torch
import torch.backends.cudnn as cudnn

sys.path.append("../../models")
from minigpt4.common.config import Config
from minigpt4.common.dist_utils import get_rank
from minigpt4.common.registry import registry
from minigpt4.conversation.conversation import CONV_VISION, Chat
from minigpt4.datasets.builders import *
from minigpt4.models import *
from minigpt4.processors import *
from minigpt4.runners import *
from minigpt4.tasks import *
from PIL import Image
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--cfg_path", default="./utils/minigpt4_eval.yaml")
    parser.add_argument("--options", nargs="+")
    parser.add_argument("--gpu_id", type=str, default="0")
    args = parser.parse_args()
    return args


def setup_seeds(config):
    seed = config.run_cfg.seed + get_rank()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True


def initialize_model():
    print("Initializing Chat")
    args = parse_args()
    cfg = Config(args)
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{args.gpu_id}"
    model_config = cfg.model_cfg
    model_config.device_8bit = 0
    model_cls = registry.get_model_class(model_config.arch)
    model = model_cls.from_config(model_config).to("cuda:0")

    vis_processor_cfg = cfg.datasets_cfg.cc_sbu_align.vis_processor.train
    vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(
        vis_processor_cfg
    )
    chat = Chat(model, vis_processor, device="cuda:{}".format(args.gpu_id))
    print("Initialization Finished")
    return vis_processor, chat, model


def model_inference(vis_processor, chat, model, prompts_eval):
    def upload_img(img):
        chat_state = CONV_VISION.copy()
        img_list = []
        chat.upload_img(img, chat_state, img_list)
        return chat_state, img_list

    def ask(user_message, chat_state):
        chat.ask(user_message, chat_state)
        return chat_state

    def answer(chat_state, img_list, num_beams=1, temperature=1.0):
        llm_message = chat.answer(
            conv=chat_state,
            img_list=img_list,
            num_beams=num_beams,
            temperature=temperature,
            max_new_tokens=300,
            max_length=2000,
        )[0]

        return llm_message, chat_state, img_list

    image_path = prompts_eval[1]
    question = prompts_eval[0]
    img = Image.open(image_path).convert("RGB")
    img = vis_processor(img).unsqueeze(0).to(model.device)
    chat_state, img_list = upload_img(img)

    chat_state = ask(question, chat_state)
    llm_message, chat_state, img_list = answer(chat_state, img_list)
    return llm_message
