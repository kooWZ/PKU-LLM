import os
import csv
import sys
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from PIL import Image
from torchvision.utils import save_image
from typing import List

sys.path.append("../../../models/")
from lavis.models import load_model_and_preprocess as lavis_load_model_and_preprocess
from blip_utils import visual_attacker as blip_attacker

from llava_llama_2.utils import get_model as llava_get_model
from llava_llama_2_utils import visual_attacker as llava_attacker
from llava_llama_2_utils import prompt_wrapper as llava_prompt_wrapper

from minigpt4.common.config import Config
from minigpt4.common.dist_utils import get_rank
from minigpt4.common.registry import registry
from minigpt_utils import (
    prompt_wrapper as minigpt4_prompt_wrapper,
    visual_attacker as minigpt4_visual_attacker,
)


class Args:
    def __init__(
        self,
        save_dir,
        options=None,
        cfg_path="",
    ):
        self.save_dir = save_dir
        self.options = options
        self.cfg_path = cfg_path


def blip_attack(
    save_dir: str,
    targets: List[str],
    template_img: str,
    constrained: bool,
    n_iters: int,
    alpha: int,
    eps: int,
):
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    print(">>> Initializing Models")
    model, vis_processor, _ = lavis_load_model_and_preprocess(
        name="blip2_vicuna_instruct",
        model_type="vicuna13b",
        is_eval=True,
        device=device,
    )
    model.eval()
    print("[Initialization Finished]\n")
    my_attacker = blip_attacker.Attacker(
        Args(save_dir), model, targets, device=model.device, is_rtp=False
    )
    img = Image.open(template_img).convert("RGB")
    img = vis_processor["eval"](img).unsqueeze(0).to(device)
    if not constrained:
        adv_img_prompt = my_attacker.attack_unconstrained(
            img=img, batch_size=8, num_iter=n_iters, alpha=alpha / 255
        )

    else:
        adv_img_prompt = my_attacker.attack_constrained(
            img=img,
            batch_size=8,
            num_iter=n_iters,
            alpha=alpha / 255,
            epsilon=eps / 255,
        )

    save_img_path = f"{save_dir}/bad_prompt.bmp"
    save_image(adv_img_prompt, save_img_path)
    return save_img_path


def llava_attack(
    llava_model_path: str,
    llava_model_base: str,
    save_dir: str,
    targets: List[str],
    template_img: str,
    constrained: bool,
    n_iters: int,
    alpha: int,
    eps: int,
):
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    print(">>> Initializing Models")
    tokenizer, model, image_processor, model_name = llava_get_model(
        llava_model_path, llava_model_base
    )
    model.eval()
    print("[Initialization Finished]\n")
    my_attacker = llava_attacker.Attacker(
        Args(save_dir),
        model,
        tokenizer,
        targets,
        device=model.device,
        image_processor=image_processor,
    )
    image = image = Image.open(template_img).convert("RGB")
    image = image_processor.preprocess(image, return_tensors="pt")[
        "pixel_values"
    ].cuda()

    text_prompt_template = llava_prompt_wrapper.prepare_text_prompt("")

    if not constrained:
        adv_img_prompt = my_attacker.attack_unconstrained(
            text_prompt_template,
            img=img,
            batch_size=8,
            num_iter=n_iters,
            alpha=alpha / 255,
        )

    else:
        adv_img_prompt = my_attacker.attack_constrained(
            text_prompt_template,
            img=img,
            batch_size=8,
            num_iter=n_iters,
            alpha=alpha / 255,
            epsilon=eps / 255,
        )

    save_img_path = f"{save_dir}/bad_prompt.bmp"
    save_image(adv_img_prompt, save_img_path)
    return save_img_path


def minigpt4_attack(
    cfg_path: str,
    save_dir: str,
    targets: List[str],
    template_img: str,
    constrained: bool,
    n_iters: int,
    alpha: int,
    eps: int,
):
    arg = Args(save_dir, cfg_path=cfg_path)
    cfg = Config(arg)

    print(">>> Initializing Models")

    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

    model_config = cfg.model_cfg
    model_config.device_8bit = 0
    model_cls = registry.get_model_class(model_config.arch)
    model = model_cls.from_config(model_config).to("cuda:0")
    model.eval()

    vis_processor_cfg = cfg.datasets_cfg.cc_sbu_align.vis_processor.train
    vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(
        vis_processor_cfg
    )

    print("[Initialization Finished]\n")
    my_attacker = minigpt4_visual_attacker.Attacker(
        arg, model, targets, device=model.device, is_rtp=False
    )

    img = Image.open(template_img).convert("RGB")
    img = vis_processor(img).unsqueeze(0).to(model.device)

    text_prompt_template = minigpt4_prompt_wrapper.minigpt4_chatbot_prompt_no_text_input

    if not constrained:
        adv_img_prompt = my_attacker.attack_unconstrained(
            text_prompt_template,
            img=img,
            batch_size=8,
            num_iter=n_iters,
            alpha=alpha / 255,
        )

    else:
        adv_img_prompt = my_attacker.attack_constrained(
            text_prompt_template,
            img=img,
            batch_size=8,
            num_iter=n_iters,
            alpha=alpha / 255,
            epsilon=eps / 255,
        )

    save_img_path = f"{save_dir}/bad_prompt.bmp"
    save_image(adv_img_prompt, save_img_path)
    return save_img_path


def gen_image(
    model: str,
    save_dir: str,
    template_img: str,
    targets: List[str] = None,
    constrained: bool = True,
    n_iters: int = 500,
    alpha: int = 1,
    eps: int = 32,
    llava_model_path: str = "ckpts/llava_llama_2_13b_chat_freeze",
    llava_model_base: str = "",
    minigpt4_cfg_path: str = "",
    corpus_file: str = "../../../dataset/derogatory_corpus.csv",
):
    if targets is None or targets == []:
        file = open(corpus_file, "r")
        data = list(csv.reader(file, delimiter=","))
        file.close()
        targets = []
        num = len(data)
        for i in range(num):
            targets.append(data[i][0])

    if model == "minigpt4":
        return minigpt4_attack(
            minigpt4_cfg_path,
            save_dir,
            targets,
            template_img,
            constrained,
            n_iters,
            alpha,
            eps,
        )
    elif model == "llava":
        return llava_attack(
            llava_model_path,
            llava_model_base,
            save_dir,
            targets,
            template_img,
            constrained,
            n_iters,
            alpha,
            eps,
        )
    elif model == "blip":
        return blip_attack(
            save_dir, targets, template_img, constrained, n_iters, alpha, eps
        )
    else:
        raise NotImplementedError
