import argparse
import copy
import os
import pickle
import sys
import uuid

import numpy as np
import openai
import spacy
from tqdm import trange

from utils.text_augmentations import *
from utils.mask_utils import *
from utils.utils import *


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mutator",
        default="PL",
        type=str,
        help="Random Replacement(RR),Random Insertion(RI),Targeted Replacement(TR),Targeted Insertion(TI),Random Deletion(RD),Synonym Replacement(SR),Punctuation Insertion(PI),Translation(TL),Policy(PL)",
    )
    parser.add_argument(
        "--serial_num",
        type=str,
        help="the serial number of the data under test in the dataset.",
    )
    parser.add_argument(
        "--path", type=str, help="dataset path"
    )
    parser.add_argument(
        "--variant_save_dir",
        type=str,
        help="dir to save the modify results",
    )
    parser.add_argument(
        "--response_save_dir",
        type=str,
        help="dir to save the modify results",
    )
    parser.add_argument(
        "--number", default="8", type=str, help="number of generated variants"
    )
    parser.add_argument(
        "--threshold", default=0.02, type=str, help="Threshold of divergence"
    )
    args = parser.parse_args()

    number = int(args.number)

    if not os.path.exists(args.variant_save_dir):
        os.makedirs(args.variant_save_dir)

    # Step1: mask input
    dataset_path = args.path
    serial_num = int(args.serial_num)

    # load dataset
    with open(dataset_path, "rb") as f:
        dataset = pickle.load(f)
    key_path = dataset_path.replace("dataset.pkl", "dataset-key.pkl")
    with open(key_path, "rb") as f:
        dataset_key = pickle.load(f)

    for i in range(number):
        tmp_method = text_aug_dict[args.mutator]

        uid_name = str(uuid.uuid4())[:4]
        target_dir = args.variant_save_dir
        
        origin_text = copy.deepcopy(dataset[serial_num])  #
        key_list = dataset_key[serial_num]  #

        if isinstance(origin_text, str):
            output_result = tmp_method(text_list=[origin_text])
            target_path = os.path.join(
                target_dir, str(uuid.uuid4())[:6] + f"-{args.mutator}"
            )
            f = open(target_path, "w")
            f.writelines(output_result)
            f.close()
        else:
            for i in range(len(origin_text)):
                origin_text[i]["content"] = tmp_method(
                    text_list=[origin_text[i]["content"]]
                )
            output_result = origin_text
            target_path = os.path.join(
                target_dir, str(uuid.uuid4())[:6] + f"-{args.mutator}.pkl"
            )
            with open(target_path, "wb") as f:
                pickle.dump(output_result, f)

    # Step2: query_model
    variant_list, name_list = load_dirs(target_dir)
    new_save_dir = args.response_save_dir
    if not os.path.exists(new_save_dir):
        os.makedirs(new_save_dir)
    try:
        variant_list = [r"\n".join(i) for i in variant_list]
    except:
        pass
    name_list, variant_list = (
        list(t) for t in zip(*sorted(zip(name_list, variant_list)))
    )
    for j in range(len(variant_list)):
        prompt = variant_list[j]
        messages = None
        if not isinstance(prompt, str):
            messages = prompt
        save_name = name_list[j]
        existing_response = [i for i in os.listdir(new_save_dir) if ".png" not in i]
        if len(existing_response) >= number:
            continue

        new_save_path = os.path.join(new_save_dir, save_name)
        if not os.path.exists(new_save_path):
            try:
                param = {}
                if "Parameters" in key_list[0]:
                    param = key_list[1]
                res_content = query_gpt(
                    "gpt-3.5-turbo-1106",
                    prompt,
                    sleep=3,
                    messages=messages,
                    param=param,
                )
            except openai.BadRequestError:
                res_content = "I cannot assist with that!"
                print(f"Blocked in {new_save_path}")
            except Exception as e:
                print(e)
                res_content = "No response!"
                print(f"NO response in {new_save_path}")

            f = open(new_save_path, "w")
            f.writelines(res_content)
            f.close()

    # Step3: divergence & detect
    diver_save_path = os.path.join(
        args.response_save_dir, f"diver_result-{args.number}.pkl"
    )
    metric = spacy.load("en_core_web_md")
    avail_dir = args.response_save_dir
    check_list = os.listdir(avail_dir)
    check_list = [os.path.join(avail_dir, check) for check in check_list]
    output_list = read_file_list(check_list)
    max_div, jailbreak_keywords = update_divergence(
        output_list,
        os.path.basename(args.path),
        avail_dir,
        select_number=number,
        metric=metric,
    )

    detection_result = detect_attack(max_div, jailbreak_keywords, args.threshold)
    if detection_result:
        print("The Input is an Attack Query!!")
    else:
        print("The Input is a Benign Query!!")
