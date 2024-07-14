import sys

import numpy as np
import torchvision.transforms as T
from mask_utils import *
from utils import *


def rand_replace_text(text_list, level=0.01):
    rate = 0.5 * level
    output_list = mask_text(text_list, rate, method="replace", mode="random")
    return output_list


def target_replace_text(text_list, level=0.01):
    rate = 0.5 * level
    output_list = mask_text(text_list, rate, method="replace", mode="heat")
    return output_list


def rand_add_text(text_list, level=0.01):
    rate = 0.5 * level
    output_list = mask_text(text_list, rate, method="add", mode="random")
    return output_list


def target_add_text(text_list, level=0.01):
    rate = 0.5 * level
    output_list = mask_text(text_list, rate, method="add", mode="heat")
    return output_list


def sm_swap_text(text_list, level=0.2, fix=False):  # random replace q%
    rate = sample_float_level(level, 0)
    if fix:
        rate = level
    whole_text = "".join(text_list)
    amount = int(len(whole_text) * rate)
    whole_text = sm_process(whole_text, amount, "swap")
    output_list = whole_text.split("\n")
    output_list = [output + "\n" for output in output_list]
    return output_list


def sm_insert_text(text_list, level=0.2, fix=False):
    rate = sample_float_level(level, 0)
    if fix:
        rate = level
    whole_text = "".join(text_list)
    amount = int(len(whole_text) * rate)
    whole_text = sm_process(whole_text, amount, "insert")
    output_list = whole_text.split("\n")
    output_list = [output + "\n" for output in output_list]
    return output_list


def sm_patch_text(
    text_list, level=0.2, fix=False
):  # random replace q% consecutive characters
    rate = sample_float_level(level, 0)
    if fix:
        rate = level
    whole_text = "".join(text_list)
    amount = int(len(whole_text) * rate)
    whole_text = sm_process(whole_text, amount, "patch")
    output_list = whole_text.split("\n")
    output_list = [output + "\n" for output in output_list]
    return output_list


def synonym_replace_text(text_list, level=20):
    rate = sample_int_level(level, 0)
    whole_text = "".join(text_list)
    length = len(whole_text.split(" "))
    rate = min(int(length / 3), rate)
    whole_text = synonym_replacement(whole_text, rate)
    output_list = whole_text.split("\n")
    output_list = [output + "\n" for output in output_list]
    return output_list


def rand_del_text(text_list, level=0.01):
    rate = 0.5 * level
    whole_text = "".join(text_list)
    whole_text = random_deletion(whole_text, rate)
    output_list = whole_text.split("\n")
    output_list = [output + "\n" for output in output_list]
    return output_list


def aeda_punc_text(text_list, level=None, misc=None):
    from textaugment import AEDA

    whole_text = "".join(text_list)
    if misc == None:
        t = AEDA()
    else:
        t = misc
    try:
        whole_text = t.punct_insertion(whole_text)
    except ValueError as e:
        print(e)
        whole_text = whole_text
    output_list = whole_text.split("\n")
    output_list = [output + "\n" for output in output_list]
    return output_list


def translate_text(text_list, level=10, misc="en"):
    from textaugment import Translate

    rate = sample_int_level(level, 0)
    target_list = ["ru", "fr", "de", "el", "id", "it", "ja", "ko", "la", "pl"]
    whole_text = "".join(text_list)
    whole_text = remove_non_utf8(whole_text)

    t = Translate(src=misc, to=target_list[rate])
    try:
        whole_text = t.augment(whole_text)
    except Exception as e:
        print(e)
        whole_text = whole_text
    output_list = whole_text.split("\n")
    output_list = [output + "\n" for output in output_list]
    return output_list


def find_index(L, b):
    for i in range(len(L) - 1):
        if b > L[i] and b < L[i + 1]:
            return i
    return -1


def policy_aug_text(text_list, level="0.24-0.52-0.24", pool="PI-TI-TL"):
    mutator_list = [text_aug_dict[_mut] for _mut in pool.split("-")]
    probability_list = [float(_value) for _value in level.split("-")]
    probability_list = [sum(probability_list[:i]) for i in range(len(level))]
    randnum = np.random.random()
    index = find_index(probability_list, randnum)
    print(index, randnum)
    output_list = mutator_list[index](text_list)
    return output_list


text_aug = [
    rand_replace_text,
    rand_add_text,
    target_replace_text,
    target_add_text,
    rand_del_text,
    synonym_replace_text,
    aeda_punc_text,
    translate_text,
]

text_aug_dict = {
    "RR": rand_replace_text,
    "RI": rand_add_text,
    "TR": target_replace_text,
    "TI": target_add_text,
    "RD": rand_del_text,
    "SR": synonym_replace_text,
    "PI": aeda_punc_text,
    "TL": translate_text,
    "PL": policy_aug_text,
}
