import argparse
import os
import random
import re
import uuid
from collections import defaultdict

import nltk
import numpy as np
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import sent_tokenize, word_tokenize
from PIL import Image, ImageDraw

alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!\"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"


def get_synonyms(word):
    synonyms = set()

    for syn in wordnet.synsets(word):
        for l in syn.lemmas():
            synonym = l.name().replace("_", " ").replace("-", " ").lower()
            synonym = "".join(
                [char for char in synonym if char in " qwertyuiopasdfghjklzxcvbnm"]
            )
            synonyms.add(synonym)

    if word in synonyms:
        synonyms.remove(word)

    return list(synonyms)


def synonym_replacement(words, n):
    stop_words = set(stopwords.words("english"))
    words = words.split()

    new_words = words.copy()
    random_word_list = list(set([word for word in words if word not in stop_words]))
    random.shuffle(random_word_list)
    num_replaced = 0

    for random_word in random_word_list:
        synonyms = get_synonyms(random_word)

        if len(synonyms) >= 1:
            synonym = random.choice(list(synonyms))
            new_words = [synonym if word == random_word else word for word in new_words]
            num_replaced += 1

        if num_replaced >= n:  # only replace up to n words
            break

    sentence = " ".join(new_words)

    return sentence


def random_deletion(sentence, p):
    # randomly delete words with probability p
    new_sentence = []
    count = 0
    for c in range(len(sentence)):
        if count > 0:
            count -= 1
            continue
        cha = sentence[c]
        r = random.uniform(0, 1)
        if r > p:
            new_sentence.append(cha)
        else:
            count = 5

    sentence = "".join(new_sentence)

    return sentence


def insert_string_at_multiple_positions(text, string_to_insert, positions):
    # Sort the positions in reverse order to prevent index shifting during insertions
    positions.sort(reverse=True)

    text_list = list(text)

    for pos in positions:
        if string_to_insert == None:
            tmp = random.choice(alphabet)
        else:
            tmp = string_to_insert
        text_list[pos:pos] = tmp

    return "".join(text_list)


def replace_at_index(text, index, replacement):
    if index < 0 or index > len(text):
        return text 

    if replacement == None:
        replacement = random.choice(alphabet)
    new_text = text[:index] + replacement + text[index + len(replacement) :]
    return new_text


def get_image_path(path):
    tmplist = []
    if os.path.exists(path):
        files = os.listdir(path)
        for file1 in files:
            m = os.path.join(path, file1)
            if os.path.isfile(m) and (".png" in file1 or ".jpg" in file1):
                tmplist.append(m)
    return tmplist


def get_number(string):
    return int(os.path.basename(string).split("-")[0])


def get_txt_path(path):
    tmplist = []
    if os.path.exists(path):
        files = os.listdir(path)
        for file1 in files:
            m = os.path.join(path, file1)
            if os.path.isfile(m):
                tmplist.append(m)
    tmplist = sorted(tmplist, key=get_number)
    return tmplist


def load_image(method, path):
    image = Image.open(path)
    return method, image


def load_rgb(rgb):
    rgb_list = rgb.split("-")
    rgb_tuple = (int(rgb_list[0]), int(rgb_list[1]), int(rgb_list[2]))
    return rgb_tuple


def load_size(size):
    size_tuple = (int(size.split("-")[0]), int(size.split("-")[1]))
    return size_tuple


def break_lines(text, max_len=30):
    text = text.replace("_", " ")
    count = 0
    text_list = []
    tmp = ""
    for i in text:
        if count > max_len and i == " ":
            text_list.append(tmp)
            tmp = ""
            count = 0
        else:
            tmp += i
            count += 1
    if tmp != "":
        text_list.append(tmp)
    return text_list


def load_position(position, size, mask_size):
    if position == "rand":
        pos1 = np.random.randint(0, int(size[0] - mask_size[0]))
        pos2 = np.random.randint(0, int(size[1] - mask_size[1]))  # down left
        return (pos1, pos2)
    else:
        size_tuple = (
            min(int(position.split("-")[0]), size[0]),
            min(int(position.split("-")[1]), size[1]),
        )
        return size_tuple


def generate_mask_pil(image, mask_type, mask_size, position):
    draw = ImageDraw.Draw(image)
    if mask_type == "r":
        rect_x, rect_y, rect_width, rect_height = (
            position[0],
            position[1],
            mask_size[0],
            mask_size[1],
        )
        draw.rectangle(
            (rect_x, rect_y, rect_x + rect_width, rect_y + rect_height), fill="black"
        )
        output_image = image
    else:
        print("not support mask!!")
        os._exit(0)
    return output_image


def important_sentences(text, n=1, rate=0.0, check_q=True):
    if check_q:
        q_list = []
    sentences = sent_tokenize(text)
    words = [word_tokenize(sentence.lower()) for sentence in sentences]

    stop_words = set(stopwords.words("english"))
    filtered_words = [
        [word for word in sentence_words if word.isalnum() and word not in stop_words]
        for sentence_words in words
    ]

    word_freq = defaultdict(int)
    for sentence in filtered_words:
        for word in sentence:
            word_freq[word] += 1

    sentence_scores = [0] * len(sentences)
    for i, sentence in enumerate(filtered_words):
        for word in sentence:
            sentence_scores[i] += word_freq[word]
        if "prompt" in sentence or "question" in sentence or "?" in sentences[i]:
            q_list.append([sentences[i], sentence_scores[i]])

    if len(sentence_scores) > n:
        if np.random.random() > rate:
            most_important = sorted(
                ((scores, i) for i, scores in enumerate(sentence_scores)), reverse=True
            )[:n]
        else:
            most_important = sorted(
                ((scores, i) for i, scores in enumerate(sentence_scores))
            )[:n]
    else:
        return None, None
    most_important_sentences = [sentences[i] for _, i in most_important]
    sorted_q = sorted(q_list, key=lambda x: x[1])[:n]
    return most_important_sentences, sorted_q


def find_string_index(long_string, substring):
    start_index = long_string.find(substring)
    end_index = start_index + len(substring) - 1
    return start_index, end_index


def random_mask_text(origin_text, method, rate):
    temp_j = []
    index_list = []
    for j in range(len(origin_text)):
        if j in temp_j:
            continue
        temp_j = []
        if np.random.random() < rate:
            if method == "replace":
                origin_text = replace_at_index(origin_text, j, "[Mask]")
                temp_j += list(range(j + 1, j + 6))
            elif method == "add":
                index_list.append(j)
                temp_j += list(range(j + 1, j + 6))
    if method == "add":
        origin_text = insert_string_at_multiple_positions(
            origin_text, "[Mask]", index_list
        )
    return origin_text


def heat_mutate(whole_text, method, rate, range_list):
    temp_j = []
    index_list = []
    for j in range(len(whole_text)):
        if j in temp_j:
            continue
        temp_j = []
        tmp_rate = rate
        for rg in range_list:
            if j < rg[1] and j >= rg[0]:
                tmp_rate = rg[-1] * rate
                break
        if np.random.random() < tmp_rate:
            if method == "replace":
                while "\n" in whole_text[j : j + 6]:
                    j += 1
                    temp_j += list(range(j + 1, j + 2))
                whole_text = replace_at_index(whole_text, j, "[Mask]")
                temp_j += list(range(j + 1, j + 6))
            elif method == "add":
                if j > 0 and whole_text[j - 1 : j + 1] == "\n":
                    j += 1
                index_list.append(j + 1)
                temp_j += list(range(j + 1, j + 6))
    if method == "add":
        whole_text = insert_string_at_multiple_positions(
            whole_text, "[Mask]", index_list
        )
    return whole_text.split("\n")


def mask_text(
    text_list, level, max=10, ensure=True, method="replace", mode="random", times=5
):
    output_list = []
    if mode == "random":
        for i in range(len(text_list)):
            pure_text_list = [line.strip() for line in text_list]
            raw_output_text = random_mask_text(
                pure_text_list[i], method=method, rate=level
            )
            output_list.append(raw_output_text + "\n")
    elif mode == "heat":
        whole_text = " ".join(text_list)
        important_list, question_list = important_sentences(whole_text, n=3)
        assert important_list != []
        if important_list == None:
            for i in range(len(text_list)):
                pure_text_list = [line.strip() for line in text_list]
                raw_output_text = random_mask_text(
                    pure_text_list[i], method=method, rate=level
                )
                output_list.append(raw_output_text + "\n")
        else:
            important_list += [
                q[0] for q in question_list if q[0] not in important_list
            ]
            range_list = []
            for i in range(len(important_list)):
                sentence = important_list[i]
                tmp_start, tmp_end = find_string_index(whole_text, sentence)
                range_list.append([tmp_start, tmp_end, times])  # int(10/(i+1))
            output_list = heat_mutate(
                whole_text, method=method, rate=level, range_list=range_list
            )
            output_list = [line + "\n" for line in output_list]
    else:
        raise NotImplementedError
    return output_list


def sm_process(text, amount, method="swap"):
    if method == "swap" or method == "insert":
        index_list = list(range(len(text)))
        select_list = random.sample(index_list, amount)
        if method == "insert":
            text = insert_string_at_multiple_positions(text, None, select_list)
        else:
            for i in select_list:
                text = replace_at_index(text, i, None)
    else:
        index_list = list(range(len(text) - amount))
        select_index = random.sample(index_list, 1)[0]
        tmp_string = ""
        for i in range(amount):
            tmp_string += random.sample(alphabet, 1)[0]
        text = replace_at_index(text, select_index, tmp_string)
    return text
