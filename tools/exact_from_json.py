#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time    : 2023/3/13 8:52
# @Author  : kingsley kwong
# @Site    :
# @File    : exact_from_json.py
# @Software: 
# @Function:

import ujson
from datasets import load_dataset

def format_gpt2_raw():
    lines = []
    with open(r'../dataset/format/raw/baike_qa_train_format.raw', 'w', encoding='utf-8') as wf:
        with open(r'../dataset/raw/baike_qa_train.json', 'r', encoding='utf-8') as f:
            for line in f.readlines():
                json_line = ujson.loads(line)
                title = json_line['title']
                answer = json_line['answer'].replace(r'\r', '').replace(r'\n', '').replace('\n', '').replace('\r', '')
                lines.append(''.join([title, answer]))
        wf.writelines(lines)


def format_gpt2_json():
    lines = []
    with open(r'../dataset/format/json/baike_qa_valid_format.json', 'w+', encoding='utf-8') as wf:
        with open(r'../dataset/raw/baike_qa_valid.json', 'r', encoding='utf-8') as f:
            for line in f.readlines():
                json_line = ujson.loads(line)
                for key in json_line:
                    json_line[key] = json_line[key]\
                        .replace(r'\r', '')\
                        .replace(r'\n', '')\
                        .replace('\n', '')\
                        .replace('\r', '')\
                        .replace(' ', '')\
                        .replace('　', '')\
                        .strip()
                lines.append(ujson.dumps(json_line))
            wf.write('\n'.join(lines))

format_gpt2_json()