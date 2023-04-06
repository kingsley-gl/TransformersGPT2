#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time    : 2023/3/9 15:22
# @Author  : kingsley kwong
# @Site    :
# @File    : gpt2.py
# @Software: 
# @Function:


from pydantic import BaseModel
from pydantic.types import List, Dict, OptionalInt, OptionalIntFloat


class GPT2ModelConfig(BaseModel):
    config_json: str = 'json/model/gpt2.json'
    vocab: str = 'vocab/vocab.txt'
    lr: float = 2.6e-5
    eps: float = 1.9e-09
    warmup_step: int = 40
    epoch: int = 100
    max_grad_norm: float = 2.0
    batch_size: int = 4
    num_worker: int = 4

    def __getitem__(self, item):
        return getattr(self, item)
