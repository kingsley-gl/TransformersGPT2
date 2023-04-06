#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time    : 2023/3/10 16:32
# @Author  : kingsley kwong
# @Site    :
# @File    : train_bert_token.py
# @Software: 
# @Function:

import argparse
from tokenizers import Tokenizer
from tokenizers.trainers import WordPieceTrainer
from tokenizers.models import WordPiece
from tokenizers.normalizers import NFD, Lowercase, StripAccents
from tokenizers import normalizers
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
import os


def build(parser_args: argparse.Namespace):
    bert_tokenizer = Tokenizer(WordPiece(unk_token="[UNK]"))
    bert_tokenizer.normalizer = normalizers.Sequence([NFD(), Lowercase(), StripAccents()])
    bert_tokenizer.pre_tokenizer = Whitespace()
    bert_tokenizer.post_processor = TemplateProcessing(
        single="[CLS] $A [SEP]",
        pair="[CLS] $A [SEP] $B:1 [SEP]:1",
        special_tokens=[
            ("[CLS]", 1),
            ("[SEP]", 2),
        ],
    )

    trainer = WordPieceTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])
    files = [r'/'.join([parser_args.dataset, raw]) for raw in os.listdir(parser_args.dataset)]
    bert_tokenizer.train(files, trainer)
    bert_tokenizer.save(parser_args.output)


def valid(parser_args: argparse.Namespace):
    tokenizer = Tokenizer.from_file(parser_args.output)
    out = tokenizer.encode('今天我们去哪里玩，去动物园好吗')
    print(out.tokens)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--vocab_size', type=int, default=50257, help='词汇大小')
    parser.add_argument('--dataset', type=str, default='../dataset/format/raw', help='原始训练语料')
    parser.add_argument('--output', type=str, default='../json/tokenizer/my-tokenizer1.json', help='分词器输出目录')
    args = parser.parse_args()
    build(args)
    valid(args)
