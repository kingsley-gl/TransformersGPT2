#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time    : 2023/3/9 15:50
# @Author  : kingsley kwong
# @Site    :
# @File    : train.py
# @Software: 
# @Function:

import argparse
import traceback
import random
import torch
from config.gpt2 import GPT2ModelConfig
from transformers import GPT2Config, GPT2LMHeadModel, AdamW, get_scheduler, DataCollatorWithPadding
from transformers import BertTokenizer, BertTokenizerFast, PreTrainedTokenizerFast
from datasets import load_dataset
from datasets import disable_caching
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import evaluate
from accelerate import Accelerator
from apex import amp
from torch.nn.utils.rnn import pad_sequence

# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
disable_caching()
seed = random.randint(1, 99)


def collate_fn(batch):
    input_ids = pad_sequence(batch, batch_first=True, padding_value=0)
    labels = pad_sequence(batch, batch_first=True, padding_value=-100)
    return input_ids, labels


class Trainer(object):
    def __init__(self, parser_args: argparse.Namespace):
        self.args = parser_args
        self.conf = GPT2ModelConfig()
        if parser_args.cuda:
            self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
            self.conf.num_worker = 2
            self.conf.batch_size = 1
        elif parser_args.npu:
            try:
                import torch_npu
            except ImportError:
                raise ImportError
            self.device = torch.device('npu') if torch.npu.is_available() else torch.device('cpu')
        else:
            self.device = torch.device('cpu')
        print(f'train with device {self.device}')

        raw_dataset = load_dataset('json', name='gpt2', data_files=parser_args.data_path, split='train',
                                   cache_dir='./.cache')
        raw_dataset = raw_dataset.shuffle(seed=seed).train_test_split(test_size=0.1)

        # 分词器
        if parser_args.tokenizer is None:
            self.tokenizer = BertTokenizerFast(vocab_file=self.conf.vocab)
        else:
            self.tokenizer = PreTrainedTokenizerFast(tokenizer_file=parser_args.tokenizer, sep_token="[SEP]",
                                                     unk_token="[UNK]", pad_token="[PAD]", cls_token="[CLS]",
                                                     mask_token="[MASK]")
        # preprocess
        self.train_dataset = raw_dataset['train'].shuffle(seed=seed)
        self.test_dataset = raw_dataset['test'].shuffle(seed=seed)

        tokenizer_func = lambda example: self.tokenizer(example['title'], example['answer'], truncation=True)
        print('preprocess')
        self.train_dataset = self.train_dataset.map(tokenizer_func, batched=True, num_proc=self.conf.num_worker)
        self.test_dataset = self.test_dataset.map(tokenizer_func, batched=True, num_proc=self.conf.num_worker)
        self.train_dataset = self.train_dataset.remove_columns(['qid', 'title', 'answer', 'category', 'desc'])
        self.test_dataset = self.test_dataset.remove_columns(['qid', 'title', 'answer', 'category', 'desc'])
        # data_collator = DataCollatorWithPadding(self.tokenizer)
        self.train_dataset.set_format('torch')
        self.test_dataset.set_format('torch')
        print('filtered')
        self.train_dataset = self.train_dataset.filter(lambda x: x['input_ids'].shape[0] < 1024,
                                                       num_proc=self.conf.num_worker)
        self.test_dataset = self.test_dataset.filter(lambda x: x['input_ids'].shape[0] < 1024,
                                                     num_proc=self.conf.num_worker)
        self.train_dataloader = DataLoader(self.train_dataset['input_ids'], shuffle=True,
                                           batch_size=self.conf.batch_size, collate_fn=collate_fn, drop_last=True)
        self.eval_dataloader = DataLoader(self.test_dataset['input_ids'], batch_size=self.conf.batch_size,
                                          collate_fn=collate_fn,
                                          drop_last=True)
        print('set')
        #
        self.total_step = len(self.train_dataloader) * self.conf.epoch
        self.model = GPT2LMHeadModel.from_pretrained('gpt2',
                                                     # cache_dir='.cache'
                                                     )
        self.model = self.model.half().npu()
        self.optimizer = AdamW(self.model.parameters(), lr=self.conf.lr, eps=self.conf.eps, correct_bias=True)
        self.scheduler = get_scheduler(name='linear', optimizer=self.optimizer,
                                       num_warmup_steps=self.conf.warmup_step,
                                       num_training_steps=self.total_step)
        self.model.to(self.device)
        self.model.train()
        if parser_args.fp16:
            self.model, self.optimizer = amp.initialize(self.model, optimizers=self.optimizer, opt_level='O1')
        if parser_args.accelerate:
            self.accelerator = Accelerator()
            self.train_dataloader, self.model, self.optimizer, self.scheduler = self.accelerator.prepare(
                self.train_dataloader, self.model, self.optimizer, self.scheduler)

    def train(self):
        print('start trainning model')
        progress_bar = tqdm(range(self.total_step))
        # torch_npu.npu.set_aoe('./dump')
        for epoch in range(self.conf.epoch):
            for i, (input, labels) in enumerate(self.train_dataloader):
                try:
                    # with torch.autograd.profiler.profile(use_npu=True) as prof:
                    if not self.args.accelerate:
                        input = input.to(self.device)
                        labels = labels.to(self.device)
                    output = self.model.forward(input, labels=labels)
                    loss = output.loss
                    if self.args.fp16:
                        with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                            scaled_loss.backward()
                            torch.nn.utils.clip_grad_norm_(amp.master_params(self.optimizer), self.conf.max_grad_norm)
                    else:
                        if self.args.accelerate:
                            self.accelerator.backward(loss)
                        else:
                            loss.backward()
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.conf.max_grad_norm)
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    print(f'''epoch {epoch + 1}, loss {loss.item() : f}, lr {self.scheduler.get_lr()} ''')
                    # print(prof)
                    # prof.export_chrome_trace('profile_data.json')
                    progress_bar.update(1)
                except Exception as e:
                    traceback.print_exc()
                    print('error input shape: ', input.shape)
                    raise Exception
            #     if i > 0:
            #         break
            # break

            module_to_save = self.model.module if hasattr(self.model, 'module') else self.model
            module_to_save.save_pretrained(f'output/testing_{epoch}')

    def eval(self):
        metric = evaluate.load('accuracy')
        self.model.eval()
        for batch in self.eval_dataloader:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            with torch.no_grad():
                output = self.model(**batch)
            logits = output.logits
            predictions = torch.argmax(logits, dim=-1)
            metric.add_batch(predictions=predictions, references=batch['labels'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tokenizer', type=str, default=None, help='自训练的分词器')
    parser.add_argument('--data_path', type=str, required=True, help='训练数据集')
    parser.add_argument('--cuda', action='store_true', help='开启cuda模式')
    parser.add_argument('--npu', action='store_true', help='开启npu模式')
    parser.add_argument('--fp16', action='store_true', help='混合精度')
    parser.add_argument('--accelerate', action='store_true', help='cuda加速模式')

    args = parser.parse_args()
    t = Trainer(args)
    t.train()
