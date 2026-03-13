# Data.py
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
import jieba
from collections import Counter


# 特殊标记
PAD_TOKEN = '<pad>'
BOS_TOKEN = '<bos>'
EOS_TOKEN = '<eos>'
UNK_TOKEN = '<unk>'

# 特殊标记索引
PAD_IDX = 0
BOS_IDX = 1
EOS_IDX = 2
UNK_IDX = 3


def tokenize_en(text):
    """英文分词：按空格切分（简单处理）"""
    return text.strip().split()


def tokenize_zh(text):
    """中文分词：使用jieba"""
    return list(jieba.cut(text.strip()))


def build_vocab(data_iter, tokenizer_fn, min_freq=2):
    """从数据迭代器构建词汇表，返回 stoi 和 itos"""
    counter = Counter()
    for src, tgt in data_iter:
        counter.update(tokenizer_fn(src))
        counter.update(tokenizer_fn(tgt))

    # 按词频排序，保留频率>=min_freq的词
    vocab_words = [word for word, freq in counter.items() if freq >= min_freq]
    vocab_words.sort(key=lambda x: counter[x], reverse=True)

    # 构建 itos: 特殊标记 + 词汇
    itos = [PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, UNK_TOKEN] + vocab_words
    stoi = {word: i for i, word in enumerate(itos)}
    return stoi, itos


def numericalize(sentence, tokenizer_fn, stoi, max_len=None):
    """将句子转换为索引序列，并添加BOS/EOS，若max_len则截断"""
    tokens = tokenizer_fn(sentence)
    if max_len:
        tokens = tokens[:max_len]
    indices = [stoi.get(token, UNK_IDX) for token in tokens]
    # 添加BOS和EOS
    indices = [BOS_IDX] + indices + [EOS_IDX]
    return indices


def collate_fn(batch, src_stoi, tgt_stoi, src_tokenizer, tgt_tokenizer, max_len=50):
    """对batch进行填充，返回模型输入和目标"""
    src_batch, tgt_batch = [], []
    for item in batch:
        src = item['translation']['en']
        tgt = item['translation']['zh']
        src_indices = numericalize(src, src_tokenizer, src_stoi, max_len)
        tgt_indices = numericalize(tgt, tgt_tokenizer, tgt_stoi, max_len)
        src_batch.append(src_indices)
        tgt_batch.append(tgt_indices)

    # 填充到相同长度
    src_padded = torch.nn.utils.rnn.pad_sequence(
        [torch.tensor(seq) for seq in src_batch], batch_first=True, padding_value=PAD_IDX
    )
    tgt_padded = torch.nn.utils.rnn.pad_sequence(
        [torch.tensor(seq) for seq in tgt_batch], batch_first=True, padding_value=PAD_IDX
    )

    tgt_input = tgt_padded[:, :-1]
    tgt_output = tgt_padded[:, 1:]

    return src_padded, tgt_input, tgt_output


def get_dataloaders(batch_size=32, max_len=50, min_freq=2, num_workers=0):
    """加载Helsinki-NLP/opus-100的zh-en数据集，返回train/valid dataloader和词汇表信息"""
    # 加载数据集（指定语言对为zh-en）
    dataset = load_dataset("Helsinki-NLP/opus-100", "en-zh")
    # dataset = load_dataset("bentrevett/multi30k")
    # dataset = dataset = load_dataset("iwslt2017", "iwslt2017-en-zh")

    train_data = dataset['train']
    valid_data = dataset['validation']

    # 定义分词器
    src_tokenizer = tokenize_en  # 英文
    tgt_tokenizer = tokenize_zh  # 中文

    # 构建词汇表（使用训练集）
    # print(train_data)
    # raise
    train_iter = [(item['translation']['en'], item['translation']['zh']) for item in train_data]
    src_stoi, src_itos = build_vocab(train_iter, src_tokenizer, min_freq)
    tgt_stoi, tgt_itos = build_vocab(train_iter, tgt_tokenizer, min_freq)

    # 创建DataLoader
    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_fn(
            batch,
            src_stoi, tgt_stoi,
            src_tokenizer, tgt_tokenizer,
            max_len
        ),
        num_workers=num_workers
    )

    valid_loader = DataLoader(
        valid_data,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_fn(
            batch,
            src_stoi, tgt_stoi,
            src_tokenizer, tgt_tokenizer,
            max_len
        ),
        num_workers=num_workers
    )

    vocab_info = {
        'src_vocab_size': len(src_stoi),
        'tgt_vocab_size': len(tgt_stoi),
        'src_stoi': src_stoi,
        'src_itos': src_itos,
        'tgt_stoi': tgt_stoi,
        'tgt_itos': tgt_itos,
        'pad_idx': PAD_IDX,
        'bos_idx': BOS_IDX,
        'eos_idx': EOS_IDX,
        'unk_idx': UNK_IDX,
    }

    return train_loader, valid_loader, vocab_info


if __name__ == "__main__":
    # 测试
    train_loader, valid_loader, vocab_info = get_dataloaders(batch_size=4)
    for src, tgt_input, tgt_output in train_loader:
        print("源序列形状:", src.shape)
        print("解码器输入形状:", tgt_input.shape)
        print("目标输出形状:", tgt_output.shape)
        break