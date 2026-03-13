# Train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
import os
import time

from transfomer import Transformer
from Data import get_dataloaders, PAD_IDX


def rate(step, d_model=512, warmup_steps=4000):
    # 防止 step=0 时出现除零错误
    step = max(step, 1)
    return (d_model ** -0.5) * min(step ** -0.5, step * (warmup_steps ** -1.5))


def train_epoch(model, dataloader, criterion, optimizer, scheduler, device, clip=1.0):
    model.train()
    total_loss = 0
    start_time = time.time()

    for i, (src, tgt_input, tgt_output) in enumerate(dataloader):
        src, tgt_input, tgt_output = src.to(device), tgt_input.to(device), tgt_output.to(device)

        optimizer.zero_grad()
        logits = model(src, tgt_input)
        loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_output.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        if scheduler:
            scheduler.step()

        total_loss += loss.item()
        if i % 100 == 0:
            print(f"  Batch {i}, Loss: {loss.item():.4f}")

    avg_loss = total_loss / len(dataloader)
    elapsed = time.time() - start_time
    return avg_loss, elapsed


def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for src, tgt_input, tgt_output in dataloader:
            src, tgt_input, tgt_output = src.to(device), tgt_input.to(device), tgt_output.to(device)
            logits = model(src, tgt_input)
            loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_output.reshape(-1))
            total_loss += loss.item()
    return total_loss / len(dataloader)


def main():
    # 超参数
    d_model = 512
    N = 6
    h = 8
    d_ff = 2048
    dropout = 0.1
    batch_size = 32
    max_len = 50
    min_freq = 2
    epochs = 100
    lr = 0.0001
    warmup_steps = 4000
    clip = 1.0
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    save_path = "transformer_model.pth"
    vocab_save_path = "vocab.pt"

    print(f"使用设备: {device}")

    print("加载数据...")
    train_loader, valid_loader, vocab_info = get_dataloaders(
        batch_size=batch_size, max_len=max_len, min_freq=min_freq
    )
    src_vocab_size = vocab_info['src_vocab_size']
    tgt_vocab_size = vocab_info['tgt_vocab_size']

    torch.save(vocab_info, vocab_save_path)
    print(f"源词汇表大小: {src_vocab_size}, 目标词汇表大小: {tgt_vocab_size}")

    model = Transformer(
        src_vocab=src_vocab_size,
        tag_vocab=tgt_vocab_size,
        d_model=d_model,
        N=N,
        h=h,
        d_ff=d_ff,
        dropout=dropout
    ).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    optimizer = optim.AdamW(model.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9)
    scheduler = LambdaLR(optimizer, lr_lambda=lambda step: rate(step, d_model, warmup_steps))

    best_valid_loss = float('inf')
    for epoch in range(1, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}")
        train_loss, train_time = train_epoch(model, train_loader, criterion, optimizer, scheduler, device, clip)
        valid_loss = evaluate(model, valid_loader, criterion, device)
        print(f"Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}, Time: {train_time:.2f}s")
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), save_path)
            print(f"  保存最佳模型，验证损失 {valid_loss:.4f}")

    print("训练完成！")


if __name__ == "__main__":
    main()