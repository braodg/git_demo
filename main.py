## main.py文件
import torch
import torch.nn as nn
import torch.optim as optim
import os
import argparse
from torch.distributed import init_process_group, get_rank, get_world_size
# 构造模型
model = nn.Linear(10, 10).to(local_rank)

outputs = model(torch.randn(20, 10).to(rank))
labels = torch.randn(20, 10).to(rank)
loss_fn = nn.MSELoss()
loss_fn(outputs, labels).backward()
optimizer = optim.SGD(model.parameters(), lr=0.001)
optimizer.step()

# ## Bash运行
# python main.py

