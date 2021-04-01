import pickle
import torch.optim as optim
import torch.nn as nn
import graph_tool as gt
from torch.utils.data import DataLoader

from agents.PPO import ActionValueLayer
from optimal_policy.data_generation import *


# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = 'cpu'

wiki_graph = gt.load_graph('data/processed/wiki_graph.gt')

emb_dim = 64
num_heads = 32
lr = 0.002
beta1, beta2 = 0.9, 0.999
batch_size = 32

g = OptimalPolicyDataGenerator()
dataset = g.generate_dataset()
train_size = int(len(dataset)*0.8)
val_size = len(dataset) - train_size
train_subset, val_subset = torch.utils.data.random_split(dataset, [train_size, val_size])
train_dataloader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=0)
val_dataloader = DataLoader(val_subset, batch_size=batch_size, shuffle=True, num_workers=0)

embedding = nn.Embedding(wiki_graph.num_vertices(), emb_dim)
net = ActionValueLayer(embedding, emb_dim, num_heads, True)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=lr, betas=(beta1, beta2))

num_epochs = 1000
log_interval = 100
for epoch in range(num_epochs):

    train_loss, val_loss = 0, 0
    train_num_correct, val_num_correct = 0, 0
    for i, data in enumerate(train_dataloader, 0):
        states, actions, mask = data['state'], data['action'], data['mask']

        optimizer.zero_grad()

        outputs = net.act(states, mask)
        outputs = torch.squeeze(outputs)
        loss = criterion(outputs, actions)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_num_correct += int(sum(torch.argmax(outputs, dim=1) == actions))

    for i, data in enumerate(val_dataloader, 0):
        states, actions, mask = data['state'], data['action'], data['mask']

        outputs = net.act(states, mask)
        outputs = torch.squeeze(outputs)
        loss = criterion(outputs, actions)
        val_loss += loss.item()
        val_num_correct += int(sum(torch.argmax(outputs, dim=1) == actions))

    print(f"Epoch: {epoch+1}, Train loss: {train_loss / train_size}, Train accuracy: {train_num_correct / train_size}, Val loss: {val_loss / val_size}, Val accuracy: {val_num_correct / val_size}")

print('Finished Training')
