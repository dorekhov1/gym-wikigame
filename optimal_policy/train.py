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
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

embedding = nn.Embedding(wiki_graph.num_vertices(), emb_dim)
net = ActionValueLayer(embedding, emb_dim, num_heads, True)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=lr, betas=(beta1, beta2))

num_epochs = 1000
for epoch in range(num_epochs):

    running_loss = 0.0
    num_correct = 0
    for i, data in enumerate(dataloader, 0):
        states, actions, mask = data['state'], data['action'], data['mask']

        optimizer.zero_grad()

        outputs = net.act(states, mask)
        outputs = torch.squeeze(outputs)
        loss = criterion(outputs, actions)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        num_correct += int(sum(torch.argmax(outputs, dim=1) == actions))
        if i % 1000 == 0:
            print(f"Running loss: {running_loss / 1000}, accuracy: {num_correct / 1000}")
            running_loss = 0.0
            num_correct = 0

print('Finished Training')
