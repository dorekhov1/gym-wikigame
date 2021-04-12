import torch
import pandas as pd
import graph_tool as gt
import plotly.express as px
from sklearn.decomposition import PCA

from agents.PPO import ActionValueLayer

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

agent = torch.load("models/action_layer.pth").to(device)
embeddings = agent.embeddings
n = embeddings.num_embeddings
dim = embeddings.embedding_dim
indices = torch.arange(0, n, dtype=torch.long).to(device)
all_embeddings = embeddings(indices).detach().cpu().numpy()

pca = PCA(n_components=2)
components = pca.fit_transform(all_embeddings)

wiki_graph = gt.load_graph("data/processed/wiki_graph.gt")
p_titles = wiki_graph.vertex_properties["title"]

titles = []
for i in range(n):
    titles.append(p_titles[i])

df = pd.DataFrame(columns=['title', 'x', 'y'])
df['title'] = titles
df['x'] = components[:, 0]
df['y'] = components[:, 1]

fig = px.scatter(df, x='x', y='y', text=titles)
fig.update_traces(textposition='top center', textfont_size=18)
fig.update_layout(title={
    'text': "Visualizing Embeddings (Using PCA to reduce dimensions from 64 to 2)",
    'font': {
        "size": 30
    }
})

fig.show()
