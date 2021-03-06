import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

from abstract_agent import AbstractAgent

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class BufferAttend1d(nn.Module):
    def __init__(self, dim_in, key_dim, val_dim):
        super().__init__()
        self.key_dim, self.val_dim = key_dim, val_dim
        self.key_fn = nn.Linear(dim_in, key_dim)
        self.query_fn = nn.Linear(dim_in, key_dim)
        self.value_fn = nn.Linear(dim_in, val_dim)

        self.fill = nn.Parameter(-1024 * torch.ones([1, 1]), requires_grad=False)

    def forward(self, x, buffer=None, residual=False, mask=None):
        if buffer is None:
            buffer = x

        query = self.key_fn(x)  # shape(..., Q, d)
        keys = self.key_fn(buffer)  # shape(..., K, d)
        vals = self.value_fn(buffer)  # shape(..., K, d)
        logits = torch.einsum("...qd, ...kd -> ...qk", query, keys) / np.sqrt(self.key_dim)  # shape(..., Q, K)

        if mask is not None:
            mask = ~mask.unsqueeze(1)  # Was lazy here...should probably fix the mask elsewhere.
            logits = torch.where(mask, logits, self.fill)

        probs = torch.exp(logits - logits.max(dim=-1, keepdim=True)[0])
        probs = probs / probs.sum(-1, keepdim=True)
        read = torch.einsum("...qk, ...kd -> ...qd", probs, vals)  # shape(..., Q, d)

        if residual:
            read = read + x

        return probs, read


class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []

    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]


class ActionValueLayer(nn.Module):

    def __init__(self, embedding, state_dim, num_heads, is_action_layer):
        super(ActionValueLayer, self).__init__()

        self.embeddings = embedding

        self.attend = BufferAttend1d(state_dim, num_heads, num_heads)
        self.is_action_layer = is_action_layer
        self.value_layer = nn.Linear(num_heads, 1)

    def act(self, state, mask=None):
        if len(state.shape) == 1:
            goal_state, links_state = state[0], state[1:]
            goal_state = torch.unsqueeze(goal_state, 0)
        else:
            goal_state, links_state = state[:, 0], state[:, 1:]
            goal_state = torch.unsqueeze(goal_state, 1)

        goal_embeddings = self.embeddings(goal_state).squeeze(-1)
        links_embeddings = self.embeddings(links_state).squeeze(-1)

        probs, read = self.attend(goal_embeddings, links_embeddings, mask=mask)

        if self.is_action_layer:
            return probs
        else:
            return self.value_layer(read)


class ActorCritic(nn.Module):
    def __init__(self, embeddings, state_dim, num_heads):
        super(ActorCritic, self).__init__()

        self.embeddings = embeddings

        self.action_layer = ActionValueLayer(self.embeddings, state_dim, num_heads, True)
        self.value_layer = ActionValueLayer(self.embeddings, state_dim, num_heads, False)

    def forward(self):
        raise NotImplementedError

    def act(self, orig_state, memory):
        state = torch.from_numpy(orig_state).long().to(device).squeeze()

        action_probs = self.action_layer.act(state)
        dist = Categorical(action_probs)
        action = dist.sample()

        memory.states.append(state)
        memory.actions.append(action)
        memory.logprobs.append(dist.log_prob(action))

        return action.item()

    def evaluate(self, state, action, mask):
        state = state.squeeze().permute([1, 0])  # (BATCH, LINKS)
        mask = mask[1:].permute([1, 0])

        action_probs = self.action_layer.act(state, mask)
        dist = Categorical(action_probs)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()

        state_value = self.value_layer.act(state, mask)

        return action_logprobs, torch.squeeze(state_value), dist_entropy


class PPO(AbstractAgent):
    def __init__(self, wiki_graph, kwargs):
        super().__init__(wiki_graph, kwargs)
        self.memory = Memory()

        self.emb_dim = kwargs['emb_dim']
        self.lr = kwargs['lr']
        self.beta1 = kwargs['beta1']
        self.beta2 = kwargs['beta2']
        self.gamma = kwargs['gamma']
        self.eps_clip = kwargs['eps_clip']
        self.k_epochs = kwargs['k_epochs']
        self.optimize_timestep = kwargs['optimize_timestep']
        self.num_heads = kwargs['num_heads']

        self.embeddings = torch.nn.Embedding(wiki_graph.num_vertices(), self.emb_dim)
        self.embeddings.weight.requires_grad = True

        self.policy = ActorCritic(self.embeddings, self.emb_dim, self.num_heads).to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.lr, betas=(self.beta1, self.beta2))
        self.policy_old = ActorCritic(self.embeddings, self.emb_dim, self.num_heads).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.MseLoss = nn.MSELoss()
        self.update_timestep = 0

    def act(self, state):
        return self.policy_old.act(state, self.memory)

    def update(self, reward, done):
        self.update_timestep += 1

        # Saving reward and is_terminals:
        self.memory.rewards.append(reward)
        self.memory.is_terminals.append(done)

        # update if its time
        if self.update_timestep % self.optimize_timestep == 0:
            self.optimize(self.memory)
            self.memory.clear_memory()

    def save(self, solved=False):
        if solved:
            torch.save(self.policy.action_layer, './models/PPO_action_layer_solved.pth')
        else:
            torch.save(self.policy.action_layer, './models/PPO_action_layer.pth')

    def optimize(self, memory):

        # Monte Carlo estimate of state rewards:
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # Normalizing the rewards:
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)

        # convert list to tensor
        old_states = torch.nn.utils.rnn.pad_sequence(memory.states, batch_first=False, padding_value=-1).to(device).detach()
        old_actions = torch.stack(memory.actions).to(device).detach()
        old_logprobs = torch.stack(memory.logprobs).to(device).detach()

        mask = old_states == -1

        old_states = torch.nn.utils.rnn.pad_sequence(memory.states, batch_first=False).to(device).detach()

        # Optimize policy for K epochs:
        for _ in range(self.k_epochs):
            # Evaluating old actions and values :
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions, mask)

            # Finding the ratio (pi_theta / pi_theta__old):
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss:
            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards) - 0.01 * dist_entropy

            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        # Copy new weights into old policy:
        self.policy_old.load_state_dict(self.policy.state_dict())
