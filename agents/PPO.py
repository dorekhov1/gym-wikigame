import pickle

import torch
import torch.nn as nn
from torch.distributions import Categorical

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = 'cpu'

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

        self.multihead_attn = nn.MultiheadAttention(state_dim, num_heads)
        self.linear_layer1 = nn.Linear(state_dim, 128)
        self.relu = nn.ReLU()
        self.linear_layer2 = nn.Linear(128, 1)
        self.is_action_layer = is_action_layer
        if is_action_layer:
            self.last_layer = nn.Softmax(dim=-1)
        else:
            self.last_layer = torch.mean

    def act(self, state, mask=None):

        goal_state, links_state = state[0], state[1:]
        goal_state = torch.unsqueeze(goal_state, 0)

        goal_embeddings = self.embeddings(goal_state)
        links_embeddings = self.embeddings(links_state)

        # goal_embeddings = goal_embeddings.expand(links_embeddings.shape[0], -1, -1)

        attn_output, attn_output_weights = self.multihead_attn(goal_embeddings, links_embeddings, links_embeddings, key_padding_mask=mask)

        # scores = self.linear_layer1(attn_output)
        # scores = self.relu(scores)
        # scores = self.linear_layer2(scores)
        # scores = torch.squeeze(scores, dim=-1)

        attn_output_weights = attn_output_weights[:, 0]

        if self.is_action_layer:
            if mask is not None:
                attn_output_weights[mask] = float('-inf')
            return self.last_layer(attn_output_weights)
        else:
            return self.last_layer(attn_output_weights, dim=1)


class ActorCritic(nn.Module):
    def __init__(self, embeddings, state_dim, action_dim, n_latent_var):
        super(ActorCritic, self).__init__()

        self.embeddings = embeddings

        self.action_layer = ActionValueLayer(self.embeddings, state_dim, 2, True)
        self.value_layer = ActionValueLayer(self.embeddings, state_dim, 2, False)

    def forward(self):
        raise NotImplementedError

    def act(self, orig_state, memory):

        state = torch.from_numpy(orig_state).long().to(device)
        if len(state.shape) == 1:
            state = torch.reshape(state, [state.shape[0], 1])

        action_probs = self.action_layer.act(state)
        # action_probs = action_probs.permute([1, 0])
        dist = Categorical(action_probs)
        action = dist.sample()

        memory.states.append(state)
        memory.actions.append(action)
        memory.logprobs.append(dist.log_prob(action))

        return action.item()

    def evaluate(self, state, action, mask):

        state = state.squeeze()
        mask = mask.squeeze()
        mask = mask[1:]
        mask = mask.permute([1, 0])

        action_probs = self.action_layer.act(state, mask)
        # action_probs = action_probs.permute([1, 0])
        dist = Categorical(action_probs)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()

        state_value = self.value_layer.act(state, mask)

        return action_logprobs, torch.squeeze(state_value), dist_entropy


class PPO:
    def __init__(self, state_dim, action_dim, n_latent_var, lr, betas, gamma, K_epochs, eps_clip):

        with open("data/processed/torch_embeddings.pickle", "rb") as handle:
            self.embeddings = pickle.load(handle)

        self.embeddings.weight.requires_grad = True

        self.lr = lr
        self.betas = betas
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs

        self.policy = ActorCritic(self.embeddings, state_dim, action_dim, n_latent_var).to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr, betas=betas)
        self.policy_old = ActorCritic(self.embeddings, state_dim, action_dim, n_latent_var).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()

    def update(self, memory):

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
        for _ in range(self.K_epochs):
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