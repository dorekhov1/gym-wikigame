import pickle

import torch
import torch.nn as nn
from torch.distributions import Categorical

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


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
        self.linear_layer = nn.Linear(state_dim*2, 1)
        self.is_action_layer = is_action_layer
        if is_action_layer:
            self.last_layer = nn.Softmax(dim=0)
        else:
            self.last_layer = torch.mean

    def act(self, state):

        if len(state.shape) == 1:
            state = torch.reshape(state, [1, state.shape[0]])
        goal_state, links_state = state[:, 0], state[:, 1:]
        goal_state = torch.unsqueeze(goal_state, 1)

        goal_embeddings = self.embeddings(goal_state)
        links_embeddings = self.embeddings(links_state)

        attn_output, _ = self.multihead_attn(links_embeddings, links_embeddings, links_embeddings)

        goal_embeddings = goal_embeddings.expand(-1, attn_output.shape[1], -1)

        state_embeddings = torch.cat((goal_embeddings, attn_output), dim=-1)
        scores = self.linear_layer(state_embeddings)
        scores = torch.squeeze(scores, dim=-1)
        if self.is_action_layer:
            return self.last_layer(scores)
        else:
            return self.last_layer(scores, dim=1)


class ActorCritic(nn.Module):
    def __init__(self, embeddings, state_dim, action_dim, n_latent_var):
        super(ActorCritic, self).__init__()

        self.embeddings = embeddings

        self.action_layer = ActionValueLayer(self.embeddings, state_dim, 2, True)
        self.value_layer = ActionValueLayer(self.embeddings, state_dim, 2, False)

    def forward(self):
        raise NotImplementedError

    def act(self, state, memory):

        state = torch.from_numpy(state).long().to(device)

        action_probs = self.action_layer.act(state)

        dist = Categorical(action_probs)
        action = dist.sample()

        memory.states.append(state)
        memory.actions.append(action)
        memory.logprobs.append(dist.log_prob(action))

        return action.item()

    def evaluate(self, state, action):

        action_probs = self.action_layer.act(state)
        dist = Categorical(action_probs)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()

        state_value = self.value_layer.act(state)

        return action_logprobs, torch.squeeze(state_value), dist_entropy


class PPO:
    def __init__(self, state_dim, action_dim, n_latent_var, lr, betas, gamma, K_epochs, eps_clip):

        with open("data/processed/torch_embeddings.pickle", "rb") as handle:
            self.embeddings = pickle.load(handle)

        self.lr = lr
        self.betas = betas
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs

        self.policy = ActorCritic(self.embeddings, state_dim, action_dim, n_latent_var).to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr, betas=betas) # TODO apparently this optimizer won't work
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
        old_states = torch.nn.utils.rnn.pad_sequence(memory.states, batch_first=True).to(device).detach()
        old_actions = torch.stack(memory.actions).to(device).detach()
        old_logprobs = torch.stack(memory.logprobs).to(device).detach()


        # Optimize policy for K epochs:
        for _ in range(self.K_epochs):
            # Evaluating old actions and values :
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)

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