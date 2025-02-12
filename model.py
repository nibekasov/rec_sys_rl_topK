import numpy as np
import torch
import torch.nn as nn
import torch_optimizer as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import wandb

from reinforce import ChooseREINFORCE, reinforce_update


class Beta(nn.Module):
    def __init__(self, num_items):
        super(Beta, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(1024, num_items),
            nn.Softmax(dim=1)  # ✅ Добавлен `dim=1` для softmax
        )
        self.optim = optim.RAdam(self.net.parameters(), lr=1e-5, weight_decay=1e-5)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, state, action):
        print(f"DEBUG: state.shape = {state.shape}")
        print(f"DEBUG: action.shape = {action.shape}")

        state = state[:, :1024] if state.shape[1] > 1024 else state  # ✅ Ограничиваем state

        predicted_action = self.net(state)

        loss = self.criterion(predicted_action, action.argmax(dim=1))

        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        return predicted_action.detach()


class Critic(nn.Module):
    def __init__(self, input_dim, hidden_size, action_dim, dropout, init_w):
        super(Critic, self).__init__()
        self.drop_layer = nn.Dropout(p=dropout)
        # input_dim + action_dim => скрытый слой
        self.linear1 = nn.Linear(input_dim + action_dim, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, 1)

        # Инициализация
        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)

    def forward(self, state, action):
        # Обрезаем, чтобы не вылетала ошибка
        max_dim = 1024
        state = state[:, :max_dim] if state.shape[1] > max_dim else state

        # Собираем совместное представление
        value_input = torch.cat([state, action], dim=1)
        assert value_input.shape[1] == self.linear1.in_features, (
            f"Expected {self.linear1.in_features}, got {value_input.shape[1]}"
        )

        x = F.relu(self.linear1(value_input))
        x = self.drop_layer(x)
        x = F.relu(self.linear2(x))
        x = self.drop_layer(x)
        x = self.linear3(x)
        return x

def safe_histogram(data, max_bins=50):
    """
    Функция для безопасного создания гистограммы с ограничением на максимальное количество бинов.
    Если данных слишком много для гистограммы с 64 бинами, то она будет уменьшена.
    """
    num_bins = min(len(data), max_bins)
    histogram, _ = np.histogram(data, bins=num_bins)
    return histogram

class DiscreteActor(nn.Module):
    def __init__(self, input_dim, hidden_size, action_dim, init_w=0):
        super(DiscreteActor, self).__init__()
        self.linear1 = nn.Linear(input_dim, hidden_size)
        self.linear2 = nn.Linear(hidden_size, action_dim)

        self.saved_log_probs = []
        self.rewards = []
        self.correction = []
        self.lambda_k = []

    def gc(self):
        """ Очищает списки хранимых значений для обновления политики """
        del self.rewards[:]
        del self.saved_log_probs[:]
        del self.correction[:]
        del self.lambda_k[:]

    def forward(self, state):
        max_dim = 1024
        state = state[:, :max_dim] if state.shape[1] > max_dim else state

        x = F.relu(self.linear1(state))
        action_scores = self.linear2(x)
        return F.softmax(action_scores, dim=1)


    def _select_action(self, state, **kwargs):
        pi_probs = self.forward(state)
        dist = Categorical(pi_probs)

        # Выбираем действие
        pi_action = dist.sample()
        pi_action = pi_action.clamp(0, pi_probs.shape[1] - 1)

        self.saved_log_probs.append(dist.log_prob(pi_action))
        return pi_probs

    def pi_beta_sample(self, state, beta, action, **kwargs):
        beta_probs = beta(state.detach(), action=action)
        pi_probs = self.forward(state)
        beta_categorical = Categorical(beta_probs)
        pi_categorical = Categorical(pi_probs)

        pi_action = pi_categorical.sample()
        pi_action = pi_action.clamp(0, pi_probs.shape[1] - 1)

        beta_action = beta_categorical.sample()
        beta_action = beta_action.clamp(0, beta_probs.shape[1] - 1)

        pi_log_prob = pi_categorical.log_prob(pi_action)
        beta_log_prob = beta_categorical.log_prob(beta_action)

        return pi_log_prob, beta_log_prob, pi_probs




    def _select_action_with_TopK_correction(self, state, beta, action, K, writer, step, **kwargs):
        pi_log_prob, beta_log_prob, pi_probs = self.pi_beta_sample(state, beta, action)

        corr = torch.exp(pi_log_prob) / torch.exp(beta_log_prob)
        l_k = K * (1 - torch.exp(pi_log_prob)) ** (K - 1)

        # Применяем safe_histogram для безопасного создания гистограммы
        hist = safe_histogram(l_k.cpu().detach().numpy())

        writer.log({"l_k": wandb.Histogram(hist)}, step=step)
        writer.log({"correction": wandb.Histogram(corr.cpu().detach().numpy())}, step=step)

        self.correction.append(corr)
        self.lambda_k.append(l_k)
        self.saved_log_probs.append(pi_log_prob)

        return pi_probs

    def to(self, device):
        super(DiscreteActor, self).to(device)
        self.device = device
        if not hasattr(self, "_step"):
            self._step = 0
        return self

    def step(self):
        if not hasattr(self, "_step"):
            self._step = 0
        self._step += 1


class Reinforce:
    def __init__(self, policy_net, value_net, dropout=0.5, init_w=0.01):
        super(Reinforce, self).__init__()

        self.debug = {}
        self.writer = wandb
        self._step = 0  # Инициализация атрибута _step

        self.algorithm = reinforce_update

        # Создаем target сети для критика и политики
        self.target_policy_net = DiscreteActor(
            policy_net.linear1.in_features, 
            policy_net.linear1.out_features, 
            policy_net.linear2.out_features
        )
        self.target_value_net = Critic(
            value_net.linear1.in_features - value_net.linear2.out_features, 
            value_net.linear1.out_features, 
            value_net.linear2.out_features,
            dropout=dropout,  # Передаем значение dropout
            init_w=init_w     # Передаем значение init_w
        )

        # Копируем веса оригинальных сетей в target-сети
        self.target_policy_net.load_state_dict(policy_net.state_dict())
        self.target_value_net.load_state_dict(value_net.state_dict())

        # Определяем оптимизаторы
        value_optimizer = optim.Ranger(
            value_net.parameters(), lr=1e-5, weight_decay=1e-2
        )
        policy_optimizer = optim.Ranger(
            policy_net.parameters(), lr=1e-5, weight_decay=1e-2
        )

        # Добавляем сети в self.nets после их инициализации
        self.nets = {
            "value_net": value_net,
            "policy_net": policy_net,
            "target_policy_net": self.target_policy_net,  # ✅ Добавлено
            "target_value_net": self.target_value_net,    # ✅ Добавлено
        }

        self.optimizers = {
            "policy_optimizer": policy_optimizer,
            "value_optimizer": value_optimizer,
        }

        self.params = {
            "reinforce": ChooseREINFORCE(ChooseREINFORCE.reinforce_with_TopK_correction),
            "K": 10,
            "gamma": 0.99,
            "min_value": -10,
            "max_value": 10,
            "policy_step": 10,
            "soft_tau": 0.001,
        }

        self.loss_layout = {
            "test": {"value": [], "policy": [], "step": []},
            "train": {"value": [], "policy": [], "step": []},
        }

    def to(self, device):
        for net_name in self.nets:
            self.nets[net_name] = self.nets[net_name].to(device)
        self.device = device
        return self

    def update(self, batch, learn=True):
        print(f"DEBUG: _step in update: {self._step}")
        if not hasattr(self, "_step"):
            self._step = 0

        self._step += 1  # Инкрементируем _step

        return reinforce_update(
            batch,
            self.params,
            self.nets,  # Теперь `target_policy_net` и `target_value_net` есть в `nets`
            self.optimizers,
            device=self.device,
            debug=self.debug,
            writer=self.writer,
            learn=learn,
            step=self._step  # Передаем _step в reinforce_update
        )
