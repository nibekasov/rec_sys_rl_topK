import os
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch_optimizer as optim
import numpy as np
import pandas as pd
import pickle
from tqdm.auto import tqdm
from time import gmtime, strftime
import argparse
import wandb  # ✅ Добавляем Weights & Biases

from IPython.display import clear_output
import matplotlib.pyplot as plt

from utils.env import FrameEnv
from utils.plot import Plotter
from model import Critic, DiscreteActor, Beta, Reinforce
from reinforce import ChooseREINFORCE
from utils.data_utils import make_items_tensor 

try:
    cuda = torch.device('cuda')
except Exception:
    print("We dont use CUDA rn")

import os

# Получаем путь к директории, где находится `main.py`
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Динамические пути к данным и моделям
DATA_DIR = os.path.join(BASE_DIR, "data", "ml-1m")
MODEL_DIR = os.path.join(BASE_DIR, "models")


# 🔹 Аргументы командной строки
parser = argparse.ArgumentParser()
parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
parser.add_argument("--dropout", type=float, default=0.5, help="dropout rate")
parser.add_argument("--frame_size", type=int, default=10, help="frame size for training")
parser.add_argument("--batch_size", type=int, default=10, help="batch size for training")
parser.add_argument("--epochs", type=int, default=20, help="training epochs")
parser.add_argument("--top_k", type=int, default=10, help="compute metrics@top_k")
parser.add_argument("--embedding_dim", type=int, default=32, help="dimension of embedding")
parser.add_argument("--policy_input_dim", type=int, default=1024, help="input dimension for policy/value net")
parser.add_argument("--policy_hidden_dim", type=int, default=4096, help="hidden dimension for policy/value net")
parser.add_argument("--data_path", type=str, default=DATA_DIR)
parser.add_argument("--model_path", type=str, default=MODEL_DIR)
parser.add_argument("--init_weight", type=float, default=0.01, help="initial weight value")
parser.add_argument("--plot_every", type=int, default=100, help="how many steps to plot the result")
parser.add_argument("--disable-cuda", action="store_true", help="Disable CUDA")
args = parser.parse_args()

# 🔹 Определяем устройство (GPU или CPU)
args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {args.device}")  # Проверяем, какое устройство используется


# 🔹 Пути к данным
args.embedding_path = os.path.join(args.data_path, 'ml20_pca128.pkl')
args.rating_path = os.path.join(args.data_path, 'ratings.csv')

cudnn.benchmark = True


if __name__ == '__main__':
    # Инициализация Weights & Biases
    wandb.init(project="reinforce-topK", config=vars(args))

    # Загружаем эмбеддинги
    movie_embeddings_key_dict = pickle.load(open(args.embedding_path, "rb"))

    # Преобразуем эмбеддинги в тензоры
    embeddings, key_to_id, id_to_key = make_items_tensor(movie_embeddings_key_dict)

    # Определяем num_items
    num_items = len(embeddings)

    # Инициализируем окружение
    env = FrameEnv(args.embedding_path, args.rating_path, num_items, frame_size=10, 
                   batch_size=25, num_workers=1, test_size=0.05)

    # Инициализация сетей и алгоритмов
    beta_net = Beta(num_items).to(args.device)
    value_net = Critic(args.policy_input_dim, args.policy_hidden_dim, num_items, 
                       args.dropout, args.init_weight).to(args.device)
    policy_net = DiscreteActor(args.policy_input_dim, args.policy_hidden_dim, num_items).to(args.device)

    policy_net.action_source = {'pi': 'beta', 'beta': 'beta'}

    reinforce = Reinforce(policy_net, value_net).to(args.device)

    # Настройка метода выбора действия
    def select_action_corr(state, action, K, writer, step, **kwargs):
        return reinforce.nets['policy_net']._select_action_with_TopK_correction(
            state, beta_net.forward, action, K=K, writer=wandb, step=step
        )

    reinforce.nets['policy_net'].select_action = select_action_corr
    reinforce.params['reinforce'] = ChooseREINFORCE(ChooseREINFORCE.reinforce_with_TopK_correction)
    reinforce.params['K'] = 10

    # Создаем объект Plotter
    plotter = Plotter(reinforce.loss_layout, [['value', 'policy']])

    # Обучение модели с логированием
    for epoch in range(args.epochs):
        for batch in tqdm(env.train_dataloader):
            loss = reinforce.update(batch)  # Вызов update
            if loss:
                plotter.log_losses(loss)  # ✅ Вызываем через экземпляр `plotter`
                wandb.log({"loss": loss})

            if reinforce._step % args.plot_every == 0:
                clear_output(True)
                print('step', reinforce._step)
                plotter.plot_loss()
                wandb.log({"step": reinforce._step, "plot_loss": plotter.get_current_loss()})



    # ✅ Закрываем `wandb` после завершения обучения
    #wandb.finish()
