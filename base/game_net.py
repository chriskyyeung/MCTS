from functools import partial
import random
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

initialization_dict = {
    "he": nn.init.kaiming_normal_,
    "he_uniform": nn.init.kaiming_uniform_,
    "xaiver": nn.init.xavier_normal_,
    "xaiver_uniform": nn.init.xavier_uniform_,
}


def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return

def init_weight(method: str, m):
    if type(m) == nn.Conv2d or type(m) == nn.Linear:
        initialization_dict[method](m.weight)

class Conv2dBlock(nn.Sequential):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        stride=1,
        padding=1,
        add_relu=True,
    ) -> None:
        super().__init__(
            nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channel)
        )
        if add_relu:
            self.append(nn.ReLU())

        pass

class ResidualBlock(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        stride=1,
        padding=1,
    ) -> None:
        super().__init__()
        self.conv1 = Conv2dBlock(in_channel, out_channel, kernel_size, stride=stride, padding=padding)
        self.conv2 = Conv2dBlock(out_channel, in_channel, kernel_size, stride=stride, padding=padding, add_relu=False)
    
    def forward(self, x):
        res = x
        x = self.conv1(x)
        x = self.conv2(x)
        return F.relu(res + x)

class PolicyValueHead(nn.Module):
    def __init__(
        self,
        in_channel,
        height,
        width,
        p_channel,
        p_output,
        v_channel,
        v_middle,
    ) -> None:
        super().__init__()

        # Policy Head, kernel fixed to be 1, with padding = 0
        self.policy_conv1 = Conv2dBlock(in_channel, p_channel, kernel_size=1, padding=0)
        self.policy_fc1 = nn.Linear(height * width * p_channel, p_output)
        self.softmax = nn.Softmax(dim=1)

        # Value Head, kernel fixed to be 1, with padding = 0
        self.value_conv1 = Conv2dBlock(in_channel, v_channel, kernel_size=1, padding=0)
        self.value_fc1 = nn.Linear(height * width * v_channel, v_middle)
        self.value_fc2 = nn.Linear(v_middle, 1)
        pass

    def forward(self, x):
        p = torch.flatten(self.policy_conv1(x), start_dim=1)
        p = self.policy_fc1(p)

        v = torch.flatten(self.value_conv1(x), start_dim=1)
        v = F.relu(self.value_fc1(v))
        v = F.tanh(self.value_fc2(v))
        return p, v

class PolicyValueLoss(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.square_loss = nn.MSELoss()
        self.ce_loss = nn.CrossEntropyLoss()
    
    def forward(self, p_target, v_target, p_head, v_head):
        v_loss = self.square_loss(v_head, v_target)
        p_loss = self.ce_loss(p_head, p_target)
        return v_loss + p_loss

class GameNet(nn.Module):
    def __init__(
        self,
        init_method: str,
        conv_config: dict,
        residual_config: dict,
        policy_value_config: dict,
    ) -> None:
        super().__init__()
        self.conv = Conv2dBlock(**conv_config)
        
        self.residual = nn.Sequential()
        for _ in range(residual_config.pop("n_block")):
            self.residual.append(ResidualBlock(**residual_config))
        
        self.pv = PolicyValueHead(**policy_value_config)
    
        # Weight initialization
        self.apply(partial(init_weight, init_method))
    
    def forward(self, x):
        x = self.conv(x)
        x = self.residual(x)
        x = self.pv(x)
        return x

class GameData(Dataset):
    def __init__(self, board, p, v) -> None:
        super().__init__()
        self.X = board # N x (height x width) x extra_dimension if any
        self.p_target = p # N x (height x width) x extra_dimension if any
        self.v_target = v # n_array


    def __len__(self):
        return len(self.v_target)

    def __getitem__(self, i):
        return self.X[i], self.p_target[i], self.v_target[i]

def train(
    game_net: GameNet,
    game_data,
    n_epoch,
    hyperparameter,
    use_cuda=True,
    seed=0,
):
    # Pre-training set-up
    seed_everything(0)
    use_cuda = torch.cuda.is_available() if use_cuda else False

    # Network setup
    game_net.train()
    criterion = PolicyValueLoss()
    optimizer = optim.adam(game_net.parameters(), lr=hyperparameter["lr"])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        **hyperparameter["schedular"]["plateau"]
    )

    # Prepare data
    train_set = GameData(game_data)
    train_loader = DataLoader(train_set, batch_size=hyperparameter["batch_size"], shuffle=True, )

    for epoch in range(n_epoch):
        epoch_loss = 0.
        n_batch = 0

        for data in train_loader:
            n_batch += 1
            board, p_target, v_target = data
            if use_cuda:
                board = board.cuda().float()
                p_target = p_target.cuda().float()
                v_target = v_target.cuda().float()

            optimizer.zero_grad()
            p, v = game_net(board)
            loss = criterion(p_target, v_target, p, v)
            loss.backward()
            epoch_loss += loss
            optimizer.step()

        scheduler.step(epoch_loss / n_batch)
    return


if __name__ == "__main__":
    import yaml
    with open("game_net.yaml", "r") as f:
        config = yaml.safe_load(f)["nn"]
    
    board = torch.randn((16,3,3,1))
    config["convolution"]["in_channel"] = board.shape[-1]
    config["residual"]["in_channel"] = config["convolution"]["out_channel"]
    config["policy_value"]["in_channel"] = config["residual"]["in_channel"]
    config["policy_value"]["height"] = board.shape[1]
    config["policy_value"]["width"] = board.shape[2]
    config["policy_value"]["p_output"] = board.shape[1] * board.shape[2]

    with torch.no_grad():
        test_net = GameNet(
            config["init_method"],
            config["convolution"],
            config["residual"],
            config["policy_value"],
        )

        p, v = test_net(board.permute(0,3,1,2))
        p = nn.Softmax(dim=1)(p).view(-1, 3, 3)

        print(p[0,:,:], v[0])
        print(p[12,:,:], v[12])
    pass