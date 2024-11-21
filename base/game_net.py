from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

initialization_dict = {
    "he": nn.init.kaiming_normal_,
    "he_uniform": nn.init.kaiming_uniform_,
    "xaiver": nn.init.xavier_normal_,
    "xaiver_uniform": nn.init.xavier_uniform_,
}

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
        **kwargs,
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
        device: str = "cpu",
        **kwargs
    ) -> None:
        super().__init__()
        self.conv = Conv2dBlock(**conv_config)
        
        self.residual = nn.Sequential()
        for _ in range(residual_config["n_block"]):
            self.residual.append(ResidualBlock(**residual_config))
        
        self.pv = PolicyValueHead(**policy_value_config)
    
        # Weight initialization
        self.apply(partial(init_weight, init_method))

        self.to(device)
    
    def dump(self, full_path: str) -> None:
        torch.save(self.state_dict(), full_path)
        return
    
    def load(self, full_path: str) -> None:
        self.load_state_dict(torch.load(full_path, weights_only=True))
        self.eval()
        return

    def forward(self, x):
        x = self.conv(x)
        x = self.residual(x)
        x = self.pv(x)
        return x

class GameData(Dataset):
    def __init__(self, board, p, v) -> None:
        super().__init__()
        self.X = board.view(*board.shape, 1) # N x (height x width) x extra_dimension if any
        self.p_target = p # N x (height x width) x extra_dimension if any
        self.v_target = v # n_array

    def to(self, device: str = "cpu") -> None:
        self.X = self.X.to(device).float()
        self.p_target = self.p_target.to(device).float()
        self.v_target = self.v_target.to(device).float()
        return

    def __len__(self):
        return len(self.v_target)

    def __getitem__(self, i):
        return self.X[i], self.p_target[i], self.v_target[i]

if __name__ == "__main__":
    from base.config import Config
    from time import time

    use_cuda = True
    device = "cuda" if use_cuda else "cpu"
    board = torch.randn((256,3,3,1)).to(device)
    config = Config.load("game_net.yaml", "tictactoe")["game_net"]

    t0 = time()
    with torch.no_grad():
        test_net = GameNet(device=device, **config,)        
        for _ in range(1000):
            p, v = test_net(board.permute(0,3,1,2))
            p = nn.Softmax(dim=1)(p).view(-1, 3, 3)

    print(time() - t0)
    print(p[0,:,:], v[0])
    print(p[12,:,:], v[12])
    pass