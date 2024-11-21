from multiprocessing import Queue, Event
from pathlib import Path
import random
import os

import numpy as np
import torch
from torch.multiprocessing import Process
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from base.config import Config
from base.game_net import GameData, GameNet, PolicyValueLoss
from base.game_state import GameState
from base.puct_node import PUCTNode, PUCTRoot


class Trainer:
    def __init__(
            self, 
            game: str,
            game_constructor: GameState,
            game_config: str,
            board_size: tuple = None,
            use_cuda: bool = False,
    ) -> None:
        self.game = game
        self.game_constructor = game_constructor
        self.game_config = game_config
        self.board_size = board_size

        self.set_device(use_cuda)
        pass

    def set_device(self, use_cuda: bool):
        self.device = "cuda" if use_cuda and torch.cuda.is_available() else "cpu"
        return

    def seed_everything(self, seed: int):
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        return

    def get_model(self, model_path, config):
        model = GameNet(device=self.device, **config)
        if Path(model_path).exists():
            print("Previous model found... Loaded")
            model.load(model_path)
        else:
            print("Previous model not found... Dumped the current one")
            model.dump(model_path)
        return model

    def save_battle_record(self, record, path) -> None:
        return torch.save(record, path)

    def load_battle_record(self, path) -> None:
        return torch.load(path)

    def self_battle(self, player: GameNet, battle_config: dict) -> None:
        # Initialize the battle record list
        battle_record = {"board": [], "p": [], "v": []}
        # save_path = battle_config["battle_path"].format(game=self.game, version=battle_config["battle_version"])

        # Get MCT and netwotk ready
        player.eval()
        root = PUCTRoot(self.device)

        for _ in tqdm(range(battle_config["n_game"])):
            # Initialize game record
            board, p, v = [], [], []

            # Initialize the game
            current = PUCTNode(self.game_constructor(), None, 0)

            while not current.state.is_game_over:
                board.append(current.state._board)
                v.append(current.state._turnID)

                # After search, root.child is the original state, current is the new state
                current: PUCTNode = root.search(current, battle_config["n_simulation"], player)
                assert root.child.N > 0, "Oops... Unexpected zero"
                p.append(root.child.N_a / root.child.N)

            battle_record["board"].append(torch.from_numpy(np.stack(board)))
            battle_record["p"].append(torch.from_numpy(np.stack(p)))
            battle_record["v"].append(torch.from_numpy(np.array(v) * current.state.game_result))

        for k, v in battle_record.items():
            battle_record[k] = torch.cat(v)

        return battle_record
        
    def mp_self_battle(self, player: GameNet, battle_config: dict, q: Queue = None, done = None) -> None:
        torch.set_num_threads(1)
        battle_record = self.self_battle(player, battle_config)

        q.put(battle_record)
        done.wait()
        return

    def generate_battle_record(self, use_cuda: bool = False):
        # Temporaily set the device
        orig_device = self.device
        self.set_device(use_cuda)

        # Multiprocessing setup
        processes = []
        record_queue = Queue()
        done = Event()

        # Battle configuration
        config = Config.load("game_net.yaml", self.game)
        player = self.get_model(config["model_in_path"], config.pop("game_net")).share_memory()
        # player = GameNet(device=self.device, **config.pop("game_net")).share_memory()
        # if Path(config["model_in_path"]).exists():
        #     player.load(config["model_in_path"])
        # else:
        #     player.dump(config["model_in_path"])

        # Start all processes
        for _ in range(config["n_process"]):
            p = Process(target=self.mp_self_battle, args=(player, config, record_queue, done))
            processes.append(p)
            p.start()
        
        # Combine the results
        b, p_target, v_target = [], [], []
        for _ in processes:
            record = record_queue.get()
            b.append(record.pop("board"))
            p_target.append(record.pop("p"))
            v_target.append(record.pop("v"))

        # Release all threads
        done.set()

        # Save the battle records
        self.save_battle_record(
            GameData(
                torch.cat(b),
                torch.cat(p_target),
                torch.cat(v_target),
            ),
            config["battle_path"]
        )

        # Reset the device in case any conflicts
        self.device = orig_device
        return


    def train(
        self,
        game_net: GameNet,
        game_data: GameData,
        n_epoch: int,
        hyperparameter: dict,
    ):
        # Pre-training set-up
        from time import time
        self.seed_everything(hyperparameter["seed"])

        # Network setup
        game_net.train()
        criterion = PolicyValueLoss()
        optimizer = optim.Adam(game_net.parameters(), lr=hyperparameter["lr"])
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            **hyperparameter["schedular"]["plateau"]
        )

        # Prepare data
        game_data.to(self.device)
        batch_size = hyperparameter["batch_size"]
        # idx = np.random.randint(len(game_data), size=16)
        # print(idx)
        # game_data.X = game_data.X[idx,:,:,:]
        # game_data.p_target = game_data.p_target[idx,:]
        # game_data.v_target = game_data.v_target[idx]
        train_loader = DataLoader(game_data, batch_size=batch_size, shuffle=True)
        print(f"Per epoch = {len(train_loader)} batch of size {batch_size}")

        for epoch in range(n_epoch):
            epoch_time = time()
            epoch_loss = 0.
            n_batch = 0

            for data in train_loader:
            # for data in tqdm(train_loader):
                n_batch += 1
                optimizer.zero_grad()

                board, p_target, v_target = data
                p, v = game_net(torch.permute(board, (0,3,1,2)))
                loss = criterion(p_target, v_target, p, v.flatten())
                loss.backward()

                epoch_loss += loss
                optimizer.step()

            epoch_loss /= n_batch
            scheduler.step(epoch_loss)
            last_lr = scheduler.get_last_lr()[0]
            print(f"Epoch {epoch} -- Loss: {epoch_loss} in {time()-epoch_time:.2f} s with lr = {last_lr:.4f}")
        
        return game_net

if __name__ == "__main__":
    from base.config import Config
    from tictactoe.tictactoe import TicTacToe
    
    use_cuda = False
    self = Trainer("tictactoe", TicTacToe, dict(), None, use_cuda=use_cuda)
    config = Config.load("game_net.yaml", "tictactoe")
    player = GameNet(device=self.device, **config["game_net"]).dump(config["model_path"])

    # self.self_battle(player, Config.load("game_net.yaml", "tictactoe"))