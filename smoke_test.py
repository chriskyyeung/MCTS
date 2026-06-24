import sys

print("Testing imports and registry...", flush=True)

try:
    import games.connect4  # noqa: F401
    import games.knucklebones  # noqa: F401
    import games.tictactoe  # noqa: F401
    from base.registry import get_game, list_games

    print("Registered games:", list_games(), flush=True)
    assert "tictactoe" in list_games()
    assert "connect4" in list_games()

    tictactoe_trainer = get_game("tictactoe")["trainer_cls"](use_cuda=False)
    print("TicTacToeTrainer instantiated", flush=True)

    connect4_trainer = get_game("connect4")["trainer_cls"](use_cuda=False)
    print("Connect4Trainer instantiated", flush=True)

    config = tictactoe_trainer.config_dir
    print("Trainer config dir:", config, flush=True)

    print("Smoke tests passed successfully.", flush=True)
except Exception:
    import traceback

    traceback.print_exc()
    sys.exit(1)
