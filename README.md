# Python implementation of MCTS 
## Description
Implementation of MCTS on
1. Tic-tac-toe
2. Connect-4
3. KnuckleBones (from Cult of the Lamb)

## Plan
- [x] Game mechanism
  - [x] Tic-tac-toe
  - [x] Connect-4
  - [x] KnuckleBones
- [x] MCTS implementation
  - [x] Close loop MCTS
    - Deterministic game
    - Non-deterministic game (i.e. with chance node)
  - [x] Open loop MCTS [non_determ]
- [ ] Alpha zero implementation
  - [x] Tic-tac-toe
  - [ ] Connect-4
  - [ ] KnuckleBones
- [x] Analysis
  - [x] EvE pipeline (multiprocessing available)
  - [x] Simple notebook for plotting EvE results

## Observation
1. Close loop MCTS on tic-tac-toe
   - Acceptable performance from the dropping winning rate trend.
2. Close loop MCTS on Connect-4
   - Didn't observe a significant advantage on 1st player.
3. MCTS on Knuckle
   - Fixed close loop issues where incomplete expansion caused the AI to not cancel out opponent's dice.
4. Alpha zero implementation on Tic-tac-toe
   - Successfully trained a high-performing model after resolving MCTS tree-reuse, value-backpropagation perspective, and exploration noise issues.
5. Analysis & Pipelines
   - Built an interactive multiprocess evaluation script and plotting notebook to run EvE, PvE, and EvP matchups.