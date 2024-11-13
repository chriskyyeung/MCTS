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
- [ ] MCTS implementation
  - [x] Close loop MCTS
    - Deterministic game
    - Non-deterministic game (i.e. with chance node)
  - [ ] Open loop MCTS (in progress)
- [ ] Analysis
  - [x] EvE pipeline (multiprocessing available)
  - [x] Simple notebook for plotting EvE results

## Observation
1. Close loop MCTS on tic-tac-toe
   - Acceptable performance from the dropping winning rate trend
2. Close loop MCTS on Connect-4
   - Didn't observe a significant advantage on 1st player
3. MCTS on Knuckle
   - Fixed close loop issues
     - Unfully expansion causes AI tends not to cancel out opponents dice