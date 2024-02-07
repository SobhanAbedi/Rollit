# Rollit game with pygame library
This is a rollit game (2/4 players) implementation using pygame library with both Player-Agent and Agent-Agent modes.
Base model incorporates <a href="Agents.py">multiple agents</a> (1 or 3) which will compete against a <a>single experimental agent</a> whose behavior is being validated.

## Base Model Agents
- Minimax Agent (MysteriousAgent)
- Partially Random Agent
- Intentionally Bad Agent

## Experimantal Agents
- Reflex Agent: A greedy agent
- Minimax Agent
- Alpha Beta Agent: Minimax agent with &alpha;-&beta; pruning
- Expectimax Agent

## Evaluation Function
Because of the large search tree inherent to this problem, use of a state evaluation function is non-mandatory. Two evaluation functions are implemented:
- Score Evaluation Function: Disk count for agent evaluated
- Better Evaluation Function: Stable scoring based on <a href="https://courses.cs.washington.edu/courses/cse573/04au/Project/mini1/RUSSIA/Final_Paper.pdf">An Analysis of Heuristics in Othello</a> by Sannidhanam and Annamalai

All the base agents use the first evaluation function while for the second agent one of two could be selected

## Running The game
The Base game can be run with the ```python rollit.py``` command with multiple optional key-value arguments:
- -a: This key sets the evaluation agent which can be selected from [MouseAgent, ReflexAgent, MinimaxAgent, AlphaBetaAgent, ExpectimaxAgent]
- -d: Evaluation agent search depth (3-6 is recommended) 
- -ea: This key sets base model agent[s] which can be selected from [MysteriousAgent, IntentionallyBadAtGameAgent, PartiallyRandomAgent]
- -ed: Base model agent search depth (3-6 is recommended)
- -n: Number of players including evaluation agent human player. This nuber should either be 2 or 4
- -ds: Game display mode which can be selected from [minimal, console, graphic]
- -fn: Evaluation function which can be selected form [scoreEvaluationFunction, betterEvaluationFunction]
- -q: This key can call specific preloaded game settings in <a href="modes.pickle">modes</a> file which include four settings from q1 to q4

note: contents of the <a href="modes.pickle">modes.pickle</a> file can be viewed in <a href="raw_modes.json">json form</a> 

Two examples:

```python rollit.py -a AlphaBetaAgent -d 2 -ea PartiallyRandomAgent -fn betterEvaluationFunction -ds minimal```

```python rollit.py -q q4```

## Game Environment
<video width="400" height="400" src="/git-media/sample.mp4"></video>