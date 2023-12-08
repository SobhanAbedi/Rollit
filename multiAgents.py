from Agents import Agent
import util
import random
from typing import List


class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """
    def __init__(self, *args, **kwargs) -> None:
        self.index = 0 # your agent always has index 0

    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        It takes a GameState and returns a tuple representing a position on the game board.
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions(self.index)

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        The evaluation function takes in the current and proposed successor
        GameStates (Game.py) and returns a number, where higher numbers are better.
        You can try and change this evaluation function if you want but it is not necessary.
        """
        nextGameState = currentGameState.generateSuccessor(self.index, action)
        return nextGameState.getScore(self.index) - currentGameState.getScore(self.index)


def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    Every player's score is the number of pieces they have placed on the board.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore(0)


class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxAgent, AlphaBetaAgent & ExpectimaxAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (Agents.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '4', **kwargs):
        self.index = 0 # your agent always has index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent which extends MultiAgentSearchAgent and is supposed to be implementing a minimax tree with a certain depth.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(**kwargs)

    def getAction(self, state):
        """
        Returns the minimax action using self.depth and self.evaluationFunction

        But before getting your hands dirty, look at these functions:

        gameState.isGameFinished() -> bool
        gameState.getNumAgents() -> int
        gameState.generateSuccessor(agentIndex, action) -> GameState
        gameState.getLegalActions(agentIndex) -> list
        self.evaluationFunction(gameState) -> float
        """
        "*** YOUR CODE HERE ***"
        # util.raiseNotDefined()
        _, action = self.chose_action(state, 0)
        return action

    def chose_action(self, state, cur_depth):
        """This function recursively finds the best action given the initial state and depth limit."""
        agent_count = state.getNumAgents()
        if state.isGameFinished() or cur_depth == self.depth:
            return self.evaluationFunction(state), ""
        index = cur_depth % agent_count
        action_list = state.getLegalActions(index)
        nxt_depth = cur_depth + 1
        if index == 0:
            chosen_act_val = 0
            chosen_act: tuple[int, int] = (-1, -1)
            for action in action_list:
                nxt_state = state.generateSuccessor(0, action)
                nxt_value, _ = self.chose_action(nxt_state, nxt_depth)
                if nxt_value > chosen_act_val:
                    chosen_act_val = nxt_value
                    chosen_act = action
        else:
            chosen_act_val = 64
            chosen_act: tuple[int, int] = (-1, -1)
            for action in action_list:
                nxt_state = state.generateSuccessor(index, action)
                nxt_value, _ = self.chose_action(nxt_state, nxt_depth)
                if nxt_value < chosen_act_val:
                    chosen_act_val = nxt_value
                    chosen_act = action
        return chosen_act_val, chosen_act


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning. It is very similar to the MinimaxAgent but you need to implement the alpha-beta pruning algorithm too.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(**kwargs)

    def getAction(self, state):
        """
        Returns the minimax action using self.depth and self.evaluationFunction

        You should keep track of alpha and beta in each node to be able to implement alpha-beta pruning.
        """
        "*** YOUR CODE HERE ***"
        # util.raiseNotDefined()
        _, action = self.chose_action(state, 0, 0, 64)
        return action

    def chose_action(self, state, cur_depth, lower_threshold, upper_threshold):
        """
        This function recursively finds the best action given the initial state and depth limit.
        Alpha-Beta pruning is also implemented.
        """
        agent_count = state.getNumAgents()
        if state.isGameFinished() or cur_depth == self.depth:
            return self.evaluationFunction(state), ""
        index = cur_depth % agent_count
        action_list = state.getLegalActions(index)
        nxt_depth = cur_depth + 1
        if index == 0:
            chosen_act_val = 0
            chosen_act: tuple[int, int] = (-1, -1)
            for action in action_list:
                nxt_state = state.generateSuccessor(index, action)
                nxt_value, _ = self.chose_action(nxt_state, nxt_depth, lower_threshold, upper_threshold)
                if nxt_value > chosen_act_val:
                    chosen_act_val = nxt_value
                    chosen_act = action
                if chosen_act_val > upper_threshold:
                    return chosen_act_val, ""
                if chosen_act_val > lower_threshold:
                    lower_threshold = chosen_act_val
            return chosen_act_val, chosen_act
        else:
            chosen_act_val = 64
            chosen_act: tuple[int, int] = (-1, -1)
            for action in action_list:
                nxt_state = state.generateSuccessor(index, action)
                nxt_value, _ = self.chose_action(nxt_state, nxt_depth, lower_threshold, upper_threshold)
                if nxt_value < chosen_act_val:
                    chosen_act_val = nxt_value
                    chosen_act = action
                if chosen_act_val < lower_threshold:
                    return chosen_act_val, ""
                if chosen_act_val < upper_threshold:
                    upper_threshold = chosen_act_val
            return chosen_act_val, chosen_act


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent which has a max node for your agent but every other node is a chance node.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(**kwargs)

    def getAction(self, state):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All opponents should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        # util.raiseNotDefined()
        _, action = self.chose_action(state, 0, 0)
        return action

    def chose_action(self, state, cur_depth, lower_threshold):
        """
        This function recursively finds the best action given the initial state and depth limit using Expectimax alg.
        When agent_count == 4, this function only considers expectimax pruning for first average node layer.
        If we implement pruning in second or third layer, not calculating average value of those layers fully, will
        introduce complications in the first average layer due to under-reporting of average values and it will
        lead to poor choices.
        """
        agent_count = state.getNumAgents()
        if state.isGameFinished() or cur_depth == self.depth:
            return self.evaluationFunction(state), ""
        index = cur_depth % agent_count
        action_list = state.getLegalActions(index)
        nxt_depth = cur_depth + 1
        if index == 0:
            chosen_act_val = -1
            chosen_act: tuple[int, int] = (-1, -1)
            for action in action_list:
                nxt_state = state.generateSuccessor(index, action)
                nxt_value, _ = self.chose_action(nxt_state, nxt_depth, lower_threshold)
                if nxt_value > chosen_act_val:
                    chosen_act_val = nxt_value
                    chosen_act = action
                if chosen_act_val > lower_threshold:
                    lower_threshold = chosen_act_val
            return chosen_act_val, chosen_act
        else:
            action_count = len(action_list)
            total_value = 0
            for i in range(action_count):
                action = action_list[i]
                nxt_state = state.generateSuccessor(index, action)
                nxt_value, _ = self.chose_action(nxt_state, nxt_depth, lower_threshold)
                total_value += nxt_value
                if index == 1:
                    if total_value < 64 * (i + 1) - action_count * (64 - lower_threshold):
                        return total_value / (i + 1.0), ""
            return total_value / action_count, ""


def betterEvaluationFunction(currentGameState):
    """
    Your extreme evaluation function.

    You are asked to read the following paper on othello heuristics and extend it for two to four player rollit game.
    Implementing a good stability heuristic has extra points.
    Any other brilliant ideas are also accepted. Just try and be original.

    The paper: Sannidhanam, Vaishnavi, and Muthukaruppan Annamalai. "An analysis of heuristics in othello." (2015).

    Here are also some functions you will need to use:

    gameState.getPieces(index) -> list
    gameState.getCorners() -> 4-tuple
    gameState.getScore() -> list
    gameState.getScore(index) -> int

    """

    "*** YOUR CODE HERE ***"

    # parity

    # corners

    # mobility

    # stability

    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction