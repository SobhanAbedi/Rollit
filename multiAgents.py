from Agents import Agent
import util
import random
from typing import List, Tuple
from Game import GameState


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

    def __init__(self, evalFn = 'betterEvaluationFunction', depth = '4', **kwargs):
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


def get_possible_actions(state: GameState, agent_count):
    legal_actions_list = [set()]*agent_count
    for i in range(agent_count):
        current_pieces = state.getPieces(i)
        current_legal_actions_list = []
        for k in range(agent_count - 1):
            if k != i:
                current_legal_actions_list.append(legal_actions_list[k])
        for piece in current_pieces:
            for dir in [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]:
                pos = (piece[0] + dir[0], piece[1] + dir[1])
                if state.isWithinBorders(pos) and state.data.board[pos[0]][pos[1]] == -1:
                    for legal_actions in current_legal_actions_list:
                        legal_actions.add(pos)
    return legal_actions_list


def count_stable_pieces(state: GameState, corners, index) -> int:
    if index not in corners:
        return 0
    board = state.data.board
    from_top = [0]*8
    from_bottom = [0]*8
    top_max = bottom_max = 8
    for i in range(8):
        for j in range(top_max):
            if board[j][i] != index:
                top_max = j
                break
        from_top[i] = top_max
        for j in range(bottom_max):
            if board[7-j][i] != index:
                bottom_max = j
                break
        from_bottom[i] = bottom_max
    top_max = bottom_max = 8
    for i in range(7, -1, -1):
        for j in range(top_max):
            if board[j][i] != index:
                top_max = j
                break
        if top_max > from_top[i]:
            from_top[i] = top_max
        else:
            break
    for i in range(7, -1, -1):
        for j in range(bottom_max):
            if board[7-j][i] != index:
                bottom_max = j
                break
        if bottom_max > from_bottom[i]:
            from_bottom[i] = bottom_max
        else:
            break
    total_stable_pieces = 0
    for i in range(8):
        col_stable = min(8, from_top[i] + from_bottom[i])
        total_stable_pieces += col_stable
    return total_stable_pieces


def betterEvaluationFunction(currentGameState: GameState):
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
    agent_count = currentGameState.getNumAgents()

    # parity
    agents_parity_score = currentGameState.getScore()
    sum_of_scores = sum(agents_parity_score)
    parity_score = (3 * agents_parity_score[0] - sum_of_scores - max(agents_parity_score[1:])) / (2 * sum_of_scores)

    # corners
    agents_corner_score = [0]*agent_count
    corners = currentGameState.getCorners()
    for corner in corners:
        if corner != -1:
            agents_corner_score[corner] += 1
    corner_score = 0
    sum_of_scores = sum(agents_corner_score)
    if sum_of_scores > 0:
        corner_score = (3 * agents_corner_score[0] - sum_of_scores - max(agents_corner_score[1:])) / (2 * sum_of_scores)

    # mobility
    possible_agents_actions = get_possible_actions(currentGameState, agent_count)
    legal_agent_actions = []
    agents_action_count = [0]*agent_count
    agents_possible_action_count = [0] * agent_count
    for i in range(agent_count):
        legal_actions = currentGameState.getLegalActions(i)
        legal_agent_actions.append(legal_actions)
        agents_action_count[i] = len(legal_actions)
        agents_possible_action_count[i] = len(possible_agents_actions[i])
    actual_mobility_score = 0
    sum_of_scores = sum(agents_action_count)
    if sum_of_scores > 0:
        actual_mobility_score = (3 * agents_action_count[0] - sum_of_scores -
                                 max(agents_action_count[1:])) / (2 * sum_of_scores)
    potential_mobility_score = 0
    sum_of_scores = sum(agents_possible_action_count)
    if sum_of_scores > 0:
        potential_mobility_score = (3 * agents_possible_action_count[0] - sum_of_scores -
                                    max(agents_possible_action_count[1:])) / (2 * sum_of_scores)

    agents_potential_action_count = [0]*agent_count
    for i in range(agent_count):
        agents_potential_action_count[i] = len(currentGameState.getPossibleActionsSimplified())

    # stability
    initial_pieces: List[Tuple[int, int]] = currentGameState.getPieces(0)
    initial_piece_count = len(initial_pieces)
    not_unstable_pieces = set(initial_pieces)
    for i in range(1, agent_count):
        for action in legal_agent_actions[i]:
            nxt_state = currentGameState.generateSuccessor(i, action)
            nxt_pieces = nxt_state.getPieces(0)
            not_unstable_pieces = not_unstable_pieces.intersection(nxt_pieces)
    unstable_piece_count = initial_piece_count - len(not_unstable_pieces)
    stable_piece_count = [0]*agent_count
    for i in range(agent_count):
        stable_piece_count[i] = count_stable_pieces(currentGameState, corners, i)
    sum_of_stables = sum(stable_piece_count)
    stability_score = ((3 * stable_piece_count[0] - sum_of_stables - unstable_piece_count) /
                       (initial_piece_count + sum_of_stables))
    score = (
            parity_score * 0.12 +
            corner_score * 0.35 +
            actual_mobility_score * 0.15 + potential_mobility_score * 0.12 +
            stability_score * 0.26
             )
    score = 32 * (score + 1)
    return score

# Abbreviation
better = betterEvaluationFunction