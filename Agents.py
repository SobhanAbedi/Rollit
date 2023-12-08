import pygame
import sys
import ast
import random


class Agent:
    """
        Base class for agents.
    """

    def __init__(self, index) -> None:
        self.index = index

    def getAction(self, state):
        """
            This method receives a GameState object and returns an action based on its strategy.
        """
        pass


class MouseAgent(Agent):

    def __init__(self, index, window_size, **kwargs) -> None:
        super().__init__(index)
        self.window_size = window_size

    def getAction(self, state):

        square_size = self.window_size // 8

        allowed_actions = state.getLegalActions(self.index)
        action = None

        while action is None:
            for event in pygame.event.get():
                if event.type == pygame.MOUSEBUTTONDOWN:
                    pos = pygame.mouse.get_pos()
                    x, y = pos[1] // square_size, pos[0] // square_size
                    if (x, y) in allowed_actions:
                        action = (x, y)
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

        return action


class KeyBoardAgent(Agent):

    def __init__(self, index, **kwargs) -> None:
        super().__init__(index)

    def getAction(self, state):

        allowed_actions = state.getLegalActions(self.index)

        print('allowed actions:')
        print(allowed_actions)

        action = None

        while action is None:

            temporary_input = input('please choose a tuple from the list above:\t')

            try:
                choice = ast.literal_eval(temporary_input)
            except Exception:
                print('your input must be in shape of a tuple.')
                continue

            if choice not in allowed_actions:
                print('your tuple must be one of the mentioned.')
                continue

            action = choice

        return action


class DestroyZeroAgent(Agent):
    def __init__(self, index, depth=1, **kwargs) -> None:
        super().__init__(index)
        self.depth = depth

    def getAction(self, state):
        """This function returns the ideal action for the given state"""
        action_value, action = self.chose_action(state, 0, self.index)
        return action

    def chose_action(self, state, cur_depth, index):
        """This function recursively finds the best action given the initial state and depth limit."""
        agent_count = state.getNumAgents()
        if state.isGameFinished() or cur_depth == self.depth:
            return self.evaluationFunction(state), ""
        action_list = state.getLegalActions(self.index)
        nxt_index = (index + 1) % agent_count
        nxt_depth = (cur_depth + 1) if nxt_index == self.index else cur_depth
        if index == 0:
            chosen_act_val = -999999
            chosen_act = ""
            for action in action_list:
                nxt_state = state.generateSuccessor(index, action)
                # There's a bug in the next line that I don't think was introduced by me. nxt_state should be changed to
                # nxt_action even though it's redundant
                nxt_value, nxt_state = self.chose_action(nxt_state, nxt_depth, nxt_index)
                if nxt_value > chosen_act_val:
                    chosen_act_val = nxt_value
                    chosen_act = action
        else:
            chosen_act_val = 999999
            chosen_act = ""
            for action in action_list:
                nxt_state = state.generateSuccessor(index, action)
                nxt_value, nxt_state = self.chose_action(nxt_state, nxt_depth, nxt_index)
                if nxt_value < chosen_act_val:
                    chosen_act_val = nxt_value
                    chosen_act = action
        return chosen_act_val, chosen_act

    def evaluationFunction(self, state):
        """This function returns score of the given state"""
        return state.getScore(0)


class MysteriousAgent(Agent):
    def __init__(self, index, depth=1, **kwargs) -> None:
        super().__init__(index)
        self.depth = depth

    def getAction(self, state):
        """This function returns the ideal action for the given state"""
        pos_inf = float('inf')
        neg_inf = -pos_inf
        min_val = pos_inf
        action_value, action = self.chose_action(state, 0, self.index, neg_inf, min_val)
        return action

    def chose_action(self, state, cur_depth, index, lower_threshold, upper_threshold):
        """
                This function recursively finds the best action given the initial state and depth limit.
                Alpha-Beta pruning is also implemented.
        """
        agent_count = state.getNumAgents()
        if state.isGameFinished() or cur_depth == self.depth:
            return (self.evaluationFunction(state), "")
        action_list = state.getLegalActions(index)
        nxt_index = (index + 1) % agent_count
        nxt_depth = (cur_depth + 1) if nxt_index == self.index else cur_depth
        if index == self.index:
            chosen_act_val = -999999
            chosen_act = ""
            for action in action_list:
                nxt_state = state.generateSuccessor(index, action)
                nxt_value, nxt_action = self.chose_action(nxt_state, nxt_depth, nxt_index, lower_threshold,
                                                          upper_threshold)
                if nxt_value > chosen_act_val:
                    chosen_act_val = nxt_value
                    chosen_act = action
                if chosen_act_val > upper_threshold:
                    return (chosen_act_val, "")
                lower_threshold = chosen_act_val if chosen_act_val > lower_threshold else lower_threshold
            return (chosen_act_val, chosen_act)
        else:
            chosen_act_val = 999999
            chosen_act = ""
            for action in action_list:
                nxt_state = state.generateSuccessor(index, action)
                nxt_value, nxt_action = self.chose_action(nxt_state, nxt_depth, nxt_index, lower_threshold,
                                                          upper_threshold)
                if nxt_value < chosen_act_val:
                    chosen_act_val = nxt_value
                    chosen_act = action
                if chosen_act_val < lower_threshold:
                    return (chosen_act_val, "")
                upper_threshold = chosen_act_val if chosen_act_val < upper_threshold else upper_threshold
            return (chosen_act_val, chosen_act)

    def evaluationFunction(self, state):
        """This function returns score of the given state"""
        return state.getScore(self.index)


class IntentionallyBadAtGameAgent(MysteriousAgent):
    def __init__(self, index, **kwargs) -> None:
        super().__init__(index, **kwargs)

    def getAction(self, state):
        """This function randomly returns one of the possible actions other than the ideal choice unless there's no other"""
        pos_inf = float('inf')
        neg_inf = -pos_inf
        min_val = pos_inf
        action_value, action = self.chose_action(state, 0, self.index, neg_inf, min_val)
        action_list = state.getLegalActions(self.index)
        if len(action_list) > 1:
            action_list.remove(action)  # Worse than random
        return random.choice(action_list)


class PartiallyRandomAgent(MysteriousAgent):
    def __init__(self, index, **kwargs) -> None:
        super().__init__(index, **kwargs)

    def getAction(self, state):
        """This function returns ideal action 25% of times and a random action rest of the time"""
        if random.random() > 0.25:
            pos_inf = float('inf')
            neg_inf = -pos_inf
            min_val = pos_inf
            action_value, action = self.chose_action(state, 0, self.index, neg_inf, min_val)
        else:
            action_list = state.getLegalActions(self.index)
            action = random.choice(action_list)
        return action
