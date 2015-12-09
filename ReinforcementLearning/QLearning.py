import numpy as np


class QLearning:
    def __init__(self,
                 goal_reward=10,
                 obstacle_reward=-10,
                 states=[5, 5],
                 transitions=[(0, 1), (0, -1), (1, 0), (-1, 0)],
                 goal=[(4, 3)],
                 obstacle=[(2, 2), (3, 2), (4, 2)]):
        self.states = {}
        self.state_X_range = range(1, states[0] + 1)
        self.state_Y_range = range(1, states[1] + 1)
        self.transitions = transitions

        self.build_mdp_state_transitions(goal, obstacle)
        self.print_states()

    def train(self, rate=0.01, threshold=0.01, max_iteration=10000):
        change_accumulant = threshold
        iteration = 0
        while change_accumulant >= threshold and iteration < max_iteration:
            change_accumulant = 0
            iteration += 1

            for state in self.states.iterkeys():
                old_current_state_utility = self.states[state][0]
                new_current_state_utility = old_current_state_utility

                best_next_reward = -np.inf
                best_next_transition = None
                for transition in self.states[state][1].iterkeys():
                    transition_reward = self.states[state][1][transition][0]

                    if best_next_reward < transition_reward:
                        best_next_reward = transition_reward
                        best_next_transition = transition

                new_current_state_utility += rate * best_next_reward
                self.states[state][0] = new_current_state_utility

                change_accumulant += new_current_state_utility - old_current_state_utility
            print "change_accumulant", change_accumulant, "iteration", iteration

        self.print_states()

    def print_states(self):
        for state in self.states.iterkeys():
            print
            print "state", state, "utility", self.states[state][0]
            for transition in self.states[state][1].iterkeys():
                print transition, "->", self.states[state][1][transition]

    def build_mdp_state_transitions(self, goal, obstacle):
        for x in self.state_X_range:
            for y in self.state_Y_range:
                if (x, y) not in obstacle:
                    current_state = (x, y)

                    current_state_utility = 0
                    self.states[current_state] = [current_state_utility, {}]

                    for (x_, y_) in self.transitions:
                        if x + x_ in self.state_X_range and y + y_ in self.state_Y_range:  # If next state is a valid state
                            next_state = (x + x_, y + y_)
                            if current_state in goal:  # When in goal state, all transition will go to itself
                                self.states[current_state][1][(x_, y_)] = (0, (x, y))
                            elif next_state in obstacle:  # When going to an obstacle state, -10 reward and stay still
                                self.states[current_state][1][(x_, y_)] = (-10, (x, y))
                            elif next_state in goal:  # When going to goal state, +10 reward and move
                                self.states[current_state][1][(x_, y_)] = (10, next_state)
                            else:
                                self.states[current_state][1][(x_, y_)] = (0, next_state)


if __name__ == '__main__':
    m = QLearning(goal_reward=10,
                  obstacle_reward=-10,
                  states=[5, 5],
                  transitions=[(0, 1), (0, -1), (1, 0), (-1, 0)],
                  goal=[(4, 3)],
                  obstacle=[(2, 2), (3, 2), (4, 2)])
    m.train()
