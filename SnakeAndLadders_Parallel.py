# %%
from pprint import pprint
import numpy as np
from pandas import DataFrame
import random
import matplotlib.pyplot as plt
from multiprocessing import Process, Manager

# Class to hold the Snake & Ladders game environment


class SnakeAndLadders:
    def __init__(self, R):
        self.R = R
        self.gamma = 1
        self.columns = 8
        self.rows = 8
        # Special actions are specified with 'S'
        self.actions = ('R', 'L', 'U', 'D', 'S')
        self.state = (1, 1)
        self.state_list = self.get_all_states()
        self.P = self.set_rewards_and_next_states()
        self.states_and_actions = self.get_state_action_list()

        print("Game Created")

    def combine_dictionaries(self, dict1, dict2):
        return {**dict1, **dict2}

    def set_rewards_and_next_states(self):
        # Reward sceheme P(R,S'|S,A)
        # ( (state), action) : (reward, next state)

        # moving up
        P_up = {((state), 'U'): (-0.5, (state[0], state[1]+1))
                for state in self.state_list if state[1] != self.rows}  # Point 6
        P_up2 = {((state), 'U'): (-1, (state))
                 for state in self.state_list if state[1] == self.rows}  # Point 5
        P_up = self.combine_dictionaries(P_up, P_up2)

        # moving down
        P_down = {((state), 'D'): (-0.5, (state[0], state[1]-1))
                  for state in self.state_list if state[1] != 1}  # Point 6
        P_down2 = {((state), 'D'): (-1, (state))
                   for state in self.state_list if state[1] == 1}  # Point 5
        P_down = self.combine_dictionaries(P_down, P_down2)

        # moving left
        P_left = {((state), 'L'): (-0.5, (state[0]-1, state[1]))
                  for state in self.state_list if state[0] != 1}  # Point 6
        P_left2 = {((state), 'L'): (-1, (state))
                   for state in self.state_list if state[0] == 1}  # Point 5
        P_left = self.combine_dictionaries(P_left, P_left2)

        # moving right
        P_right = {((state), 'R'): (-0.5, (state[0] + 1, state[1]))
                   for state in self.state_list if state[0] != self.columns}  # Point 6
        P_right2 = {((state), 'R'): (-1, (state))
                    for state in self.state_list if state[0] == self.columns}  # Point 5
        P_right = self.combine_dictionaries(P_right, P_right2)

        P = {**P_up, **P_down, **P_left, **P_right}  # Combine the P's

        [P.pop(key) for key in [((3, 8), act)
                                for act in self.actions if act != 'S']]  # Point 4
        P = {**P, ((3, 8), 'S'): (-3, (7, 5))}

        [P.pop(key) for key in [((4, 5), act)
                                for act in self.actions if act != 'S']]  # Point 3
        P = {**P, ((4, 5), 'S'): (15, (1, 8))}

        [P.pop(key) for key in [((6, 1), act)
                                for act in self.actions if act != 'S']]  # Point 2
        P = {**P, ((6, 1), 'S'): (self.R, (3, 4))}

        [P.pop(key) for key in [((1, 8), act)
                                for act in self.actions if act != 'S']]  # Point 10
        P[((1, 7), 'U')] = (9.5, (1, 8))
        P[((2, 8), 'L')] = (9.5, (1, 8))

        # pprint(P)
        return P

    def get_state_action_list(self):
        sa = list(self.P.keys())

        dict_of_sa = dict()
        for i in self.state_list:
            dict_of_sa.update({i: [k[1] for k in sa if k[0] == i]})

        return dict_of_sa

    def set_actions(self, actions):
        self.actions = actions

    def set_state(self, state):
        self.state = state

    def get_all_states(self):
        return [(c+1, r+1) for c in range(self.columns) for r in range(self.rows)]

# Set of functions to use in dynamic programming


def EquiprobablePolicy(states_and_actions):
    # actions : R, L, U, D
    # Pi --> P(action|state)

    # Remove Up from top row
    for i in [k for k in states_and_actions.keys() if k[1] == 8]:
        states_and_actions[i] = [t for t in states_and_actions[i] if t != 'U']

    # Remove Down from bottom row
    for i in [k for k in states_and_actions.keys() if k[1] == 1]:
        states_and_actions[i] = [t for t in states_and_actions[i] if t != 'D']

    # Remove Left from first column
    for i in [k for k in states_and_actions.keys() if k[0] == 1]:
        states_and_actions[i] = [t for t in states_and_actions[i] if t != 'L']

    # Remove Right from last ccolumn
    for i in [k for k in states_and_actions.keys() if k[0] == 8]:
        states_and_actions[i] = [t for t in states_and_actions[i] if t != 'R']

    # return policy
    policy = {((state), action): 1/len(states_and_actions[state])
              for state in states_and_actions.keys() for action in states_and_actions[state]}
    return policy


def EvaluatePolicy(game, policy, value):

    theta = 1e-100
    gamma = game.gamma

    # initialize value function to zero
    StateValue = value
    i = 0
    delta = 100
    while(delta > theta and i < 10000):
        delta = 0
        for s in game.state_list:

            v = StateValue[s]

            game.states_and_actions
            # Reward sceheme P(R,S'|S,A)
            # ( (state), action) : (reward, next state)

            # v(s) = sigma { P_Pi(a|s)*P(r,s'|s,a)*(r + gamma*V(s'))}
            temp_val_array = [policy[s, a]*(game.P[s, a][0] + gamma*StateValue[game.P[s, a][1]])
                              for a in game.states_and_actions[s] if (s, a) in policy.keys()]
            StateValue[s] = sum(temp_val_array)
            delta = max(delta, abs(StateValue[s]-v))

        i = i + 1

    print("Policy Evalutation finished in {} steps".format(i))
    return StateValue


def ImprovePolicy(game, policy, value):
    gamma = game.gamma
    policy_is_stable = True

    # formatting the policy to match with the earlier format of policy ((s,a):P)
    policy_formatted = dict()
    for i in game.state_list:
        policy_formatted.update({i: j[1] for j in policy.keys() if j[0] == i})

    for s in policy_formatted.keys():
        old_a = policy_formatted[s]
        # policy(s) = argmax (sigma( P(s',r|s,a)*(r + gamma*V(s')) ))
        temp_array = [(game.P[s, a][0] + gamma*value[game.P[s, a][1]])
                      for a in game.states_and_actions[s]]
        # print(temp_array)
        indx = temp_array.index(max(temp_array))
        # print(indx)
        policy_formatted[s] = game.states_and_actions[s][indx]
        # pprint(policy_formatted)
        if policy_formatted[s] != old_a:
            policy_is_stable = False

    policy = {(s, policy_formatted[s]): 1 for s in policy_formatted.keys()}
    return policy_is_stable, policy, policy_formatted, value


def OptimizePolicy(game):

    # initialize policy to random policy
    policy = {(s, game.states_and_actions[s][0]): 1 for s in game.state_list if len(
        game.states_and_actions[s]) > 0}
    # initialize value function to zero
    StateValue = {state: 0 for state in game.state_list}

    policy_is_stable = False
    while not policy_is_stable:
        value = EvaluatePolicy(game, policy, StateValue)
        policy_is_stable, policy, policy_formatted, value = ImprovePolicy(
            game, policy, StateValue)
        print_matrix(policy_formatted)
        print_matrix(value)

    return policy_formatted, value


def print_matrix(StateValue):
    rows = max(k[1] for k in StateValue.keys())
    columns = max(k[0] for k in StateValue.keys())
    for i in range(columns):
        for j in range(rows):
            if not (j+1, i+1) in StateValue.keys():
                StateValue[(j+1, i+1)] = '-'

    array = [[StateValue[(r+1, columns-c)] for r in range(rows)]
             for c in range(columns)]
    # print(array)
    #list_as_array = np.round(np.array((array)),6)
    list_as_array = DataFrame(array)
    print(list_as_array)


# Set of functions for Monte Carlo
def SimulateGame_random(game, N):
    # ( (state), action) : (reward, next state)
    # start from a random state
    state_action = random.sample(game.P.keys(), 1)[0]

    # Start the list by adding the initial element
    SAR_sequence = [
        (state_action[0], state_action[1], game.P[state_action][0])]
    i = 1
    while i < N:
        # print(i)
        _, next_state = game.P[state_action]
        if(next_state == (1, 8)):
            SAR_sequence.append((next_state, 0, 0))
            break
        # print(next_state)
        available_actions = [k for k in game.P.keys() if k[0] == next_state]
        state_action = random.sample(available_actions, 1)[0]
        SAR_sequence.append(
            (state_action[0], state_action[1], game.P[state_action][0]))
        # print(state_action)
        i = i + 1
    print("Game simulated for {} steps".format(i))
    # pprint(SAR_sequence)
    return SAR_sequence, i


def pick_action_with_probability(ap):
    actions = list(ap.keys())
    prob = [ap[k] for k in actions]
    i = np.random.choice(len(actions), 1, p=prob)
    return actions[i[0]]


def SimulateGame_with_policy(game, N, policy):
    # ( (state), action) : (reward, next state)
    # start from a random state
    s = random.sample(game.P.keys(), 1)[0][0]
    actions_and_probabilities = {k[1]: policy[k]
                                 for k in policy.keys() if k[0] == s}
    a = pick_action_with_probability(actions_and_probabilities)

    # Start the list by adding the initial element
    SAR_sequence = [(s, a, game.P[s, a][0])]
    i = 1
    state_action = (s, a)
    while i < N:
        # print(i)
        _, next_state = game.P[state_action]
        if(next_state == (1, 8)):
            SAR_sequence.append((next_state, 0, 0))
            break
        # print(next_state)
        actions_and_probabilities = {k[1]: policy[k]
                                     for k in policy.keys() if k[0] == next_state}
        a = pick_action_with_probability(actions_and_probabilities)
        #available_actions = [k for k in game.P.keys() if k[0]==next_state]
        state_action = (next_state, a)
        SAR_sequence.append(
            (state_action[0], state_action[1], game.P[state_action][0]))
        # print(state_action)
        i = i + 1
    #print("Game simulated for {} steps".format(i))
    # pprint(SAR_sequence)
    return SAR_sequence, i

def run_game_in_parallel(StateValue, game, policy,val_3_5):
    Returns = {state: [] for state in game.state_list}
    G = 0
    if policy == False:
        seq, steps = SimulateGame_random(game, 1000)
    else:
        seq, steps = SimulateGame_with_policy(game, 1000, policy)

    unexplored_states = game.state_list.copy()
    for t in reversed(range(steps)):
        R = seq[t][2]
        state = seq[t][0]
        #print("State {}, \t Reward: {}".format(state,R))
        G = R + game.gamma*G

        # For every visit, uncomment the following
        #ret = Returns[state]
        # ret.append(G)
        #Returns[state] = ret

        # For last visit
        if state in unexplored_states:
            unexplored_states.remove(state)
            Returns[state] = [G]
        # print(Returns[state])

    # Recalculate StateValue
    #pprint(Returns[(4,5)])
    for state in game.state_list:
        if not len(Returns[state]) == 0:
            ct = StateValue[state][1]
            val = (StateValue[state][0]*ct +
                    np.average(Returns[state]))/(ct+1)
            StateValue[state] = (val, ct+1)
    #StateValue = {state : (StateValue[state]*i + np.average(Returns[state]))/(i+1) for state in game.state_list if len(Returns[state])!= 0}
    # print(StateValue[(4,5)])
    val_3_5.append(StateValue[(3,5)][0])


# %%
# MC simulation
if __name__ == "__main__":
    
    R = 0
    game = SnakeAndLadders(R)
    policy = EquiprobablePolicy(game.states_and_actions)
    N = 1000000
    # Last visit MC prediction
    # StateValue --> state : (value, count)
    print("Creating manager")
    StateValue = Manager().dict({state: (0, 0) for state in game.state_list})
    val_3_5 = Manager().list()
    print("Starting")
    procs = list()
    #val_3_5 = []
    for i in range(N):
        p = Process(target=run_game_in_parallel, args=(StateValue,game,policy,val_3_5))
        procs.append(p)
        p.start()
        #run_game_in_parallel(StateValue,game,policy)
        #val_3_5.append(StateValue[(3,5)][0])
    for p in procs:
        p.join()

    print(StateValue)
    print_matrix({state:StateValue[state][0] for state in StateValue.keys()})
    plt.plot(val_3_5)
    plt.show()
    # policy = EquiprobablePolicy(game.states_and_actions)
    # for i in range(2):
    #     SimulateGame_with_policy(game,1000,policy)


# # %%
# # Problem 1 in Dynamic Programming
# R = 0
# game = SnakeAndLadders(R)
# policy = EquiprobablePolicy(game.states_and_actions)
# # initialize value function to zero
# StateValue = {state: 0 for state in game.state_list}
# StateValue = EvaluatePolicy(game, policy, StateValue)
# print_matrix(StateValue)
# # print(policy)

# # %%
# # Problem 2

# R = -0.25
# game = SnakeAndLadders(R)
# policy_formatted, StateValue = OptimizePolicy(game)

# print_matrix(policy_formatted)
# print_matrix(StateValue)

# # %%
# # Problem 3

# R = (-1e-15, 1e-15)
# games = [SnakeAndLadders(i) for i in R]
# policies = list()
# for i in range(len(games)):
#     print("Evaluating R = {}".format(R[i]))
#     policy_formatted, value = OptimizePolicy(games[i])
#     policies.append(policy_formatted)
#     print_matrix(policy_formatted)
#     print_matrix(value)

# # %%
# # Problem 4
# R = 3
# game = SnakeAndLadders(R)
# policy_formatted, StateValue = OptimizePolicy(game)
# # %%
