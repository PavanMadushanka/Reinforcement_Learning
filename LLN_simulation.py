# Assignment 3 | Reinforcement Learning
# Pavan Saranguhewa

# %%
import numpy as np
import matplotlib.pyplot as plt
import SnakeAndLadders as snl
import random
from pprint import pprint
import statistics


class LLN:
    def __init__(self, alpha, num_list, prob_list):
        self.alpha = alpha
        self.num_list = num_list
        self.prob_list = prob_list

    def step(self, n, prev):
        if prev == None:
            return self.sample_X()
        else:
            return (1-self.alpha(n))*prev + self.alpha(n)*self.sample_X()

    def sample_X(self):
        i = np.random.choice(len(self.num_list), 1, p=self.prob_list)
        return num_list[i[0]]


def simulate_game_step(game, policy, state):
    actions_and_probabilities = {k[1]: policy[k]
                                 for k in policy.keys() if k[0] == state}
    a = snl.pick_action_with_probability(actions_and_probabilities)
    reward, next_state = game.P[state, a]
    # pprint(("Step generated : ", a, reward, next_state))
    return (a, reward, next_state)


def TD0(game, policy, N, alpha=0.05):
    # initialize value function to zero
    # StateValue --> state : (value, count)
    StateValue = {state: (0, 0) for state in game.state_list}
    for episode in range(N):

        # Update alpha
        if (episode+1) % (N/10) == 0:
            alpha = alpha/2
            print("Alpha updated to {}".format(alpha))

        # # pick a starting state at random
        s = random.sample(game.P.keys(), 1)[0][0]

        # Simulate an episode upto 1000 steps
        for _ in range(1000):
            (_, r, s_next) = simulate_game_step(game, policy, s)
            # TD(0) update
            # pprint((StateValue[s], alpha, r, game.gamma, StateValue[s_next]))
            StateValue[s] = (StateValue[s][0] + alpha*(r + game.gamma *
                                                       StateValue[s_next][0] - StateValue[s][0]), StateValue[s][1] + 1)

            # Check for terminal state
            if not s_next == (1, 8):
                s = s_next
            else:
                # print("{} : Reached the goal in {} steps".format(episode+1, i))
                break

    print("TD(0) result for {} runs".format(N))
    snl.print_matrix({state: StateValue[state][0]
                      for state in StateValue.keys()})
    return StateValue


def epsilon_greedy_policy(states_and_actions, Q, epsilon):
    # policy --> (state,action) : probability
    policy = {((state), action): epsilon/len(states_and_actions[state])
              for state in states_and_actions.keys() for action in states_and_actions[state]}
    for state in states_and_actions.keys():
        if state == (1, 8):
            continue
        # pprint(state)
        max = None
        max_action = None
        for action in states_and_actions[state]:
            if max == None:
                max = Q[state, action]
                max_action = action
            else:
                if Q[state, action] > max:
                    max_action = action
                    max = Q[state, action]
        # pprint(states_and_actions[state])
        policy[state, max_action] = 1 - epsilon + \
            epsilon/len(states_and_actions[state])
    # pprint(policy)
    return policy


def SARSA(game, N, epsilon, alpha=0.05):
    Q = {((state), action): 0 for state in game.states_and_actions.keys()
         for action in game.states_and_actions[state]}

    for episode in range(N):

        if (episode+1) % (N/5) == 0:
            alpha = alpha/2
            #print("Alpha updated to {}".format(alpha))

        # Pick a starting state at random
        s = random.sample(game.P.keys(), 1)[0][0]

        for _ in range(1000):
            policy = epsilon_greedy_policy(
                game.states_and_actions, Q, epsilon)  # update the policy
            (a, r, s_next) = simulate_game_step(game, policy, s)
           
            # pprint((StateValue[s], alpha, r, game.gamma, StateValue[s_next]))
            # StateValue[s] = (StateValue[s][0] + alpha*(r + game.gamma *
            # StateValue[s_next][0] - StateValue[s][0]), StateValue[s][1] + 1)
            # SARSA update
            # Q[s, a] = Q[s, a] + alpha*(r + game.gamma*Q[s_next, a_next] - Q[s, a])
            # Expected SARSA update
            # if s == (2,8):
            #     print(s_next)
            #     print("policy")
            #     pprint([policy[s_next, a_t] for a_t in game.states_and_actions[s_next]])
            #     print("Q")
            #     pprint([Q[s_next, a_t] for a_t in game.states_and_actions[s_next]])
            #     pprint([policy[s_next, a_t]*Q[s_next, a_t] for a_t in game.states_and_actions[s_next]])
            # Expected SARSA update
            Q[s, a] = Q[s, a] + alpha*(r + game.gamma*sum(
                [policy[s_next, a_t]*Q[s_next, a_t] for a_t in game.states_and_actions[s_next]]) - Q[s, a])
            # if s == (2,8):
            #     print("Q[s, a] = Q[s, a] \t+ alpha*(r \t+ game.gamma\t*sum([policy[s_next, a_t]*Q[s_next, a_t] for a_t in game.states_and_actions[s_next]]) \t- Q[s, a])")
            #     print("Q[s, a] = {} \t+{}*({} \t+ {}\t*{} \t- Q[s, a])".format(Q[s, a],alpha,r,game.gamma,sum([policy[s_next, a_t]*Q[s_next, a_t] for a_t in game.states_and_actions[s_next]])))
            
            # Check for terminal state
            if not s_next == (1, 8):
                s = s_next
            else:
                #print("{} : Reached the goal in {} steps".format(episode+1, i))
                break
    snl.print_matrix({state: Q[state, action] for (state, action) in Q.keys(
    ) if Q[state, action] == max({Q[state, a_t] for (s_t, a_t) in Q.keys() if s_t == state})})
    return Q


def Q_Learning(game, N, epsilon, alpha=0.05):
    Q = {((state), action): 0 for state in game.states_and_actions.keys()
         for action in game.states_and_actions[state]}

    for episode in range(N):

        if (episode+1) % (N/5) == 0:
            alpha = alpha/2
            #print("Alpha updated to {}".format(alpha))

        # pick a state at random
        s = random.sample(game.P.keys(), 1)[0][0]


        for _ in range(1000):
            policy = epsilon_greedy_policy(
                game.states_and_actions, Q, epsilon)  # update the policy
            (a, r, s_next) = simulate_game_step(game, policy, s)

            # pprint((StateValue[s], alpha, r, game.gamma, StateValue[s_next]))
            # StateValue[s] = (StateValue[s][0] + alpha*(r + game.gamma *
            # StateValue[s_next][0] - StateValue[s][0]), StateValue[s][1] + 1)
            # SARSA update
            # Q[s, a] = Q[s, a] + alpha*(r + game.gamma*Q[s_next, a_next] - Q[s, a])
            # Expected SARSA update
            # if s == (2,8):
            #     print(s_next)
            #     print("policy")
            #     pprint([policy[s_next, a_t] for a_t in game.states_and_actions[s_next]])
            #     print("Q")
            #     pprint([Q[s_next, a_t] for a_t in game.states_and_actions[s_next]])
            #     pprint([policy[s_next, a_t]*Q[s_next, a_t] for a_t in game.states_and_actions[s_next]])
            
            # pprint([Q[s_next,a_t] for a_t in game.states_and_actions[s_next]])
            # print(s_next)

            # Q-Learning update
            # Had to handle the terminal state separately as max() function does not support empty lists
            if(s_next == (1,8)):
                Q[s, a] = Q[s, a] + alpha*(r - Q[s, a])
            else:    
                Q[s, a] = Q[s, a] + alpha*(r + game.gamma*max([Q[s_next,a_t] for a_t in game.states_and_actions[s_next]]) - Q[s, a])
            # if s == (2,8):
            #     print("Q[s, a] = Q[s, a] \t+ alpha*(r \t+ game.gamma\t*sum([policy[s_next, a_t]*Q[s_next, a_t] for a_t in game.states_and_actions[s_next]]) \t- Q[s, a])")
            #     print("Q[s, a] = {} \t+{}*({} \t+ {}\t*{} \t- Q[s, a])".format(Q[s, a],alpha,r,game.gamma,sum([policy[s_next, a_t]*Q[s_next, a_t] for a_t in game.states_and_actions[s_next]])))
            # #if s == (1, 3):
            #    snl.print_matrix({state: Q[state, action] for (state, action) in Q.keys(
            #    ) if Q[state, action] == max({Q[state, a_t] for (s_t, a_t) in Q.keys() if s_t == state})})
            
            # Handle the terminal state
            if not s_next == (1, 8):
                s = s_next
            else:
                #print("{} : Reached the goal in {} steps".format(episode+1, i))
                break

    snl.print_matrix({state: Q[state, action] for (state, action) in Q.keys(
    ) if Q[state, action] == max({Q[state, a_t] for (s_t, a_t) in Q.keys() if s_t == state})})
    return Q

# Problems & Answers
# %%
# Problem 1 : Simulating LLN for alpha = (1+2n)/(n**2)
def alpha(n):
    return (1+2*n)/(n ** 2)


num_list = [2, 4, 8, 16]
prob_list = [0.25, 0.5, 0.2, 0.05]

lln = LLN(alpha, num_list, prob_list)


S_n_sequences = []
for seq in range(1000):
    print("Sequence:\t{}".format(seq))
    S_n = None
    current_seq = []
    for n in range(1, 50000):
        S_n = lln.step(n, S_n)
        current_seq.append(S_n)
    S_n_sequences.append(current_seq)
    # plt.plot(S_n_sequences[seq], linewidth=0.5)

# pprint(S_n_sequences)
# print("\n")
S_n_mean = []
S_n_var = []
for i in range(len(S_n_sequences[0])):
    # print({S_n_sequences[a][i] for a in range(len(S_n_sequences))})
    # pprint(statistics.mean({S_n_sequences[a][i] for a in range(len(S_n_sequences))}))
    S_n_mean.append(statistics.mean(
        {S_n_sequences[a][i] for a in range(len(S_n_sequences))}))
    S_n_var.append(statistics.variance(
        {S_n_sequences[a][i] for a in range(len(S_n_sequences))}))

plt.plot(S_n_mean, linewidth=0.5, label='S_n mean')
plt.plot(S_n_var, linewidth=0.5, label='S_n variance')
plt.legend()
plt.xscale('log')
plt.ylim(-0.5,5.5)
plt.grid(color='gray', linestyle='-', linewidth=0.1, which='both', axis='both')
plt.title('Mean and Variance of S_n (with 1000 runs)')
plt.show()


# %%
# Problem 2 : Creating game in Assignment 1 with equiprobable policy
# Bulltet 1 : R = -0.25

print("Finding state value for R = -0.25")
R = -0.25
game = snl.SnakeAndLadders(R, gamma=0.98)
policy = snl.EquiprobablePolicy(game.states_and_actions)
# initialize value function to zero
StateValue = {state: 0 for state in game.state_list}
StateValue = snl.EvaluatePolicy(game, policy, StateValue)
snl.print_matrix(StateValue)

game = snl.SnakeAndLadders(R, gamma=0.98)
policy = snl.EquiprobablePolicy(game.states_and_actions)
StateValue = TD0(game, policy, N=100000)


# %%
# Bulltet 1 : R = 0.25

print("Finding state value for R = 0.25")
R = 0.25
game = snl.SnakeAndLadders(R, gamma=0.98)
policy = snl.EquiprobablePolicy(game.states_and_actions)
# initialize value function to zero
StateValue = {state: 0 for state in game.state_list}
StateValue = snl.EvaluatePolicy(game, policy, StateValue)
snl.print_matrix(StateValue)

game = snl.SnakeAndLadders(R, gamma=0.98)
policy = snl.EquiprobablePolicy(game.states_and_actions)
StateValue = TD0(game, policy, N=100000)



###################################################################################
# %%
# Bullet 2 : Optimal poicy using SARSA for R=-0.25

R = -0.25
game = snl.SnakeAndLadders(R, gamma=0.98)
policy_formatted, StateValue = snl.OptimizePolicy(game)
snl.print_matrix(policy_formatted)
snl.print_matrix(StateValue)

game = snl.SnakeAndLadders(R, gamma=0.98)
Q = SARSA(game, N=1000000, epsilon=0.1)
# pprint(Q)
snl.print_matrix({state: action for state in game.state_list for action in game.states_and_actions[state] if Q[state, action] == max(
    {Q[state, a_t] for a_t in game.states_and_actions[state]})})

# %%
# Bullet 2 : Optimal poicy using SARSA for R=0.25

R = 0.25
game = snl.SnakeAndLadders(R, gamma=0.98)
policy_formatted, StateValue = snl.OptimizePolicy(game)
snl.print_matrix(policy_formatted)
snl.print_matrix(StateValue)

game = snl.SnakeAndLadders(R, gamma=0.98)
Q = SARSA(game, N=1000000, epsilon=0.1)
# pprint(Q)
snl.print_matrix({state: action for state in game.state_list for action in game.states_and_actions[state] if Q[state, action] == max(
    {Q[state, a_t] for a_t in game.states_and_actions[state]})})

# %%
# Bullet 3 : Optimal poicy using SARSA for R=-0.25

R = -0.25
game = snl.SnakeAndLadders(R, gamma=0.98)
policy_formatted, StateValue = snl.OptimizePolicy(game)
snl.print_matrix(policy_formatted)
snl.print_matrix(StateValue)

game = snl.SnakeAndLadders(R, gamma=0.98)
Q = Q_Learning(game, N=1000000, epsilon=0.1)
# pprint(Q)
snl.print_matrix({state: action for state in game.state_list for action in game.states_and_actions[state] if Q[state, action] == max(
    {Q[state, a_t] for a_t in game.states_and_actions[state]})})
    

# %%
# Bullet 3 : Optimal poicy using SARSA for R=0.25
R = 0.25
game = snl.SnakeAndLadders(R, gamma=0.98)
policy_formatted, StateValue = snl.OptimizePolicy(game)
snl.print_matrix(policy_formatted)
snl.print_matrix(StateValue)

game = snl.SnakeAndLadders(R, gamma=0.98)
Q = Q_Learning(game, N=1000000, epsilon=0.1)
# pprint(Q)
snl.print_matrix({state: action for state in game.state_list for action in game.states_and_actions[state] if Q[state, action] == max(
    {Q[state, a_t] for a_t in game.states_and_actions[state]})})

