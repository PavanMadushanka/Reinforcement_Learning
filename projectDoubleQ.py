# Class to store MDP information
import random
from pprint import pprint
from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt

class MDP():

    # Using the sample in https://banay.me/maximisation-bias-q-learning/#fn:1
    def __init__(self):
        self.states = [1,2,3]
        self.actions = ['L','R']
        self.states_and_actions = self.get_states_and_actions()
        self.terminal_states = [3]  # Use brackets even if there is only one terminal state
        
    # Combine two dictionaries
    def combine_dictionaries(self, dict1, dict2):
        return {**dict1, **dict2}

    def get_P(self,state,action):
        # This should return (reward, next_state)
        # This MDP sample is found in sutton book https://banay.me/maximisation-bias-q-learning/#fn:1

        # Under if, put the state, action combinations that result in randomR
        if state==2:
            next_state = 3
            reward = norm(-0.1, 1).rvs()  # norm(mean, std. deviation) !!not variance!

        else:
            # Set the fixed next states and rewards
            # for s in self.states:
            #     for a in self.actions:
            #         P_temp[s,a] = (1/s, random.sample(self.states, 1)[0])
            if state==1:
                if action=='L':
                    reward = 0
                    next_state = 2
                if action=='R':
                    reward = 0
                    next_state = 3

        return (reward, next_state)
       
    def get_states_and_actions(self):
        sa = {}
        sa[1] = ['L','R']
        sa[2] = [i for i in range(10)]
        sa[3] = []  # Terminal state
        return sa


# Function to create equiprobable policy
def equiprobable_policy(states_and_actions):
    return {(s,a):1/len(mdp.states_and_actions[s]) for s in mdp.states for a in mdp.states_and_actions[s]}

# Pick an action with probability
def pick_action_with_probability(ap):
    actions = list(ap.keys())
    # print(actions)
    prob = [ap[k] for k in actions]
    # print(prob)
    i = np.random.choice(len(actions), 1, p=prob)
    return actions[i[0]]

# Simulate one epoch in the MDP
def simulate_mdp_step(mdp, policy, state):
    actions_and_probabilities = {k[1]: policy[k]
                                 for k in policy.keys() if k[0] == state}
    a = pick_action_with_probability(actions_and_probabilities)
    reward, next_state = mdp.get_P(state, a)
    # pprint(("Step generated : ", a, reward, next_state))
    return (a, reward, next_state)

def epsilon_greedy_policy(states_and_actions, Q, epsilon):
    # policy --> (state,action) : probability
    policy = {((state), action): epsilon/len(states_and_actions[state])
              for state in states_and_actions.keys() for action in states_and_actions[state]}
    for state in states_and_actions.keys():
        
        # If terminal state is present, skip creating policy for that state
        if len(states_and_actions[state])==0:
           continue
        
        max_action = None

        # find max action and break max actions randomly
        
        # pprint([Q[state,action] for action in states_and_actions[state]])
        # pprint(np.max([Q[state,action] for action in states_and_actions[state]]))
        # pprint([action for action in states_and_actions[state]])
        max_actions = np.where(np.max([Q[state,action] for action in states_and_actions[state]]) ==  [Q[state,action] for action in states_and_actions[state]] )[0]
        # pprint(max_actions)
        max_action = states_and_actions[state][np.random.choice(max_actions)]
        # pprint(max_action)
        # for action in states_and_actions[state]:
        #     if max == None:
        #         max = Q[state, action]
        #         max_action = action
        #     else:
        #         if Q[state, action] > max:
        #             max_action = action
        #             max = Q[state, action]
        # pprint(states_and_actions[state])
        policy[state, max_action] = 1 - epsilon + \
            epsilon/len(states_and_actions[state])
    # pprint(policy)
    return policy

def Combine_Q(Q1,Q2,mdp):
    return {((state), action): Q1[(state),action]+Q2[(state),action] for state in mdp.states_and_actions.keys()
         for action in mdp.states_and_actions[state]}

def Double_Q_Learning(mdp, gamma, N, epsilon, alpha=0.05):
    # Initialize Q1 and Q2 with all values to zero
    Q1 = {((state), action): 0 for state in mdp.states_and_actions.keys()
         for action in mdp.states_and_actions[state]}
    Q2 = {((state), action): 0 for state in mdp.states_and_actions.keys() for action in mdp.states_and_actions[state]}
    
    Q_sequence = [Q1.copy()]
    ARS_seq_list = []
    for episode in range(N):

        if (episode+1) % (N/5) == 0:
            alpha = alpha/2
            #print("Alpha updated to {}".format(alpha))

        # pick a state at random
        s = random.sample(mdp.states, 1)[0]
        while(s in mdp.terminal_states):
            s = random.sample(mdp.states, 1)[0]
        ARS_seq = [(None,None,s)]
        # print("State: {}".format(s))

        for i in range(1000):

            policy = epsilon_greedy_policy(
                mdp.states_and_actions, Combine_Q(Q1,Q2,mdp), epsilon)  # update the policy
            (a, r, s_next) = simulate_mdp_step(mdp, policy, s)
            ARS_seq.append((a,r,s_next))
            # print("S_next: {}".format(s_next))
            # Q-Learning update
            # Handle the terminal state separately as max() function does not support empty lists
            if(s_next in mdp.terminal_states):
                if(np.random.rand() < 0.5):
                    Q1[s, a] = Q1[s, a] + alpha*(r - Q1[s, a])
                else:
                    Q2[s, a] = Q2[s, a] + alpha*(r - Q2[s, a])
            else:
                if (np.random.rand() < 0.5):
                    max_action_Q1 = [a_t2 for a_t2 in mdp.states_and_actions[s_next] if Q1[s_next,a_t2]==max([Q1[s_next,a_t] for a_t in mdp.states_and_actions[s_next]])][0]
                    #pprint({act:Q1[s_next,act] for act in mdp.states_and_actions[s_next]})
                    #print("max action Q1: {}".format(max_action_Q1))    
                    Q1[s, a] = Q1[s, a] + alpha*(r + gamma*Q2[s_next,max_action_Q1] - Q1[s, a])
                else:
                    max_action_Q2 = [a_t2 for a_t2 in mdp.states_and_actions[s_next] if Q2[s_next,a_t2]==max([Q2[s_next,a_t] for a_t in mdp.states_and_actions[s_next]])][0]
                    #pprint({act:Q2[s_next,act] for act in mdp.states_and_actions[s_next]})
                    #print("max action Q2: {}".format(max_action_Q2))    
                    Q2[s, a] = Q2[s, a] + alpha*(r + gamma*Q1[s_next,max_action_Q2] - Q2[s, a])

            # Handle the terminal state
            if not (s_next in mdp.terminal_states):
                s = s_next
            else:
                # print("{} : Reached the goal in {} steps".format(episode+1, i))
                break
        # pprint(Q)
        Q_sequence.append(Combine_Q(Q1,Q2,mdp))
        ARS_seq_list.append(ARS_seq)

    #return Q_sequence
    return ARS_seq_list

def Q_Learning(mdp, gamma, N, epsilon, alpha=0.05):
    Q = {((state), action): 0 for state in mdp.states_and_actions.keys()
         for action in mdp.states_and_actions[state]}
    Q_sequence = [Q.copy()]
    ARS_seq_list = []
    for episode in range(N):

        # if (episode+1) % (N/5) == 0:
        #     alpha = alpha/2
            #print("Alpha updated to {}".format(alpha))

        # pick a state at random
        s = random.sample(mdp.states, 1)[0]
        while(s in mdp.terminal_states):
            s = random.sample(mdp.states, 1)[0]
        s = 1
        ARS_seq = [(None,None,s)]
        # print("State: {}".format(s))

        for i in range(1000):
            policy = epsilon_greedy_policy(
                mdp.states_and_actions, Q, epsilon)  # update the policy
            (a, r, s_next) = simulate_mdp_step(mdp, policy, s)
            ARS_seq.append((a,r,s_next))
            # print("S_next: {}".format(s_next))
            # Q-Learning update
            # Handle the terminal state separately as max() function does not support empty lists
            if(s_next in mdp.terminal_states):
                Q[s, a] = Q[s, a] + alpha*(r - Q[s, a])
            else:    
                Q[s, a] = Q[s, a] + alpha*(r + gamma*max([Q[s_next,a_t] for a_t in mdp.states_and_actions[s_next]]) - Q[s, a])

            # Handle the terminal state
            if not (s_next in mdp.terminal_states):
                s = s_next
            else:
                # print("{} : Reached the goal in {} steps".format(episode+1, i))
                break
        # pprint(Q)
        #Q_sequence.append(Q.copy())
        ARS_seq_list.append(ARS_seq)

    #return Q_sequence
    return ARS_seq_list

def learn(iterations,learning_method,mdp,gamma,N,epsilon,alpha):
    for i in range(iterations):
        if i%100==0:
            print('{}:\t{}'.format(learning_method,i))
        ars_seq = learning_method(mdp,gamma,N,epsilon,alpha)
        #ars_seq = Double_Q_Learning(mdp,1,N=100,epsilon=0.5)

        #count the number of time the action 'L' is taken
        #pprint([[ars[0] for ars in ars_temp if ars[1]==0] for ars_temp in ars_seq])
        #pprint([[ars[0] for ars in ars_temp if ars[0]=='L' and ars[1]==0] for ars_temp in ars_seq])
        L_cnt = [100*len([ars[0] for ars in ars_temp if ars[0]=='L' and ars[1]==0]) for ars_temp in ars_seq ]
        R_cnt = [100*len([ars[0] for ars in ars_temp if ars[0]=='R' and ars[1]==0 ]) for ars_temp in ars_seq ]
        #pprint([L_cnt, R_cnt])
        if i ==0:
            L_average = L_cnt
            #R_average = R_cnt
        else:
            L_average = [(L_average[k]*i + L_cnt[k])/(i+1) for k in range(len(L_cnt)) ]
            #R_average = [(R_average[k]*i + R_cnt[k])/(i+1) for k in range(len(R_cnt)) ]

    return L_average



# Main code to simulate the MDP
mdp = MDP()
policy = equiprobable_policy(mdp.states_and_actions)

# Get the average of multiple runs

i=100
Q_L = learn(iterations=i,learning_method=Q_Learning,mdp=mdp,gamma=1,N=300,epsilon=0.1,alpha=0.1)
dQ_L = learn(iterations=i,learning_method=Double_Q_Learning,mdp=mdp,gamma=1,N=300,epsilon=0.1,alpha=0.1)

plt.plot(Q_L,linewidth=1, label='With Q-Learning')
plt.plot(dQ_L,linewidth=1, label='With double Q-Learning')

#plt.plot(R_average,linewidth=1, label='Avarage count of action \'R\'')
plt.legend()
plt.xlabel('Episodes')
plt.ylabel('\'L\' action choice percentage')
plt.title('Average \'L\' action choice variation based on {} runs'.format(i))
plt.show()












# for i in range(100):
#     if i%100==0:
#         print(i)

#     #q_seq = Q_Learning(mdp,1,N=250,epsilon=0.1)
#     # q_seq = Double_Q_Learning(mdp,1,N=250,epsilon=0.1)
#     # if i==0:
#     #     val_array = {a:[q[a] for q in q_seq ] for a in q_seq[0].keys()}
#     # else:
#     #     temp_array = {a:[q[a] for q in q_seq ] for a in q_seq[0].keys()}
#     #     # Running average of the values
#     #     val_array = {a:[(i*temp_array[a][q] + val_array[a][q])/(i+1) for q in range(len(temp_array[a]))] for a in temp_array.keys()}

#     ars_seq = Q_Learning(mdp,gamma=1,N=300,epsilon=0.1,alpha=0.1)
#     #ars_seq = Double_Q_Learning(mdp,1,N=100,epsilon=0.5)

#     #count the number of time the action 'L' is taken
#     #pprint([[ars[0] for ars in ars_temp if ars[1]==0] for ars_temp in ars_seq])
#     #pprint([[ars[0] for ars in ars_temp if ars[0]=='L' and ars[1]==0] for ars_temp in ars_seq])
#     L_cnt = [100*len([ars[0] for ars in ars_temp if ars[0]=='L' and ars[1]==0]) for ars_temp in ars_seq ]
#     R_cnt = [100*len([ars[0] for ars in ars_temp if ars[0]=='R' and ars[1]==0 ]) for ars_temp in ars_seq ]
#     #pprint([L_cnt, R_cnt])
#     if i ==0:
#         L_average = L_cnt
#         R_average = R_cnt
#     else:
#         L_average = [(L_average[k]*i + L_cnt[k])/(i+1) for k in range(len(L_cnt)) ]
#         R_average = [(R_average[k]*i + R_cnt[k])/(i+1) for k in range(len(R_cnt)) ]

############################################################################
# plot the variation of Q

# plt.plot(val_array[1,'L'], linewidth=1, label='State:1, Action:\'L\'')
# plt.plot(val_array[1,'R'], linewidth=1, label='State:1, Action:\'R\'')
# plt.legend()
# plt.title('Avarage Q value variation based on {} runs'.format(i+1))
# plt.show()





