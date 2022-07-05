#%% parse file (state, action, reward)
import pandas

dataframe = pandas.read_csv('Ass4Data.txt', header=None, sep='\s',names=['state','action','reward'], engine='python')
print(dataframe.head())

# %%
states = dataframe.state.unique()
actions = dataframe.action.unique()
states_and_actions = dataframe[['state','action']].drop_duplicates().sort_values(by='state')

# %%
## Q Learning
alpha = 0.05

# Q = {((state), action): 0 for state in game.states_and_actions.keys()
#         for action in game.states_and_actions[state]}
Q = {}
for step in range(10): #range(dataframe.shape[0]-1):

    if (step+1) % 10000 == 0:
        alpha = alpha/2
        #print("Alpha updated to {}".format(alpha))

    # Read the MDP step from the record
    (s,a,r1) = dataframe.iloc[step].values.tolist()
    (s_next,a_next,r2) = dataframe.iloc[step+1].values.tolist()
    print("state:{}\taction:{}\treward:{}\tnext_state:{}".format(s,a,r,s_next))
  

    #Q[s, a] = Q[s, a] + alpha*(r + game.gamma*max([Q[s_next,a_t] for a_t in game.states_and_actions[s_next]]) - Q[s, a])
    Q[s,a] = Q[s,a] + alpha(r + max)

    # snl.print_matrix({state: Q[state, action] for (state, action) in Q.keys(
    # ) if Q[state, action] == max({Q[state, a_t] for (s_t, a_t) in Q.keys() if s_t == state})})


# %%
