import numpy as np
import NmodelDynamics as pn


## N model parameters
rho = 0.9 # load

N = 200 # we assume that each buffer can hold up to N jobs. It is a simplified assumtion allowing the use of the value iteration.
lambda_1 = 1.3 * rho # class 1 jobs arrival rate
lambda_2 = 0.4 * rho # class 2 jobs arrival rate
m_1 = 1 # service completion rate of class 1 jobs by server 1
m_2 = 1 / 2 # service completion rate of class 1 jobs by server 2
m_3 = 1 # service completion rate of class 2 jobs by server 2
nu = lambda_1 + lambda_2 + m_1 + m_2 + m_3 # uniformization constant




### Value iteration algorithm#######################

V = dict() # value function. Given a state (x_1, x_2), where x_1<N and x_2<N, V[x_1,x_2] returns the relative value function at this state
action = dict() # optimal actions dictionary
V_temp = dict()

for i in range(N+1):
    for j in range(N+1):
        V[(i, j)] = 0
        action[(i, j)] = 1
        V_temp[(i, j)] = 0

# value iteration
for iteration in range(2000):
    print(iteration)
    for i in range(0, N+1):
        for j in range(0, N+1):
            if i == 0 and j == 0:
                V_temp[(i, j)] = 3 * i + j + lambda_1 * V[(i + 1, j)] / nu + lambda_2 * V[(i, j + 1)] / nu + (
                            m_1 + m_2) * V[(i, j)] / nu + (m_3) * V[(i, j)] / nu
            elif i == 0:
                if V[(i, j)] < V[(i, j - 1)]:
                    action[(i, j)] = 1
                    if j == N:
                        V_temp[(i, j)] = 3 * i + j + lambda_1 * V[(i + 1, j)] / nu + lambda_2 * V[(i, j)] / nu + (
                                    m_1 + m_2) * V[(i, j)] / nu + (m_3) * V[(i, j)] / nu
                    else:
                        V_temp[(i, j)] = 3 * i + j + lambda_1 * V[(i + 1, j)] / nu + lambda_2 * V[(i, j + 1)] / nu + (
                                    m_1 + m_2) * V[(i, j)] / nu + (m_3) * V[(i, j)] / nu
                else:
                    action[(i, j)] = 2
                    if j == N:
                        V_temp[(i, j)] = 3 * i + j + lambda_1 * V[(i + 1, j)] / nu + lambda_2 * V[(i, j)] / nu + m_1 * \
                                         V[(i, j)] / nu + m_2 * V[(i, j)] / nu + m_3 * V[(i, j - 1)] / nu
                    else:
                        V_temp[(i, j)] = 3 * i + j + lambda_1 * V[(i + 1, j)] / nu + lambda_2 * V[
                            (i, j + 1)] / nu + m_1 * V[(i, j)] / nu + m_2 * V[(i, j)] / nu + m_3 * V[(i, j - 1)] / nu
            elif j == 0:
                if V[(i - 1, j)] < V[(i, j)]:
                    action[(i, j)] = 1
                    if i == N:
                        V_temp[(i, j)] = 3 * i + j + lambda_1 * V[(i, j)] / nu + lambda_2 * V[(i, j + 1)] / nu + (
                                    m_1 + m_2) * V[(i - 1, j)] / nu + (m_3) * V[(i, j)] / nu
                    else:
                        V_temp[(i, j)] = 3 * i + j + lambda_1 * V[(i + 1, j)] / nu + lambda_2 * V[(i, j + 1)] / nu + (
                                    m_1 + m_2) * V[(i - 1, j)] / nu + (m_3) * V[(i, j)] / nu
                else:
                    action[(i, j)] = 2
                    if i == N:
                        V_temp[(i, j)] = 3 * i + j + lambda_1 * V[(i, j)] / nu + lambda_2 * V[(i, j + 1)] / nu + m_1 * \
                                         V[(i - 1, j)] / nu + m_2 * V[(i, j)] / nu + m_3 * V[(i, j)] / nu
                    else:
                        V_temp[(i, j)] = 3 * i + j + lambda_1 * V[(i + 1, j)] / nu + lambda_2 * V[
                            (i, j + 1)] / nu + m_1 * V[(i - 1, j)] / nu + m_2 * V[(i, j)] / nu + m_3 * V[(i, j)] / nu
            else:
                if (m_2 * V[(i - 1, j)] + m_3 * V[(i, j)]) < (m_2 * V[(i, j)] + m_3 * V[(i, j - 1)]):
                    action[(i, j)] = 1
                    if i == N and j == N:
                        V_temp[(i, j)] = 3 * i + j + lambda_1 * V[(i, j)] / nu + lambda_2 * V[(i, j)] / nu + (
                                    m_1 + m_2) * V[(i - 1, j)] / nu + (m_3) * V[(i, j)] / nu
                    elif i == N:
                        V_temp[(i, j)] = 3 * i + j + lambda_1 * V[(i, j)] / nu + lambda_2 * V[(i, j + 1)] / nu + (
                                    m_1 + m_2) * V[(i - 1, j)] / nu + (m_3) * V[(i, j)] / nu
                    elif j == N:
                        V_temp[(i, j)] = 3 * i + j + lambda_1 * V[(i + 1, j)] / nu + lambda_2 * V[(i, j)] / nu + (
                                    m_1 + m_2) * V[(i - 1, j)] / nu + (m_3) * V[(i, j)] / nu
                    else:
                        V_temp[(i, j)] = 3 * i + j + lambda_1 * V[(i + 1, j)] / nu + lambda_2 * V[(i, j + 1)] / nu + (
                                    m_1 + m_2) * V[(i - 1, j)] / nu + (m_3) * V[(i, j)] / nu
                else:
                    action[(i, j)] = 2
                    if i == N and j == N:
                        V_temp[(i, j)] = 3 * i + j + lambda_1 * V[(i, j)] / nu + lambda_2 * V[(i, j)] / nu + m_1 * V[
                            (i - 1, j)] / nu + m_2 * V[(i, j)] / nu + m_3 * V[(i, j - 1)] / nu
                    elif i == N:
                        V_temp[(i, j)] = 3 * i + j + lambda_1 * V[(i, j)] / nu + lambda_2 * V[(i, j + 1)] / nu + m_1 * \
                                         V[(i - 1, j)] / nu + m_2 * V[(i, j)] / nu + m_3 * V[(i, j - 1)] / nu
                    elif j == N:
                        V_temp[(i, j)] = 3 * i + j + lambda_1 * V[(i + 1, j)] / nu + lambda_2 * V[(i, j)] / nu + m_1 * \
                                         V[(i - 1, j)] / nu + m_2 * V[(i, j)] / nu + m_3 * V[(i, j - 1)] / nu
                    else:
                        V_temp[(i, j)] = 3 * i + j + lambda_1 * V[(i + 1, j)] / nu + lambda_2 * V[
                            (i, j + 1)] / nu + m_1 * V[(i - 1, j)] / nu + m_2 * V[(i, j)] / nu + m_3 * V[
                                             (i, j - 1)] / nu

    a = V_temp[(0, 0)]
    for i in range(0, N+1):
        for j in range(0, N+1):
            V[(i, j)] = V_temp[(i, j)] - a


#########################################################################



# saving optimal actions
action_matr = np.zeros((N+1, N+1))

for i in range(0, N+1):
    for j in range(0, N+1):
        action_matr[i, j] = action[(i, j)]


# we save optimal actions for all states (x_1,x_2) such that x_1<N and x_2<N.
# action[(x_1, x_2)] = 1 means class 1 jobs have priority
# action[(x_1, x_2)] = 2 means class 2 jobs have priority
np.save('action09.npy', action_matr)




alpha = [1.3 * rho, 0.4 * rho]
mu = [1., 0.5, 1.]
name = 'N'

network = pn.ProcessingNetwork.Nmodel_from_load(load=rho)


####### simulation of the optimal policy ################

state = [0, 0] # initial state
average_cost = 0
time = 20*10**6 # total time steps

h = [3, 1] # one-step cost

for t in range(time):
    try:
        if action[(state[0], state[1])] == 1:
            state = network.next_state_N1(state, [1, 0])
        else:
            state = network.next_state_N1(state, [0, 1])
    except:
        print('state', state, 'is outside of the truncation.')
        # we give priority for the buffers with more jobs
        if state[0] > state[1]:
            state = network.next_state_N1(state, [1, 0])
        else:
            state = network.next_state_N1(state, [0, 1])

    average_cost = 1./(t+1)*np.dot(h, state) + t/(t+1)*average_cost

    if t%10**5 == 0:
       print(t, average_cost)
print('load:', rho, 'average cost:', average_cost)


###########################################################