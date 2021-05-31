import numpy as np
import NmodelDynamics as pn

rho = 0.9
N = 200
lambda_1 = 1.3 * rho
lambda_2 = 0.4 * rho
m_1 = 1
m_2 = 1 / 2
m_3 = 1
nu = lambda_1 + lambda_2 + m_1 + m_2 + m_3

V = dict()
action = dict()
V_temp = dict()

for i in range(N+1):
    for j in range(N+1):
        V[(i, j)] = 0
        action[(i, j)] = 1
        V_temp[(i, j)] = 0

for iteration in range(10000):
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

action_matr = np.zeros((N+1, N+1))

for i in range(0, N+1):
    for j in range(0, N+1):
        action_matr[i, j] = action[(i, j)]

np.save('action09.npy', action_matr)

alpha = [1.3 * rho, 0.4 * rho]
mu = [1., 0.5, 1.]
name = 'N'

network = pn.ProcessingNetwork.Nmodel_from_load(load=rho)

T = 0
state = [0, 0]
summ = 0
time = 2* 10**8 * int(sum(mu) /sum(alpha) +1 )
print('total time:', time)
for t in range(time):

    if action[(state[0], state[1])]==1:
        state = network.next_state_N1(state, 0)
    else:
        state = network.next_state_N1(state, 1)

    summ = 1./(t+1) * (3*state[0]+state[1])+t/(t+1)*summ

    if t% 10**5 == 0:
       print(t, summ)
print('answer', T, summ)
