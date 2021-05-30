"""
Multiclass Queuing Network scheduling policy optimization using
Proximal Policy Optimization method with Approximating Martingale-Process variance reduction
PPO:
https://arxiv.org/abs/1707.06347 (by Schulman et al., 2017)
Generalized Advantage Estimation:
https://arxiv.org/pdf/1506.02438.pdf (by Schulman et al., 2017)
Approximating Martingale-Process Method (by Henderson, Glynn, 2002):
https://web.stanford.edu/~glynn/papers/2002/HendersonG02.pdf
"""
import ray  # package for distributed computations
import numpy as np
from actor_utils import Policy
from value_function import NNValueFunction
from utils import Logger, Scaler
import os
import argparse
import NmodelDynamics as pn
import random
import datetime
import copy

from simulation import run_policy, run_weights

ray.init()

def diag_dot(A, B):
    # returns np.diag(np.dot(A, B))
    return np.einsum("ij,ji->i", A, B)

def add_disc_sum_rew(trajectories, policy, network, gamma, lam, scaler, iteration):
    """
    compute value function for further training of Value Neural Network
    :param trajectory: simulated data
    :param network: queuing network
    :param policy: current policy
    :param gamma: discount factor
    :param lam: lambda parameter in GAE
    :param scaler: normalization values
    """
    start_time = datetime.datetime.now()
    for trajectory in trajectories:

        if iteration!=1:

            values = trajectory['values']
            observes = trajectory['observes']
            unscaled_obs = trajectory['unscaled_obs']


            ###### compute expectation of the value function of the next state ###########
            probab_of_actions = policy.sample(observes) # probability of choosing actions according to a NN policy

            action_array = network.next_state_probN(unscaled_obs) # transition probabilities for fixed actions

            # expectation of the value function for fixed actions
            value_for_each_action_list = []
            for act in action_array:
                value_for_each_action_list.append(diag_dot(act, trajectory['values_set'].T))
            value_for_each_action = np.vstack(value_for_each_action_list)

            P_pi = diag_dot(probab_of_actions, value_for_each_action)  # expectation of the value function
            ##############################################################################################################

            # td-error computing
            tds_pi = trajectory['rewards'] - values + gamma*P_pi[:, np.newaxis]#gamma * np.append(values[1:], values[-1]), axis=0)#

            # value function computing for futher neural network training
            disc_sum_rew = discount(x=tds_pi,   gamma= lam*gamma, v_last = tds_pi[-1]) + values
        else:
            disc_sum_rew = discount(x=trajectory['rewards'],   gamma= gamma, v_last = trajectory['rewards'][-1])

        trajectory['disc_sum_rew'] = disc_sum_rew


    end_time = datetime.datetime.now()
    time_policy = end_time - start_time
    print('add_disc_sum_rew time:', int((time_policy.total_seconds() / 60) * 100) / 100., 'minutes')

    burn = 1 # cut the last 'burn' points of the generated trajectories

    unscaled_obs = np.concatenate([t['unscaled_obs'][:-burn] for t in trajectories])
    disc_sum_rew = np.concatenate([t['disc_sum_rew'][:-burn] for t in trajectories])
    if iteration ==1:
        scaler.update(np.hstack((unscaled_obs, disc_sum_rew)))
    scale, offset = scaler.get()
    observes = (unscaled_obs - offset[:-1]) * scale[:-1]
    disc_sum_rew_norm = (disc_sum_rew - offset[-1]) * scale[-1]
    if iteration ==1:
        for t in trajectories:
            t['observes'] = (t['unscaled_obs'] - offset[:-1]) * scale[:-1]

    return observes, disc_sum_rew_norm

def discount(x, gamma, v_last):
    """ Calculate discounted forward sum of a sequence at each point """
    disc_array = np.zeros((len(x), 1))
    disc_array[-1] = v_last
    for i in range(len(x) - 2, -1, -1):
        if x[i+1]!=0:
            disc_array[i] = x[i] + gamma * disc_array[i + 1]

    return disc_array


def add_value(trajectories, val_func, scaler, possible_states):
    """
    # compute value function from the Value Neural Network
    :param trajectory_whole: simulated data
    :param val_func: Value Neural Network
    :param scaler: normalization values
    :param possible_states: transitions that are possible for the queuing network
    """
    start_time = datetime.datetime.now()
    scale, offset = scaler.get()


    # approximate value function for trajectory_whole['unscaled_obs']
    for trajectory in trajectories:
        values = val_func.predict(trajectory['observes'])
        trajectory['values'] = values / scale[-1] + offset[-1]

        # approximate value function of the states where transitions are possible from trajectory_whole['unscaled_obs']
        values_set = np.zeros(( len(possible_states)+1, len(trajectory['observes'])))

        new_obs = (trajectory['unscaled_last'] - offset[:-1]) * scale[:-1]
        values = val_func.predict(new_obs)
        values = values / scale[-1] + offset[-1]
        values_set[-1] = np.squeeze(values)

        for count, trans in enumerate(possible_states):
            new_obs =(trajectory['unscaled_last'] + trans - offset[:-1]) * scale[:-1]
            values = val_func.predict(new_obs)
            values = values / scale[-1] + offset[-1]
            values_set[count] = np.squeeze(values)

        trajectory['values_set'] = values_set.T


    end_time = datetime.datetime.now()
    time_policy = end_time - start_time
    print('add_value time:', int((time_policy.total_seconds() / 60) * 100) / 100., 'minutes')

def build_train_set(trajectories, gamma, scaler):
    """
    # data pre-processing for training
    :param trajectory_whole:  simulated data
    :param scaler: normalization values
    :return: data for further Policy and Value neural networks training
    """

    for trajectory in trajectories:
        values = trajectory['values']

        unscaled_obs = trajectory['unscaled_obs']


        ###### compute expectation of the value function of the next state ###########
        action_array = network.next_state_probN(unscaled_obs) # transition probabilities for fixed actions

        # expectation of the value function for fixed actions
        value_for_each_action_list = []
        for act in action_array:
            value_for_each_action_list.append(diag_dot(act, trajectory['values_set'].T))
        value_for_each_action = np.vstack(value_for_each_action_list)
        ##############################################################################################################

        # # expectation of the value function w.r.t the actual actions in data
        distr_fir = np.eye(len(network.actions))[trajectory['actions_glob']]

        P_a = diag_dot(distr_fir, value_for_each_action)

        advantages = trajectory['rewards'] - values +gamma*P_a[:, np.newaxis]# gamma * np.append(values[1:], values[-1]), axis=0)  #
        trajectory['advantages'] = np.asarray(advantages)


    start_time = datetime.datetime.now()
    burn = 1


    unscaled_obs = np.concatenate([t['unscaled_obs'][:-burn] for t in trajectories])
    disc_sum_rew = np.concatenate([t['disc_sum_rew'][:-burn] for t in trajectories])

    scale, offset = scaler.get()
    actions = np.concatenate([t['actions'][:-burn] for t in trajectories])
    advantages = np.concatenate([t['advantages'][:-burn] for t in trajectories])
    observes = (unscaled_obs - offset[:-1]) * scale[:-1]
    advantages = advantages  / (advantages.std() + 1e-6) # normalize advantages



    # ########## averaging value function estimations over all data ##########################
    # states_sum = {}
    # states_number = {}
    # states_positions = {}
    #
    # for i in range(len(unscaled_obs)):
    #     if tuple(unscaled_obs[i]) not in states_sum:
    #         states_sum[tuple(unscaled_obs[i])] = disc_sum_rew[i]
    #         states_number[tuple(unscaled_obs[i])] = 1
    #         states_positions[tuple(unscaled_obs[i])] = [i]
    #
    #     else:
    #         states_sum[tuple(unscaled_obs[i])] +=  disc_sum_rew[i]
    #         states_number[tuple(unscaled_obs[i])] += 1
    #         states_positions[tuple(unscaled_obs[i])].append(i)
    #
    # for key in states_sum:
    #     av = states_sum[key] / states_number[key]
    #     for i in states_positions[key]:
    #         disc_sum_rew[i] = av
    # ########################################################################################

    end_time = datetime.datetime.now()
    time_policy = end_time - start_time
    print('build_train_set time:', int((time_policy.total_seconds() / 60) * 100) / 100., 'minutes')
    return observes,  actions, advantages, disc_sum_rew


def log_batch_stats(observes, actions, advantages, disc_sum_rew, logger, episode):
    # metadata tracking

    time_total = datetime.datetime.now() - logger.time_start
    logger.log({'_mean_act': np.mean(actions),
                '_mean_adv': np.mean(advantages),
                '_min_adv': np.min(advantages),
                '_max_adv': np.max(advantages),
                '_std_adv': np.var(advantages),
                '_mean_discrew': np.mean(disc_sum_rew),
                '_min_discrew': np.min(disc_sum_rew),
                '_max_discrew': np.max(disc_sum_rew),
                '_std_discrew': np.var(disc_sum_rew),
                '_Episode': episode,
                '_time_from_beginning_in_minutes': int((time_total.total_seconds() / 60) * 100) / 100.
                })

# TODO: check shadow name
def main(network, num_policy_iterations, no_of_actors, episode_duration, no_arrivals, gamma, lam, clipping_parameter,
         ep_v, bs_v, lr_v, ep_p, bs_p, lr_p, kl_targ, hid1_mult):
    """
    # Main training loop
    :param: see ArgumentParser below
    """

    obs_dim = network.buffers_num
    act_dim = network.action_size_per_buffer
    now = datetime.datetime.utcnow().strftime("%b-%d_%H-%M-%S")  # create unique directories
    time_start= datetime.datetime.now()
    logger = Logger(logname=network.network_name, now=now, time_start=time_start)


    scaler = Scaler(obs_dim + 1)
    val_func = NNValueFunction(obs_dim, hid1_mult, ep_v, bs_v, lr_v) # Value Neural Network initialization
    policy = Policy(obs_dim, act_dim, kl_targ, hid1_mult, ep_p, bs_p, lr_p, clipping_parameter) # Policy Neural Network initialization


    ############## creating set of initial states for episodes in simulations##########################
    run_policy(network, policy, scaler, logger, gamma, policy_iter_num=0, no_episodes=1, time_steps=episode_duration)
    ###########################################################################

    iteration = 0  # count of policy iterations
    weights_set = []
    scaler_set = []
    while iteration < num_policy_iterations:
        # decrease clipping_range and learning rate
        iteration += 1
        alpha = 1. - iteration / num_policy_iterations
        policy.clipping_range = max(0.01, alpha*clipping_parameter)
        policy.lr = max(0.05, alpha)*lr_p
        # save policy NN parameters each 10th iteration
        if iteration % 10 == 1:
            weights_set.append(policy.get_weights())
            scaler_set.append(copy.copy(scaler))

        # simulate trajectories
        trajectories = run_policy(network, policy, scaler, logger, gamma, iteration,
                                      no_episodes=no_of_actors, time_steps=episode_duration)  #simulation
        # add estimated values to episodes
        add_value(trajectories, val_func, scaler, network.next_state_list())
        # calculate values from data
        observes, disc_sum_rew_norm = add_disc_sum_rew(trajectories, policy, network, gamma, lam, scaler, iteration)
        # update value function
        val_func.fit(observes, disc_sum_rew_norm, logger)
        # add estimated values to episodes
        add_value(trajectories, val_func, scaler, network.next_state_list())
        # compute advantage function estimates
        observes, actions, advantages, disc_sum_rew = build_train_set(trajectories, gamma, scaler)
        log_batch_stats(observes, actions, advantages, disc_sum_rew, logger, iteration)  # add various stats

        policy.update(observes, actions, np.squeeze(advantages), logger)  # update policy

        logger.write(display=True)  # write logger results to file and stdout

    ########## save policy NN parameters of the final policy and normalization parameters ##############################
    weights = policy.get_weights()
    file_weights = os.path.join(logger.path_weights, 'weights_' + str(iteration) + '.npy')
    np.save(file_weights, weights)

    file_scaler = os.path.join(logger.path_weights, 'scaler_' + str(iteration) + '.npy')
    scale, offset = scaler.get()
    np.save(file_scaler, np.asarray([scale, offset]))
    weights_set.append(policy.get_weights())
    scaler_set.append(copy.copy(scaler))
    #####################################################################

    performance_evolution_all, ci_all = run_weights(network, weights_set, policy, scaler,
                                        time_steps=int(1. / sum(network.p_arriving) * no_arrivals))


    file_res = os.path.join(logger.path_weights,
                            'average_' + str(performance_evolution_all[-1]) + '+-' +str(ci_all[-1]) + '.txt')
    file = open(file_res, "w")
    for i in range(len(ci_all)):
        file.write(str(performance_evolution_all[i])+'\n')
    file.write('\n')
    for i in range(len(ci_all)):
        file.write(str(ci_all[i])+'\n')


    logger.close()





if __name__ == "__main__":

    network = pn.ProcessingNetwork.Nmodel_from_load(load=0.95)

    parser = argparse.ArgumentParser(description=('Train policy for a queueing network '
                                                  'using Proximal Policy Optimizer'))

    parser.add_argument('-n', '--num_policy_iterations', type=int, help='Number of policy iterations to run',
                        default = 200)
    parser.add_argument('-b', '--no_of_actors', type=int, help='Number of episodes per training batch',
                        default = 2)
    parser.add_argument('-t', '--episode_duration', type=int, help='Number of time-steps per an episode',
                        default = 10**3)
    parser.add_argument('-x', '--no_arrivals', type=int, help='Number of arrivals to evaluate policies',
                        default = 5*10**6)

    parser.add_argument('-g', '--gamma', type=float, help='Discount factor',
                        default = 0.998)
    parser.add_argument('-l', '--lam', type=float, help='Lambda for Generalized Advantage Estimation',
                        default = 0.992)
    parser.add_argument('-c', '--clipping_parameter', type=float, help='Initial clipping parameter',
                        default = 0.2)

    parser.add_argument('-e', '--ep_v', type=float, help='number of epochs for value NN training',
                        default = 10)
    parser.add_argument('-s', '--bs_v', type=float, help='minibatch size for value NN training',
                        default = 10)
    parser.add_argument('-r', '--lr_v', type=float, help='learning rate for value NN training',
                        default = 2.5 * 10**(-4))

    parser.add_argument('-p', '--ep_p', type=float, help='number of epochs for policy NN training',
                        default = 3)
    parser.add_argument('-w', '--bs_p', type=float, help='minibatch size for policy NN training',
                        default = 3)
    parser.add_argument('-q', '--lr_p', type=float, help='learning rate for policy NN training',
                        default = 5. * 10 ** (-4))

    parser.add_argument('-k', '--kl_targ', type=float, help='D_KL target value',
                        default = 0.003)
    parser.add_argument('-m', '--hid1_mult', type=int, help='Size of first hidden layer for value and policy NNs',
                        default = 10)


    args = parser.parse_args()
    main(network,  **vars(args))