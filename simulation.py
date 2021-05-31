import ray  # package for distributed computations
import numpy as np
from actor_utils import Policy
import os
import random

MAX_ACTORS = 4  # max number of parallel simulations

def run_policy(network, policy, scaler, logger, gamma, policy_iter_num, no_episodes, time_steps):
    """
    Run given policy and collect data
    :param network_id: queuing network structure and first-order info
    :param policy: queuing network policy
    :param scaler: normalization values
    :param logger: metadata accumulator
    :param gamma: discount factor
    :param policy_iter_num: policy iteration
    :param episodes: number of parallel simulations (episodes)
    :param time_steps: max time steps in an episode
    :return: trajectories = (states, actions, rewards)
    """

    total_steps = 0
    burn = 1 # cut the last 'burn' time-steps of episodes

    scale, offset = scaler.get()

    '''
    initial_states_set = random.sample(scaler.initial_states, k=no_episodes)
    trajectory, total_steps = policy.run_episode(network, scaler, time_steps,  initial_states_set[0])
    trajectories = []
    trajectories.append(trajectory)
    '''
    #### declare actors for distributed simulations of a current policy#####
    remote_network = ray.remote(Policy)


    simulators = [remote_network.remote(policy.nn.obs_dim, policy.nn.act_dim, policy.nn.hid1_mult, policy.kl_targ,
                                        policy.epochs, policy.batch_size, policy.lr, policy.clipping_range)
                  for _ in range(MAX_ACTORS)]

    actors_per_run = no_episodes // MAX_ACTORS  # do not run more parallel processes than number of cores
    remainder = no_episodes - actors_per_run * MAX_ACTORS
    weights = policy.get_weights()  # get neural network parameters
    for s in simulators:  # assign the neural network weights to all actors
        s.set_weights.remote(weights)
    ######################################################

    ######### save neural network parameters to file ###########
    file_weights = os.path.join(logger.path_weights, 'weights_' + str(policy_iter_num) + '.npy')
    np.save(file_weights, np.array(weights, dtype = object))
    ##################

    initial_states_set = random.sample(scaler.initial_states, k=no_episodes)  # sample initial states for episodes

    ######### policy simulation ########################
    accum_res = []  # results accumulator from all actors
    trajectories = []  # list of trajectories
    for j in range(actors_per_run):
        accum_res.extend(ray.get([simulators[i].run_episode.remote(network, scaler, time_steps,
                                                                   initial_states_set[j * MAX_ACTORS + i]) for i in
                                  range(MAX_ACTORS)]))
    if remainder > 0:
        accum_res.extend(ray.get([simulators[i].run_episode.remote(network, scaler, time_steps,
                                                                   initial_states_set[actors_per_run * MAX_ACTORS + i])
                                  for i in range(remainder)]))
    print('simulation is done')

    for i in range(len(accum_res)):
        trajectories.append(accum_res[i][0])
        total_steps += accum_res[i][1]  # total time-steps

    #################################################

    average_reward = np.mean([t['rewards'] for t in trajectories])

    #### normalization of the states in data ####################
    unscaled = np.concatenate([t['unscaled_obs'][:-burn] for t in trajectories])
    for t in trajectories:
        t['observes'] = (t['unscaled_obs'] - offset[:-1]) * scale[:-1]

    ##################################################################

    scaler.update_initial(np.hstack((unscaled, np.zeros(len(unscaled))[np.newaxis].T)))

    ########## results report ##########################
    print('Average cost: ', -average_reward)

    logger.log({'_AverageReward': -average_reward,
                'Steps': total_steps
                })
    ####################################################
    return trajectories



def run_weights(network, weights_set, policy, scaler, time_steps):
    initial_state_0 = np.zeros(policy.nn.obs_dim + 1)

    episodes = len(weights_set)

    remote_network = ray.remote(Policy)

    simulators = [remote_network.remote(policy.nn.obs_dim, policy.nn.act_dim, policy.nn.hid1_mult, policy.kl_targ,
                                        policy.epochs, policy.batch_size, policy.lr, policy.clipping_range)
                  for _ in range(episodes)]

    res = []

    ray.get([s.set_weights.remote(weights_set[i]) for i, s in enumerate(simulators)])

    res.extend(ray.get([simulators[i].policy_performance.remote(network, scaler, time_steps, initial_state_0, i)
                        for i in range(episodes)]))

    print('simulation is done')

    average_cost_set = np.zeros(episodes)
    ci_set = np.zeros(episodes)

    for i in range(episodes):
        average_cost_set[res[i][1]] = res[i][0]
        ci_set[res[i][1]] = res[i][2]

    print('Average cost: ', average_cost_set)
    print('CI: ', ci_set)
    return average_cost_set, ci_set