import numpy as np
import tensorflow as tf
import ray.experimental
import datetime
import sklearn

class PolicyNN(tf.keras.Model):
    def __init__(self, obs_dim, act_dim, kl_targ, hid1_mult):
        super().__init__()

        self.beta = 3  # dynamically adjusted D_KL loss multiplier
        self.kl_targ = kl_targ
        self.hid1_mult = hid1_mult


        self.lr_multiplier = 1.0  # dynamically adjust lr when D_KL out of control
        self.obs_dim = obs_dim
        self.act_dim = act_dim


        hid1_size = self.obs_dim * self.hid1_mult  # 10 empirically determined
        hid3_size = len(self.act_dim) * 10  # 10 empirically determined
        hid2_size = int(np.sqrt(hid1_size * hid3_size))

        self.hid1 = tf.keras.layers.Dense(hid1_size, activation='relu')
        self.hid2 = tf.keras.layers.Dense(hid2_size, activation='relu')
        self.hid3 = tf.keras.layers.Dense(hid3_size, activation='relu')
        self.a = tf.keras.layers.Dense(2, activation='softmax')

    def call(self, input_data, step_type=(), network_state=()):
        x1 = self.hid1(input_data)
        x2 = self.hid2(x1)
        x3 = self.hid3(x2)
        a = self.a(x3)
        return a


class Policy(object):
    def __init__(self, obs_dim, act_dim, kl_targ, hid1_mult,  clipping_range=0.2):
        self.nn = PolicyNN(obs_dim, act_dim, kl_targ, hid1_mult)
        self.lr = 5. * 10 ** (-4)
        self.a_opt = tf.keras.optimizers.Adam(learning_rate=self.lr)
        self.clip_pram = clipping_range
        self.epochs = 3

    def loss(self, probs, actions, adv, old_probs):
        adv = tf.cast(adv, 'float32')
        entropy = tf.reduce_mean(tf.math.negative(tf.math.multiply(probs, tf.math.log(probs))))
        # print(probability)
        # print(entropy)
        sur1 = []
        sur2 = []
        act_probs = tf.reduce_sum(probs * tf.one_hot(indices=actions.astype('int32').flatten(), depth=probs.shape[1]), axis=1)
        act_probs_old = tf.reduce_sum(old_probs * tf.one_hot(indices=actions.astype('int32').flatten(), depth=probs.shape[1]), axis=1)


        for pb, t, op in zip(act_probs, adv, act_probs_old):
            t = tf.constant(t)
            op = tf.constant(op)
            # print(f"t{t}")
            # ratio = tf.math.exp(tf.math.log(pb + 1e-10) - tf.math.log(op + 1e-10))
            ratio = tf.math.divide(pb, op)
            # print(f"ratio{ratio}")
            s1 = tf.math.multiply(ratio, t)
            # print(f"s1{s1}")
            s2 = tf.math.multiply(tf.clip_by_value(ratio, 1.0 - self.clip_pram, 1.0 + self.clip_pram), t)
            # print(f"s2{s2}")
            sur1.append(s1)
            sur2.append(s2)

        sr1 = tf.stack(sur1)
        sr2 = tf.stack(sur2)

        # closs = tf.reduce_mean(tf.math.square(td))
        loss = tf.math.negative(tf.reduce_mean(tf.math.minimum(sr1, sr2)) + 0.001 * entropy)
        # print(loss)
        return loss

    def update(self, states, actions, adv, logger):
        old_probs = self.nn(np.array(states))
        for _ in range(self.epochs):
            adv = tf.reshape(adv, (len(adv),))
            with tf.GradientTape() as tape:
                p = self.nn(states, training=True)
                a_loss = self.loss(p, actions, adv, old_probs)

            grads = tape.gradient(a_loss, self.nn.trainable_variables)
            self.a_opt.apply_gradients(zip(grads, self.nn.trainable_variables))

        logger.log({'PolicyLoss': a_loss,
                  #'Clipping': clipping_range,
                  #'Max ratio': max(ratios),
                  #'Min ratio': min(ratios),
                  #'Mean ratio': np.mean(ratios),
                  #'PolicyEntropy': entropy,
                  #'KL': kl,
                  #'Beta': self.beta,
                  #'_lr_multiplier': self.lr_multiplier}
                    })




    def sample(self, obs):
        """
        :param obs: state
        :return: if stochastic=True returns pi(a|x), else returns distribution with prob=1 on argmax[ pi(a|x) ]
        """
        if obs.ndim == 1:
            obs = tf.expand_dims(obs, axis=0)

        return self.nn(obs)





    def run_episode(self, network, scaler, time_steps,  skipping_steps, initial_state, rpp = False):
        """
        One episode simulation
        :param network: queuing network
        :param scaler: normalization values
        :param time_steps: max number of time steps
        :param skipping_steps: number of steps for which control is fixed
        :param initial_state: initial state for the episode
        :return: collected data
        """


        policy_buffer = {} # save action disctribution of visited states

        total_steps = 0 # count steps
        #action_optimal_sum = 0 # count actions that coinside with the optimal policy
        #total_zero_steps = 0 # count states for which all actions are optimal

        observes = np.zeros((time_steps, network.buffers_num))
        actions = np.zeros((time_steps, network.stations_num), 'int8')
        actions_glob = np.zeros((time_steps,  ), 'int8')
        rewards = np.zeros((time_steps, 1))
        unscaled_obs = np.zeros((time_steps, network.buffers_num), 'int32')
        unscaled_last = np.zeros((time_steps, network.buffers_num), 'int32')

        #array_actions=np.zeros((time_steps, 2))



        scale, offset = scaler.get()

        ##### modify initial state according to the method of intial states generation######
        if scaler.initial_states_procedure =='previous_iteration':
            if sum(initial_state[:-1]) > 300 :
                initial_state = np.zeros(network.buffers_num+1, 'int8')
            state = np.asarray(initial_state[:-1],'int32')
        else:
            state = np.asarray(initial_state, 'int32')

        ###############################################################

        t = 0
        while t < time_steps: # run until visit to the empty state (regenerative state)
            unscaled_obs[t] = state
            state_input = (state - offset[:-1]) * scale[:-1]  # center and scale observations

            ###### compute action distribution according to Policy Neural Network for state###


            if tuple(state) not in policy_buffer:
                act_distr = self.sample(state_input)
                policy_buffer[tuple(state)] = act_distr.numpy()
            distr = policy_buffer[tuple(state)][0] # distribution for each station

            # array_actions[0][t] = distr
            # for ar_i in range(1, network.stations_num):
            #     distr = [a * b for a in distr for b in policy_buffer[tuple(state)][ar_i][0]]
            #     array_actions[ar_i][t] = policy_buffer[tuple(state)][ar_i][0]
            distr = distr / np.sum(distr)
            ############################################


            act_ind = np.random.choice(len(distr), 1, p=distr) # sample action according to distribution 'distr'
            action_full = network.dict_absolute_to_binary_action[act_ind[0]]
            action_for_server = network.dict_absolute_to_per_server_action[act_ind[0]]

            ######### check optimality of the sampled action ################
            # if len(state)==3 and state[0]<140 and state[1]<140 and state[2]<140:
            #     if state[0]==0 or state[2]==0:
            #         total_zero_steps += 1
            #
            #     action_optimal = network.comparison_policy[tuple(state)]
            #     if all(action_full == action_optimal) or state[0]==0 or state[2]==0:
            #         action_optimal_sum += 1
            #
            # else:
            #     action_optimal = network.comparison_policy[tuple(state>0)]
            #     if all(action_full == action_optimal):
            #         action_optimal_sum += 1
            #######################



            rewards[t] =  -3*state[0]- state[1]

            unscaled_last[t] = state
            state = network.next_state_N1(state, action_full)
            actions[t] = action_for_server
            observes[t] = state_input
            actions_glob[t] = act_ind[0]

            # for i in range(skipping_steps-1):
            #      rewards[t] +=  -3*state[0]- state[1]
            #      unscaled_last[t] = state
            #      if len(state) == 3 and state[0] < 140 and state[1] < 140 and state[2] < 140:
            #          if state[0] == 0 or state[2] == 0:
            #              total_zero_steps += 1
            #
            #          action_optimal = network.comparison_policy[tuple(state)]
            #          if all(action_full == action_optimal) or state[0] == 0 or state[2] == 0:
            #              action_optimal_sum += 1
            #
            #      else:
            #          action_optimal = network.comparison_policy[tuple(state > 0)]
            #          if all(action_full == action_optimal):
            #              action_optimal_sum += 1
            #
            #
            #      state = network.next_state_N1(state, action_full) # move to the next state
            t += 1

        total_steps += len(actions)
        # record simulation

        trajectory = {#'observes': observes,
                      'actions': actions,
                      'actions_glob': actions_glob,
                      'rewards': rewards / skipping_steps,
                      'unscaled_obs': unscaled_obs,
                      'unscaled_last': unscaled_last
                  }

        print('Network:', network.network_name + '.', 'time of an episode:',
               'Average cost:',
              -np.mean(trajectory['rewards']))

        return trajectory, total_steps#, action_optimal_sum, total_zero_steps, array_actions


    def policy_performance(self, network, scaler, time_steps, initial_state, id, batch_num = 50, stochastic=True):


        average_performance_batch = np.zeros(batch_num)
        policy_buffer = {}
        batch_size = time_steps//batch_num

        time_steps = batch_size * batch_num



        scale, offset = scaler.get()

        if scaler.initial_states_procedure =='previous_iteration':
            if sum(initial_state[:-1]) > 300 :
                initial_state = np.zeros(network.buffers_num+1, 'int8')


            state = np.asarray(initial_state[:-1],'int32')
        else:
            state = np.asarray(initial_state, 'int32')
        print(state)


        batch = -1
        k = 0
        for t in range(time_steps):
            if t % batch_size == 0:
                batch += 1
                print(int(batch / batch_num * 100), '% is done')
                k = -1
            k += 1

            state_input = (state - offset[:-1]) * scale[:-1]  # center and scale observations


            if tuple(state) not in policy_buffer:

                act_distr = self.sample([state_input], stochastic)
                policy_buffer[tuple(state)] = act_distr
            distr = policy_buffer[tuple(state)][0][0]  # distribution for each station


            for ar_i in range(1, network.stations_num):
                distr = [a * b for a in distr for b in policy_buffer[tuple(state)][ar_i][0]]

            distr = distr / sum(distr)
            act_ind = np.random.choice(len(distr), 1, p=distr)

            action_full = network.dict_absolute_to_binary_action[act_ind[0]]

            #average_performance = 1/(t+1) * np.sum(state) + t / (t+1) * average_performance
            average_performance_batch[batch] = 1/(k+1) * ( 3*state[0]+ state[1]) + k / (k+1) * average_performance_batch[batch]

            state = network.next_state_N1(state, action_full)

            if np.sum(state)>5000:
                average_performance_batch = 0
                break

        average_performance = np.mean(average_performance_batch)
        ci = np.std(average_performance_batch)*1.96 / np.sqrt(batch_num)


        #optimal_ratio = action_optimal_sum / total_steps

        print(id, ' average_' + str(average_performance)+'+-' +str(ci))
        return average_performance, id, ci


    # def _loss_initial_op(self):
    #     optimizer = tf.train.AdamOptimizer(self.lr_ph)
    #     self.train_init = optimizer.minimize(self.kl)

    def initilize_rpp(self, observes, action_distr, batch_size=256):
        """
        Policy Neural Network update
        :param observes: states
        :param actions: actions
        :param advantages: estimation of antantage function at observed states
        :param logger: statistics accumulator
        """

        num_batches = max(observes.shape[0] // batch_size, 1)
        batch_size = observes.shape[0] // num_batches

        for e in range(20):
            x_train, *y_train = sklearn.utils.shuffle(observes, *action_distr)
            for j in range(num_batches):
                start = j * batch_size
                end = (j + 1) * batch_size
                feed_dict = {self.obs_ph: x_train[start:end, :],
                             self.beta_ph: self.beta,
                             self.lr_ph: self.lr * self.lr_multiplier,
                             self.act_ph: 0 * observes[:, 0:2],
                             self.advantages_ph: 0 * observes[:, 0]}
                for i in range(len(self.act_dim)):
                    feed_dict[self.old_act_prob_ph[i]] = y_train[i][start:end, :]

                self.sess.run(self.train_init, feed_dict)



    def close_sess(self):
        """ Close TensorFlow session """
        self.sess.close()

    def get_obs_dim(self):
        return self.obs_dim

    def get_hid1_mult(self):
        return self.hid1_mult

    def get_act_dim(self):
        return self.act_dim

    def get_weights(self):
        return self.variables.get_weights()

    def set_weights(self, weights):
        # Set the weights in the network.
        self.variables.set_weights(weights)

    def get_kl_targ(self):
        return self. kl_targ