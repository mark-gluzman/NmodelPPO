import numpy as np
import tensorflow as tf

from sklearn.utils import shuffle

class PolicyNN(tf.keras.Model):
    def __init__(self, obs_dim, act_dim, hid1_mult):
        super().__init__()

        self.hid1_mult = hid1_mult

        self.obs_dim = obs_dim
        self.act_dim = act_dim

        hid1_size = self.obs_dim * self.hid1_mult
        hid3_size = len(self.act_dim) * 10  # 10 empirically determined
        hid2_size = int(np.sqrt(hid1_size * hid3_size))

        self.hid1 = tf.keras.layers.Dense(hid1_size, activation='relu')
        self.hid2 = tf.keras.layers.Dense(hid2_size, activation='relu')
        self.hid3 = tf.keras.layers.Dense(hid3_size, activation='relu')
        self.a = tf.keras.layers.Dense(2, activation='softmax')

        # known issue, requires passing a sample input data, see
        # https://stackoverflow.com/questions/60106829/cannot-build-custom-keras-model-with-custom-loss/60986815#60986815
        self.call(np.array([[0, 0]]))

    def call(self, input_data, step_type=(), network_state=()):
        x1 = self.hid1(input_data)
        x2 = self.hid2(x1)
        x3 = self.hid3(x2)
        a = self.a(x3)
        return a


class Policy(object):
    def __init__(self, obs_dim, act_dim, hid1_mult, kl_targ, ep_p, bs_p, lr_p, clipping_range):
        self.nn = PolicyNN(obs_dim, act_dim, hid1_mult)

        self.lr = lr_p
        self.clipping_range = clipping_range
        self.epochs = ep_p
        self.batch_size = bs_p
        self.kl_targ = kl_targ

    def sur_loss(self, probs, actions, adv, old_probs):
        """
        :param probs: current action probabilities
        :param actions: geenrated actions
        :param adv: advantage function estimates
        :param old_probs: action probabilities before training
        :return: surrogate loss function
        """
        adv = tf.cast(adv, 'float32')
        self.entropy = tf.reduce_mean(tf.math.negative(tf.math.multiply(probs, tf.math.log(probs))))

        act_probs = tf.reduce_sum(probs * tf.one_hot(indices=actions.astype('int32').flatten(), depth=probs.shape[1]), axis=1)
        act_probs_old = tf.reduce_sum(old_probs * tf.one_hot(indices=actions.astype('int32').flatten(), depth=probs.shape[1]), axis=1)

        ratios = tf.math.exp(tf.math.log(act_probs) - tf.math.log(act_probs_old))
        clipped_ratios = tf.clip_by_value(ratios, clip_value_min=1 - self.clipping_range, clip_value_max=1 + self.clipping_range)
        loss_clip = tf.minimum(tf.multiply(adv, ratios), tf.multiply(adv, clipped_ratios))

        loss = tf.math.negative(tf.reduce_mean(loss_clip))#- self.entropy*0.0001

        return loss

    def update(self, states, actions, adv, logger):
        """
        policy NN parameters update
        :param states: generated states
        :param actions: generated actions
        :param adv: estimates of the advantage fucntion
        :param logger: stats tracker
        """
        states = np.array(states, dtype='float32')
        old_probs = self.nn(states).numpy()
        bat_per_epoch = int(len(states) / self.batch_size)
        policy_loss = 0
        a_opt = tf.keras.optimizers.Adam(learning_rate=self.lr)
        for epoch in range(self.epochs):
            states, actions, adv, old_probs = shuffle(states, actions, adv, old_probs)
            policy_loss = 0
            for i in range(bat_per_epoch):
                n = i * self.batch_size
                batch_loss, kl = self.batch_step(states[n:n + self.batch_size], actions[n:n + self.batch_size],
                                 adv[n:n + self.batch_size], old_probs[n:n + self.batch_size], optimizer=a_opt)
                policy_loss += batch_loss.numpy()

                if kl.numpy() > self.kl_targ * 3:  # early stopping if D_KL diverges badly
                    print('early stopping: D_KL diverges badly')
                    break


            print('Policy NN learning epoch #', epoch, ' Loss = ', policy_loss)
        probs = self.nn(states)
        entropy = tf.reduce_mean(tf.math.negative(tf.math.multiply(probs, tf.math.log(probs))))
        kl = tf.reduce_mean(tf.math.multiply(probs, tf.math.log(tf.math.divide(probs, old_probs))))
        act_probs = tf.reduce_sum(
            probs * tf.one_hot(indices=actions.astype('int32').flatten(), depth=probs.shape[1]),axis=1)
        act_probs_old = tf.reduce_sum(
            old_probs * tf.one_hot(indices=actions.astype('int32').flatten(), depth=probs.shape[1]), axis=1)

        ratios = tf.math.exp(tf.math.log(act_probs) - tf.math.log(act_probs_old))


        logger.log({'PolicyLoss': policy_loss,
                  'Clipping': self.clipping_range,
                  'Max ratio': np.max(ratios),
                  'Min ratio': np.min(ratios),
                  'Mean ratio': np.mean(ratios),
                  'PolicyEntropy': entropy.numpy(),
                  'KL': kl.numpy(),
                  'lr': self.lr})

    def batch_step(self, states, actions, adv, old_probs, optimizer):
        """
        minibatch gradient computation and optimization
        :param states: generated states
        :param actions: generated actions
        :param adv: estimates of the advantage fucntion
        :param old_probs: action probabilities before training
        :param optimizer: NN optimizer
        :return:
        """

        with tf.GradientTape() as tape:
            # Make prediction
            p = self.nn(states, training=True)
            # Calculate loss
            loss = self.sur_loss(p, actions, adv, old_probs)
        # Calculate gradients
        grads = tape.gradient(loss, self.nn.trainable_variables)
        # Update model
        optimizer.apply_gradients(zip(grads, self.nn.trainable_variables))
        kl = tf.reduce_mean(tf.math.multiply(p, tf.math.log(tf.math.divide(p, old_probs))))
        return loss, kl

    def set_weights(self, weights):
        self.nn.set_weights(weights)

    def get_weights(self):
        return self.nn.get_weights()

    def sample(self, obs, stochastic = True):
        """
        returns action probabilities for states in 'obs'
        :param obs: state
        :return: if stochastic=True returns pi(a|x), else returns distribution with prob=1 on argmax[ pi(a|x) ]
        """
        obs = np.array(obs, dtype = 'float32')
        if obs.ndim == 1:
            obs = tf.expand_dims(obs, axis=0)
        pr = self.nn(obs).numpy()

        if stochastic:
            return pr
        else:
            determ_prob = []

            inx = np.argmax(pr.numpy())
            ar = np.zeros(self.nn.act_dim)
            ar[inx] = 1
            determ_prob.extend([ar[np.newaxis]])

            return determ_prob

    def run_episode(self, network, scaler, time_steps, initial_state):
        """
        One episode simulation
        :param network: queuing network
        :param scaler: normalization values
        :param time_steps: max number of time steps
        :param initial_state: initial state for the episode
        :return: collected data
        """

        policy_buffer = {} # save action disctribution of visited states

        total_steps = 0 # count steps

        observes = np.zeros((time_steps, network.buffers_num))
        actions = np.zeros((time_steps, network.stations_num), 'int8')
        actions_glob = np.zeros((time_steps,  ), 'int8')
        rewards = np.zeros((time_steps, 1))
        unscaled_obs = np.zeros((time_steps, network.buffers_num), 'int32')
        unscaled_last = np.zeros((time_steps, network.buffers_num), 'int32')

        scale, offset = scaler.get()

        ##### modify initial state according to the method of intial states generation######
        if sum(initial_state[:-1]) > 300 :
            initial_state = np.zeros(network.buffers_num+1, 'int8')
        state = np.asarray(initial_state[:-1],'int32')
        ###############################################################

        t = 0
        while t < time_steps: # run until visit to the empty state (regenerative state)
            unscaled_obs[t] = state
            state_input = (state - offset[:-1]) * scale[:-1]  # center and scale observations

            ###### compute action distribution according to Policy Neural Network for state###
            if tuple(state) not in policy_buffer:
                act_distr = self.sample(state_input)
                policy_buffer[tuple(state)] = act_distr
            distr = policy_buffer[tuple(state)][0] # distribution for each station
            distr = distr / np.sum(distr)
            ############################################


            act_ind = np.random.choice(len(distr), 1, p=distr) # sample action according to distribution 'distr'
            action_full = network.dict_absolute_to_binary_action[act_ind[0]]
            action_for_server = network.dict_absolute_to_per_server_action[act_ind[0]]

            rewards[t] = - np.dot(network.holding_cost, state)

            unscaled_last[t] = state
            state = network.next_state_N1(state, action_full)
            actions[t] = action_for_server
            observes[t] = state_input
            actions_glob[t] = act_ind[0]

            t += 1

        total_steps += len(actions)
        # record simulation
        no_opt_actions = check_optimality(unscaled_obs, actions)
        trajectory = {#'observes': observes,
                      'actions': actions,
                      'actions_glob': actions_glob,
                      'rewards': rewards,
                      'unscaled_obs': unscaled_obs,
                      'unscaled_last': unscaled_last,
                      'no_opt_actions': no_opt_actions
                  }



        print('Network:', network.network_name + '.',
               'Average cost:', -np.mean(trajectory['rewards']))

        return trajectory, total_steps






    def policy_performance(self, network, scaler, time_steps, initial_state, id, batch_num = 50, stochastic=True):
        """
        policy evaluation
        :param network: queueing network
        :param scaler: normalization stats
        :param time_steps: episode length
        :param initial_state: episode initial state
        :param id: evaluation process id
        :param batch_num: method computes CI splitting the episode on 'batch_num' batches
        :return: policy performance
        """
        average_performance_batch = np.zeros(batch_num)
        policy_buffer = {}
        batch_size = time_steps//batch_num

        time_steps = batch_size * batch_num

        scale, offset = scaler.get()

        if sum(initial_state[:-1]) > 300:
            initial_state = np.zeros(network.buffers_num+1, 'int8')
        state = np.asarray(initial_state[:-1],'int32')

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

            ###### compute action distribution according to Policy Neural Network for state###
            if tuple(state) not in policy_buffer:
                act_distr = self.sample(state_input)
                policy_buffer[tuple(state)] = act_distr
            distr = policy_buffer[tuple(state)][0] # distribution for each station
            distr = distr / np.sum(distr)
            ############################################

            act_ind = np.random.choice(len(distr), 1, p=distr)

            action_full = network.dict_absolute_to_binary_action[act_ind[0]]

            average_performance_batch[batch] = \
                1/(k+1) * (np.dot(network.holding_cost, state)) + k / (k+1) * average_performance_batch[batch]

            state = network.next_state_N1(state, action_full)

            if np.sum(state)>5000:
                average_performance_batch = 0
                break

        average_performance = np.mean(average_performance_batch)
        ci = np.std(average_performance_batch)*1.96 / np.sqrt(batch_num)


        print(id, ' average_' + str(average_performance)+'+-' +str(ci))
        return average_performance, id, ci


def check_optimality(unscaled_obs, actions, file_name = 'action09.npy'):
    """
    :param trajectory: generate trajectory
    :param file_name: file with array that contains optimal actions
    :return: percentage of optimal actions chosen at trajectory
    """
    optimal_actions = np.load(file_name)-1.
    no_opt_actions = 0
    total_decision_time_steps = len(unscaled_obs)
    for k in range(len(unscaled_obs)):
        state = unscaled_obs[k]
        action = actions[k][0]
        if state[0] < optimal_actions.shape[0] and state[1] < optimal_actions.shape[1] and state[0]*state[1] > 0:
            if action == optimal_actions[state[0], state[1]]:
                no_opt_actions += 1
        else:
            total_decision_time_steps -= 1

    return no_opt_actions*100 // total_decision_time_steps
