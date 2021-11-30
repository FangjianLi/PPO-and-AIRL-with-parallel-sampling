import tensorflow as tf


class Discriminator:
    def __init__(self, env):
        """
        :param env:
        Output of this Discriminator is reward for learning agent. Not the cost.
        Because discriminator predicts  P(expert|s,a) = 1 - P(agent|s,a).
        """



        self.expert_s = tf.placeholder(dtype=tf.float32, shape=[None] + list(env.observation_space.shape))
        self.expert_a = tf.placeholder(dtype=tf.int32, shape=[None])
        self.agent_s = tf.placeholder(dtype=tf.float32, shape=[None] + list(env.observation_space.shape))
        self.agent_a = tf.placeholder(dtype=tf.int32, shape=[None])
        self.expert_sa_p=tf.placeholder(dtype=tf.float32, shape=[None])
        self.agent_sa_p=tf.placeholder(dtype=tf.float32, shape=[None])

        with tf.variable_scope('discriminator'):
            self.scope = tf.get_variable_scope().name
            expert_a_one_hot = tf.one_hot(self.expert_a, depth=env.action_space.n)
            # add noise for stabilise training
            expert_a_one_hot += tf.random_normal(tf.shape(expert_a_one_hot), mean=0.2, stddev=0.1, dtype=tf.float32)/1.2
            expert_s_a = tf.concat([self.expert_s, expert_a_one_hot], axis=1)


            agent_a_one_hot = tf.one_hot(self.agent_a, depth=env.action_space.n)
            # add noise for stabilise training
            agent_a_one_hot += tf.random_normal(tf.shape(agent_a_one_hot), mean=0.2, stddev=0.1, dtype=tf.float32)/1.2
            agent_s_a = tf.concat([self.agent_s, agent_a_one_hot], axis=1)

            #


            with tf.variable_scope('network') as network_scope:
                #the probability of the expert's (s,a) is from expert
                prob_1 = self.construct_network(input=expert_s_a)

                network_scope.reuse_variables()  # share parameter, it can reuse the parameter within the parameter
                #the probability of the agent's (s,a) is from agent
                prob_2 = self.construct_network(input=agent_s_a)

            with tf.variable_scope('loss'):
                p_expert = tf.clip_by_value(prob_1, 0.01, 1)
                d_expert=p_expert/(p_expert+self.expert_sa_p)
                loss_expert = tf.reduce_mean(tf.log(d_expert))

                p_agent=tf.clip_by_value(1 - prob_2, 0.01, 1)
                d_agent=p_agent/(p_agent+self.agent_sa_p)

                loss_agent = tf.reduce_mean(tf.log(d_agent))
                loss = loss_expert + loss_agent
                loss = -loss
                # this is the exact formulation of GAIL loss function of the discriminator
                tf.summary.scalar('discriminator', loss)

            optimizer = tf.train.AdamOptimizer(learning_rate=1e-4)
            self.train_op = optimizer.minimize(loss)

            self.rewards = tf.log(tf.clip_by_value(prob_2, 1e-10, 1))  # log(P(expert|s,a)) larger is better for agent

    def construct_network(self, input):
        layer_1 = tf.layers.dense(inputs=input, units=40, activation=tf.nn.leaky_relu, name='layer1')
        layer_2 = tf.layers.dense(inputs=layer_1, units=40, activation=tf.nn.leaky_relu, name='layer2')
        prob = tf.layers.dense(inputs=layer_2, units=1, activation=tf.sigmoid, name='prob')
        return prob  #the sigmoid is used to generate probability

    def train(self, expert_s, expert_a, agent_s, agent_a, expert_sa_p,agent_sa_p):
        return tf.get_default_session().run(self.train_op, feed_dict={self.expert_s: expert_s,
                                                                      self.expert_a: expert_a,
                                                                      self.agent_s: agent_s,
                                                                      self.agent_a: agent_a,
                                                                      self.expert_sa_p: expert_sa_p,
                                                                      self.agent_sa_p: agent_sa_p})

    def get_rewards(self, agent_s, agent_a):
        return tf.get_default_session().run(self.rewards, feed_dict={self.agent_s: agent_s,
                                                                     self.agent_a: agent_a})

    def get_trainable_variables(self): #this might be used to design the regulator
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)

