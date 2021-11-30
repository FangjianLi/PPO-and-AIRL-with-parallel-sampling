import numpy as np
from others.sampler_v2 import batch_sampler
from others.utils import StructEnv
from others.utils import StructEnv_AIRL
from network_models.policy_net_continuous_discrete import Policy_net
from network_models.AIRL_net_discriminator_blend import Discriminator
import ray
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import gym


ray.init()


@ray.remote
def test_function_gym(args_envs, network_values, discrete_env_check, EPISODE_LENGTH, i, units_p_i, units_v_i):
    tf.autograph.set_verbosity(
        0, alsologtostdout=False
    )
    tf.compat.v1.logging.set_verbosity(
        tf.compat.v1.logging.ERROR
    )
    tf.reset_default_graph()
    units_p = units_p_i
    units_v = units_v_i

    sampler = batch_sampler()

    env = StructEnv(gym.make(args_envs))
    env.reset()

    env.seed(0)

    Policy_a = Policy_net('Policy_a_{}'.format(i), env, units_p, units_v)
    net_param = Policy_a.get_trainable_variables()
    net_para_values = network_values

    net_operation = []
    for i in range(len(net_param)):
        net_operation.append(tf.assign(net_param[i], net_para_values[i]))

    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())
        sess.run(net_operation)

        episode_length = 0

        render = False
        sampler.sampler_reset()
        reward_episode_counter = []

        while True:

            episode_length += 1
            act, v_pred = Policy_a.act(obs=[env.obs_a])

            if discrete_env_check:
                act = np.asscalar(act)
            else:
                act = np.reshape(act, env.action_space.shape)

            v_pred = np.asscalar(v_pred)

            next_obs, reward, done, info = env.step(act)
            sampler.sampler_traj(env.obs_a.copy(), act, reward, v_pred)

            if render:
                env.render()

            if done:
                sampler.sampler_total(0)
                reward_episode_counter.append(env.get_episode_reward())
                env.reset()
            else:
                env.obs_a = next_obs.copy()

            if episode_length >= EPISODE_LENGTH:
                last_value = np.asscalar(Policy_a.get_value([next_obs]))
                sampler.sampler_total(last_value)
                env.reset()
                break

    return sampler.sampler_get_parallel(), np.mean(reward_episode_counter)


@ray.remote
def AIRL_test_function_gym(args_envs, network_policy_values, network_discrim_values, discrete_env_check, EPISODE_LENGTH,
                           i, units_p_i, units_v_i, lr_d, num_batches):
    tf.autograph.set_verbosity(
        0, alsologtostdout=False
    )
    tf.compat.v1.logging.set_verbosity(
        tf.compat.v1.logging.ERROR
    )
    tf.reset_default_graph()
    units_p = units_p_i
    units_v = units_v_i

    sampler = batch_sampler()

    env = StructEnv_AIRL(gym.make(args_envs))
    env.reset()

    env.seed(0)

    Policy_a = Policy_net('Policy_a_{}'.format(i), env, units_p, units_v)
    Discrim_a = Discriminator('Discriminator_a_{}'.format(i), env, lr_d, num_batches)

    network_policy_param = Policy_a.get_trainable_variables()
    network_discrim_param = Discrim_a.get_trainable_variables()

    network_policy_operation = []
    for i in range(len(network_policy_param)):
        network_policy_operation.append(tf.assign(network_policy_param[i], network_policy_values[i]))

    network_discrim_operation = []
    for i in range(len(network_discrim_param)):
        network_discrim_operation.append(tf.assign(network_discrim_param[i], network_discrim_values[i]))

    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())
        sess.run(network_policy_operation)
        sess.run(network_discrim_operation)

        episode_length = 0

        render = False
        sampler.sampler_reset()
        reward_episode_counter = []
        reward_episode_counter_airl = []

        while True:

            episode_length += 1
            act, v_pred = Policy_a.act(obs=[env.obs_a])

            if discrete_env_check:
                act = act.item()
            else:
                act = np.reshape(act, env.action_space.shape)

            v_pred = v_pred.item()

            next_obs, reward, done, info = env.step(act)

            agent_sa_ph = sess.run(Policy_a.act_probs, feed_dict={Policy_a.obs: [env.obs_a.copy()],
                                                                  Policy_a.acts: [act]})

            reward_a = Discrim_a.get_rewards([env.obs_a.copy()], [act], agent_sa_ph)
            reward_a = reward_a.item()
            env.step_airl(reward_a)

            sampler.sampler_traj(env.obs_a.copy(), act, reward_a, v_pred)

            if render:
                env.render()

            if done:
                sampler.sampler_total(0)
                reward_episode_counter.append(env.get_episode_reward())
                reward_episode_counter_airl.append(env.get_episode_reward_airl())
                env.reset()
            else:
                env.obs_a = next_obs.copy()

            if episode_length >= EPISODE_LENGTH:
                last_value = np.asscalar(Policy_a.get_value([next_obs]))
                sampler.sampler_total(last_value)
                env.reset()
                break

    return sampler.sampler_get_parallel(), np.mean(reward_episode_counter), np.mean(reward_episode_counter_airl)
