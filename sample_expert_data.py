import argparse
import gym
import numpy as np
from network_models.policy_net_continuous_discrete import Policy_net
from others.interact_with_envs import test_function_gym
import tensorflow as tf
import ray
import os
import warnings
import time



def check_and_create_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--savedir', help='save directory', default='trained_models/')
    parser.add_argument('--model_index', help='save model name', default='model.ckpt')

    parser.add_argument('--traj_savedir', default='trajectory/')

    parser.add_argument("--envs_1", default="BipedalWalker-v2")
    parser.add_argument('--iteration', default=1, type=int)
    parser.add_argument('--min_length', default=80000, type=int)
    parser.add_argument('--num_parallel_sampler', default=10, type=int)

    return parser.parse_args()


def main(args):
    start_timer = time.time()

    model_save_dir = args.savedir + args.envs_1 + '/'
    traj_save_dir = args.traj_savedir + args.envs_1 + '/'
    check_and_create_dir(traj_save_dir)

    args_restore = np.load(model_save_dir + "setup.npy", allow_pickle=True).item()


    env = gym.make(args.envs_1)
    print(env.observation_space.shape)

    discrete_env_check = isinstance(env.action_space, gym.spaces.discrete.Discrete)
    env.seed(0)

    Policy = Policy_net('policy', env, args_restore.units_p, args_restore.units_v)

    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, model_save_dir + args.model_index)

        policy_value = sess.run(Policy.get_trainable_variables())

        environment_sampling = []

        for i in range(args.num_parallel_sampler):
            x1 = test_function_gym.remote(args.envs_1, policy_value, discrete_env_check,
                                          np.ceil(args.min_length / args.num_parallel_sampler), i,
                                          args_restore.units_p, args_restore.units_v)
            environment_sampling.append(x1)

        results = ray.get(environment_sampling)

        sampling_unpack = np.concatenate([result[0] for result in results], axis=1)
        evaluation_1 = np.mean([result[1] for result in results])

        observation_batch_total, action_batch_total, _, _, _, _ = sampling_unpack

        observation_batch_total = np.array([observation_batch for observation_batch in observation_batch_total])
        action_batch_total = np.array([action_batch for action_batch in action_batch_total])

    print("The average episode reward is: {}".format(evaluation_1))
    print("It takes: {}".format(time.time() - start_timer))
    print(np.shape(observation_batch_total))
    print(np.shape(action_batch_total))

    np.save(traj_save_dir + 'observations.npy', observation_batch_total)
    np.save(traj_save_dir + 'actions.npy', action_batch_total)


if __name__ == '__main__':
    args = argparser()
    main(args)
