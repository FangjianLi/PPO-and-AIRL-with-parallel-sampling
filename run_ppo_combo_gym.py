import argparse
import gym
import numpy as np
import tensorflow as tf
from network_models.policy_net_continuous_discrete import Policy_net
from algo.ppo_combo import PPOTrain
from others.interact_with_envs import test_function_gym
import os
import ray
import warnings
import time

tf.reset_default_graph()
tf.autograph.set_verbosity(
    0, alsologtostdout=False
)


def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--savedir', help='save directory', default='trained_models/')
    parser.add_argument('--model_save', help='save model name', default='model.ckpt')
    parser.add_argument('--reward_savedir', help="reward save directory", default='rewards_record/')
    # reward.npy

    # The environment
    parser.add_argument("--envs_1", default="BipedalWalker-v2")

    # The hyperparameter of PPO_training
    parser.add_argument('--gamma', default=0.99, type=float)
    parser.add_argument('--lambda_1', default=0.95, type=float)
    parser.add_argument('--lr_policy', default=5e-5, type=float)  # 1e-4
    parser.add_argument('--ep_policy', default=1e-9, type=float)
    parser.add_argument('--lr_value', default=5e-5, type=float)  # 1e-4
    parser.add_argument('--ep_value', default=1e-9, type=float)
    parser.add_argument('--clip_value', default=0.1, type=float)  # 0.2
    parser.add_argument('--alter_value', default=False, type=bool)

    # The hyperparameter of the policy network
    parser.add_argument('--units_p', default=[64, 64, 64], type=int)
    parser.add_argument('--units_v', default=[96, 96, 96], type=int)

    # The hyperparameter of the training
    parser.add_argument('--iteration', default=500, type=int)
    parser.add_argument('--batch_size', default=512, type=int)
    parser.add_argument('--num_epoch_policy', default=6, type=int)
    parser.add_argument('--num_epoch_value', default=10, type=int)
    parser.add_argument('--sample_size', default=20000, type=int)  # 20000
    parser.add_argument('--num_parallel_sampler', default=10, type=int)

    # The hyperparameter of restoring the model
    parser.add_argument('--model_restore', help='filename of model to recover', default='model.ckpt')
    parser.add_argument('--continue_s', default=False, type=bool)
    parser.add_argument('--log_file', help='file to record the continuation of the training', default='continue.txt')

    return parser.parse_args()


def check_and_create_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def main(args):
    model_save_dir = args.savedir + args.envs_1 + '/'
    reward_save_dir = args.reward_savedir + args.envs_1 + '/'
    check_and_create_dir(model_save_dir)
    check_and_create_dir(reward_save_dir)

    if args.continue_s:
        args = np.load(model_save_dir + "setup.npy", allow_pickle=True).item()

    env = gym.make(args.envs_1)
    print(env.observation_space.shape)

    discrete_env_check = isinstance(env.action_space, gym.spaces.discrete.Discrete)
    env.seed(0)

    if not discrete_env_check:
        print(env.action_space.low)
        print(env.action_space.high)

    Policy = Policy_net('policy', env, args.units_p, args.units_v)
    Old_Policy = Policy_net('old_policy', env, args.units_p, args.units_v)
    PPO = PPOTrain(Policy, Old_Policy, gamma=args.gamma, lambda_1=args.lambda_1, lr_policy=args.lr_policy,
                   lr_value=args.lr_value, clip_value=args.clip_value)
    saver = tf.train.Saver()

    reward_recorder = []

    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())

        if args.continue_s:
            saver.restore(sess, model_save_dir + args.model_restore)
            reward_recorder = np.load(reward_save_dir + "reward.npy").tolist()
            with open(model_save_dir + args.log_file, 'a+') as r_file:
                r_file.write(
                    "the continue point: {}, the lr_policy: {}, the lr_value: {} \n".format(len(reward_recorder),
                                                                                            args.lr_policy,
                                                                                            args.lr_value))
        else:
            np.save(model_save_dir + "setup.npy", args)

        for iteration in range(args.iteration):

            policy_value = sess.run(Policy.get_trainable_variables())

            environment_sampling = []

            start_timer = time.time()
            for i in range(args.num_parallel_sampler):
                x1 = test_function_gym.remote(args.envs_1, policy_value, discrete_env_check,
                                              np.ceil(args.sample_size / args.num_parallel_sampler), i,
                                              args.units_p, args.units_v)
                environment_sampling.append(x1)

            results = ray.get(environment_sampling)

            sampling_unpack = np.concatenate([result[0] for result in results], axis=1)
            evaluation_1 = np.mean([result[1] for result in results])

            observation_batch_total, action_batch_total, rtg_batch_total, gaes_batch_total, \
            value_next_batch_total, reward_batch_total = sampling_unpack

            observation_batch_total = np.array([observation_batch for observation_batch in observation_batch_total])

            action_batch_total = np.array([action_batch for action_batch in action_batch_total])

            rtg_batch_total = np.array([rtg_batch for rtg_batch in rtg_batch_total])

            gaes_batch_total = np.array([gaes_batch for gaes_batch in gaes_batch_total])
            value_next_batch_total = np.array([value_next_batch for value_next_batch in value_next_batch_total])
            reward_batch_total = np.array([reward_batch for reward_batch in reward_batch_total])

            gaes_batch_total = (gaes_batch_total - np.mean(gaes_batch_total)) / (
                    np.std(gaes_batch_total) + 1e-10)

            end_timer = time.time()

            print("at {}, the average episode reward is: {}, takes {}s".format(iteration, evaluation_1,
                                                                               end_timer - start_timer))
            reward_recorder.append(evaluation_1)
            if iteration % 5 == 0 and iteration > 0:
                np.save(reward_save_dir + "reward.npy", reward_recorder)
                saver.save(sess, model_save_dir + args.model_save)

            inp_batch = [observation_batch_total, action_batch_total, gaes_batch_total, rtg_batch_total,
                         value_next_batch_total, reward_batch_total]

            PPO.assign_policy_parameters()

            # train
            for epoch in range(args.num_epoch_policy):
                total_index = np.arange(args.sample_size)
                np.random.shuffle(total_index)
                for i in range(0, args.sample_size, args.batch_size):
                    sample_indices = total_index[i:min(i + args.batch_size, args.sample_size)]
                    sampled_inp_batch = [np.take(a=a, indices=sample_indices, axis=0) for a in inp_batch]
                    PPO.train_policy(obs=sampled_inp_batch[0], actions=sampled_inp_batch[1], gaes=sampled_inp_batch[2])

            if args.alter_value:
                for epoch in range(args.num_epoch_value):
                    total_index = np.arange(args.sample_size)
                    np.random.shuffle(total_index)
                    for i in range(0, args.sample_size, args.batch_size):
                        sample_indices = total_index[i:min(i + args.batch_size, args.sample_size)]
                        sampled_inp_batch = [np.take(a=a, indices=sample_indices, axis=0) for a in inp_batch]
                        PPO.train_value_v(obs=sampled_inp_batch[0], v_preds_next=sampled_inp_batch[4],
                                          rewards=sampled_inp_batch[5])
            else:
                for epoch in range(args.num_epoch_value):
                    total_index = np.arange(args.sample_size)
                    np.random.shuffle(total_index)
                    for i in range(0, args.sample_size, args.batch_size):
                        sample_indices = total_index[i:min(i + args.batch_size, args.sample_size)]

                        sampled_inp_batch = [np.take(a=a, indices=sample_indices, axis=0) for a in inp_batch]

                        PPO.train_value(obs=sampled_inp_batch[0], rtg=sampled_inp_batch[3])


if __name__ == '__main__':
    print("PPO training starts")
    args = argparser()
    warnings.filterwarnings("ignore")
    tf.reset_default_graph()
    main(args)
