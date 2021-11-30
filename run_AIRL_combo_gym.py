# this is the newer one


import argparse
import gym
import numpy as np
import tensorflow as tf
from network_models.policy_net_continuous_discrete import Policy_net
from algo.ppo_combo import PPOTrain
from others.interact_with_envs import AIRL_test_function_gym
from network_models.AIRL_net_discriminator_blend import Discriminator
import ray
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


import warnings


tf.reset_default_graph()
tf.autograph.set_verbosity(
    0, alsologtostdout=False
)

tf.compat.v1.logging.set_verbosity(
    tf.compat.v1.logging.ERROR
)


def check_and_create_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--savedir', help='save directory', default='trained_models_AIRL/')
    parser.add_argument('--model_save', help='save model name', default='model.ckpt')
    parser.add_argument('--reward_savedir', help="reward save directory", default='rewards_record_AIRL/')

    # expert data
    parser.add_argument('--expert_traj_dir', help="expert data directory", default='trajectory/')

    # The environment
    parser.add_argument("--envs_1", default="BipedalWalker-v2")

    # The hyperparameter of PPO_training
    parser.add_argument('--gamma', default=0.99, type=float)
    parser.add_argument('--lambda_1', default=0.95, type=float)
    parser.add_argument('--lr_policy', default=1e-4, type=float)
    parser.add_argument('--ep_policy', default=1e-9, type=float)
    parser.add_argument('--lr_value', default=1e-4, type=float)
    parser.add_argument('--ep_value', default=1e-9, type=float)
    parser.add_argument('--clip_value', default=0.1, type=float)
    parser.add_argument('--alter_value', default=False, type=bool)

    # The hyperparameter of the policy network
    parser.add_argument('--units_p', default=[64, 64, 64], type=int)
    parser.add_argument('--units_v', default=[96, 96, 96], type=int)

    # The hyperparameter of the policy training
    parser.add_argument('--iteration', default=int(1e3), type=int)
    parser.add_argument('--batch_size', default=512, type=int)
    parser.add_argument('--num_epoch_policy', default=6, type=int)  #6, 10
    parser.add_argument('--num_epoch_value', default=10, type=int)
    parser.add_argument('--min_length', default=40000, type=int)
    parser.add_argument('--num_parallel_sampler', default=10, type=int)

    # The hyperparameter of the discriminator network
    parser.add_argument('--lr_discrim', default=1e-4, type=float)

    # The hyperparameter of the discriminator training
    parser.add_argument('--num_expert_dimension', default=40000, type=int)
    parser.add_argument('--num_epoch_discrim', default=5, type=int)
    parser.add_argument('--batch_size_discrim', default=4000, type=int)

    # The hyperparameter of restoring the model
    parser.add_argument('--model_restore', help='filename of model to recover', default='model.ckpt')
    parser.add_argument('--continue_s', default=False, type=bool)
    parser.add_argument('--log_file', help='file to record the continuation of the training', default='continue_C1.txt')

    return parser.parse_args()


def main(args):

    model_save_dir = args.savedir + args.envs_1 + '/'
    expert_traj_dir = args.expert_traj_dir + args.envs_1 + '/'
    reward_save_dir = args.reward_savedir + args.envs_1 + '/'
    check_and_create_dir(model_save_dir)
    check_and_create_dir(reward_save_dir)


    if args.continue_s:
        args = np.load(model_save_dir+"setup.npy", allow_pickle=True).item()


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
    saver = tf.train.Saver(max_to_keep=50)

    expert_observations = np.load(expert_traj_dir+'observations.npy')

    print(np.shape(expert_observations))

    expert_actions = np.load(expert_traj_dir + 'actions.npy')

    if not discrete_env_check:
        act_dim = env.action_space.shape[0]
        expert_actions = np.reshape(expert_actions, [-1, act_dim])
    else:
        expert_actions = expert_actions.astype(np.int32)

    print(np.shape(expert_actions))

    discrim_ratio = int(np.floor(args.num_expert_dimension / args.min_length))
    discrim_batch_number = args.num_expert_dimension / args.batch_size_discrim

    D = Discriminator('AIRL_discriminator', env, args.lr_discrim, discrim_batch_number)

    origin_reward_recorder = []
    AIRL_reward_recorder = []
    counter_d = 0

    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())

        if args.continue_s:
            saver.restore(sess, model_save_dir + args.model_restore)
            origin_reward_recorder = np.load(reward_save_dir + "origin_reward.npy").tolist()
            AIRL_reward_recorder = np.load(reward_save_dir + "airl_reward.npy").tolist()

            with open(model_save_dir + args.log_file, 'a+') as r_file:
                r_file.write(
                    "the continue point: {}, the lr_policy: {}, the lr_value: {}, the lr_discrim: {} \n".format(
                        len(origin_reward_recorder), args.lr_policy, args.lr_value, args.lr_discrim))

        else:
            np.save(model_save_dir+"setup.npy", args)



        for iteration in range(args.iteration):

            policy_value = sess.run(Policy.get_trainable_variables())
            discriminator_value = sess.run(D.get_trainable_variables())


            environment_sampling = []

            for i in range(args.num_parallel_sampler):
                x1 = AIRL_test_function_gym.remote(args.envs_1, policy_value, discriminator_value,
                                                       discrete_env_check,
                                                       np.ceil(args.min_length / args.num_parallel_sampler),
                                                       i, args.units_p, args.units_v, args.lr_discrim, discrim_batch_number)
                environment_sampling.append(x1)

            results = ray.get(environment_sampling)
            print(np.shape(results))

            sampling_unpack = np.concatenate([result[0] for result in results], axis=1)
            evaluation_1 = np.mean([result[1] for result in results])
            evaluation_AIRL = np.mean([result[2] for result in results])



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

            counter_d += 1

#             if counter_d >= 2 + (iteration / 500) * 50 or iteration == 0:
#                 print("D updated")
            if counter_d >= 0:

                counter_d = 0
                expert_sa_ph = sess.run(Policy.act_probs, feed_dict={Policy.obs: expert_observations,
                                                                     Policy.acts: expert_actions})
                agent_sa_ph = sess.run(Policy.act_probs, feed_dict={Policy.obs: observation_batch_total,
                                                                    Policy.acts: action_batch_total})

                discrim_batch_expert = [expert_observations, expert_actions, expert_sa_ph]
                discrim_batch_agent = [observation_batch_total, action_batch_total, agent_sa_ph]

                for epoch_discrim in range(args.num_epoch_discrim):

                    total_index_agent = np.arange(args.min_length)
                    total_index_expert = np.arange(args.min_length * discrim_ratio)

                    np.random.shuffle(total_index_agent)
                    np.random.shuffle(total_index_expert)

                    for i in range(0, args.min_length, args.batch_size_discrim):
                        sample_indices_agent = total_index_agent[i:min(i + args.batch_size_discrim, args.min_length)]
                        sample_indices_expert = total_index_expert[i * discrim_ratio:min(
                            i * discrim_ratio + args.batch_size_discrim * discrim_ratio,
                            args.min_length * discrim_ratio)]

                        sampled_batch_agent = [np.take(a=a, indices=sample_indices_agent, axis=0) for a in
                                               discrim_batch_agent]
                        sampled_batch_expert = [np.take(a=a, indices=sample_indices_expert, axis=0) for a in
                                                discrim_batch_expert]

                        D.train(expert_s=sampled_batch_expert[0],
                                expert_a=sampled_batch_expert[1],
                                agent_s=sampled_batch_agent[0],
                                agent_a=sampled_batch_agent[1],
                                expert_sa_p=sampled_batch_expert[2],
                                agent_sa_p=sampled_batch_agent[2]
                                )

            print("at {}, the average episode reward is: {}".format(iteration, evaluation_1))
            print("at {}, the average episode AIRL reward is: {}".format(iteration, evaluation_AIRL))
            origin_reward_recorder.append(evaluation_1)
            AIRL_reward_recorder.append(evaluation_AIRL)

            if iteration % 5 == 0 and iteration > 0:
                np.save(reward_save_dir + "origin_reward.npy", origin_reward_recorder)
                np.save(reward_save_dir + "airl_reward.npy", AIRL_reward_recorder)
                saver.save(sess, model_save_dir + '{}'.format(iteration) + args.model_save)


            inp_batch = [observation_batch_total, action_batch_total, gaes_batch_total, rtg_batch_total,
                         value_next_batch_total, reward_batch_total]

            PPO.assign_policy_parameters()

            # train
            if args.alter_value:
                for epoch in range(args.num_epoch_value):
                    total_index = np.arange(args.min_length)
                    np.random.shuffle(total_index)
                    for i in range(0, args.min_length, args.batch_size):
                        sample_indices = total_index[i:min(i + args.batch_size, args.min_length)]
                        sampled_inp_batch = [np.take(a=a, indices=sample_indices, axis=0) for a in inp_batch]
                        PPO.train_value_v(obs=sampled_inp_batch[0], v_preds_next=sampled_inp_batch[4],
                                          rewards=sampled_inp_batch[5])
            else:
                for epoch in range(args.num_epoch_value):
                    total_index = np.arange(args.min_length)
                    np.random.shuffle(total_index)
                    for i in range(0, args.min_length, args.batch_size):
                        sample_indices = total_index[i:min(i + args.batch_size, args.min_length)]

                        sampled_inp_batch = [np.take(a=a, indices=sample_indices, axis=0) for a in inp_batch]

                        PPO.train_value(obs=sampled_inp_batch[0], rtg=sampled_inp_batch[3])

            for epoch in range(args.num_epoch_policy):
                total_index = np.arange(args.min_length)
                np.random.shuffle(total_index)
                for i in range(0, args.min_length, args.batch_size):
                    sample_indices = total_index[i:min(i + args.batch_size, args.min_length)]
                    sampled_inp_batch = [np.take(a=a, indices=sample_indices, axis=0) for a in inp_batch]
                    PPO.train_policy(obs=sampled_inp_batch[0], actions=sampled_inp_batch[1], gaes=sampled_inp_batch[2])


if __name__ == '__main__':
    print("17")
    args = argparser()
    warnings.filterwarnings("ignore")

    main(args)
