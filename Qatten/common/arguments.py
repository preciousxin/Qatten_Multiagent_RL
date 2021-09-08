import argparse

"""
Here are the param for the training

"""


def get_common_args():
    parser = argparse.ArgumentParser()
    # the environment setting
    parser.add_argument('--difficulty', type=str, default='7', help='the difficulty of the game')
    parser.add_argument('--game_version', type=str, default='latest', help='the version of the game')
    parser.add_argument('--map', type=str, default='3m', help='the map of the game')    # matrixgame import Matrix_game2Env
    parser.add_argument('--seed', type=int, default=123, help='random seed')
    parser.add_argument('--step_mul', type=int, default=8, help='how many steps to make an action')
    parser.add_argument('--replay_dir', type=str, default='', help='absolute path to save the replay')
    # The alternative algorithms are vdn, coma, central_v, qmix, qtran_base,
    # qtran_alt, reinforce, coma+commnet, central_v+commnet, reinforce+commnet，
    # coma+g2anet, central_v+g2anet, reinforce+g2anet, maven
    parser.add_argument('--alg', type=str, default='qatten', help='the algorithm to train the agent')
    parser.add_argument('--last_action', type=bool, default=True, help='whether to use the last action to choose action')
    parser.add_argument('--reuse_network', type=bool, default=True, help='whether to use one network for all agents')
    parser.add_argument('--gamma', type=float, default=0.99, help='discount factor')
    parser.add_argument('--optimizer', type=str, default="RMS", help='optimizer')
    parser.add_argument('--evaluate_epoch', type=int, default=20, help='number of the epoch to evaluate the agent')   #default = 20
    parser.add_argument('--model_dir', type=str, default='./model', help='model directory of the policy')
    parser.add_argument('--result_dir', type=str, default='./result', help='result directory of the policy')
    parser.add_argument('--load_model', type=bool, default=False, help='whether to load the pretrained model')
    parser.add_argument('--learn', type=bool, default=True, help='whether to train the model')
    parser.add_argument('--cuda', type=bool, default=True, help='whether to use the GPU')
    args = parser.parse_args()
    return args

# arguments of qatten
def get_qatten_args(args):
    # network
    args.rnn_hidden_dim = 64
    # Qatten multi-head number
    args.num_head = 4
    # embedding dim for query and key
    args.query_hidden_dim1 = 64
    args.query_hidden_dim2 = 32
    args.key_hidden_dim = 32
    args.weight_hidden_dim1 = 64
    args.weight_hidden_dim2 = args.num_head
    args.c_hidden_dim1 = 32
    args.c_hidden_dim2 = 1

    # if head weight
    args.is_head_weight = False
    args.lr = 5e-4

    # epsilon greedy
    args.epsilon = 1
    args.min_epsilon = 0.05
    anneal_steps = 500000           #defalut 500000
    args.anneal_epsilon = (args.epsilon - args.min_epsilon) / anneal_steps
    args.epsilon_anneal_scale = 'step'

    # the number of the epoch to train the agent
    args.n_epoch = 20000         #defalut 20000

    # the number of the episodes in one epoch
    args.n_episodes = 1

    # the number of the train steps in one epoch
    args.train_steps = 1

    # # how often to evaluate
    args.evaluate_cycle = 100   #default 100

    # experience replay
    args.batch_size = 32
    args.buffer_size = int(5e3)

    # how often to save the model
    args.save_cycle = 5000     # default 5000

    # how often to update the target_net
    args.target_update_cycle = 200       #default 200

    # prevent gradient explosion
    args.grad_norm_clip = 10

    return args



    


