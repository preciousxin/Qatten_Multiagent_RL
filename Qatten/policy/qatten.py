import torch
import os
from network.qatten_net import QattenNet
from network.base_net import RNN


class Qatten:
    def __init__(self, args):
        self.args = args
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        self.state_shape = args.state_shape
        self.obs_shape = args.obs_shape
        input_shape = self.obs_shape
        if args.last_action:
            input_shape += self.n_actions
        if args.reuse_network:
            input_shape += self.n_agents

        self.eval_rnn = RNN(input_shape, args)
        self.target_rnn = RNN(input_shape, args)
        self.eval_qatten_net = QattenNet(args)
        self.target_qatten_net = QattenNet(args)

        if self.args.cuda:
            self.eval_rnn.cuda()
            self.target_rnn.cuda()
            self.eval_qatten_net.cuda()
            self.target_qatten_net.cuda()

        # load model
        self.model_dir = args.model_dir + '/' + args.alg + '/' + args.map
        if self.args.load_model:
            if os.path.exists(self.model_dir + '/rnn_net_params.pkl'):
                path_rnn = self.model_dir + '/rnn_net_params.pkl'
                path_qatten = self.model_dir + '/qatten_net_params.pkl'
                map_location = 'cuda:0' if self.args.cuda else 'cpu'
                self.eval_rnn.load_state_dict(torch.load(path_rnn, map_location=map_location))
                self.eval_qatten_net.load_state_dict(torch.load(path_qatten, map_location=map_location))
                print('Successfully load the model: {} and {}'.format(path_rnn, path_qatten))
            else:
                raise Exception("No model!")

        # 让target_net和eval_net的网络参数相同
        self.target_rnn.load_state_dict(self.eval_rnn.state_dict())
        self.target_qatten_net.load_state_dict(self.eval_qatten_net.state_dict())

        #########*******  parameters ********###########
        self.eval_parameters = list(self.eval_qatten_net.parameters()) + list(self.eval_rnn.parameters())
        if args.optimizer == "RMS":
            self.optimizer = torch.optim.RMSprop(self.eval_parameters, lr=args.lr)

        # 执行过程中，要为每个agent都维护一个eval_hidden
        # 学习过程中，要为每个episode的每个agent都维护一个eval_hidden、target_hidden
        self.eval_hidden = None
        self.target_hidden = None
        print('Init alg Qatten')

    def learn(self, batch, max_episode_len, train_step, epsilon=None):
        episode_num = batch['o'].shape[0]
        self.init_hidden(episode_num)
        # convert the numpy in batch to tensor
        for key in batch.keys():
            if key == 'u':
                batch[key] = torch.tensor(batch[key], dtype=torch.long)
            else:
                batch[key] = torch.tensor(batch[key], dtype=torch.float32)

        o, o_next, s, s_next, u, r, avail_u, avail_u_next, terminated = batch['o'], batch['o_next'],\
                                                                        batch['s'], batch['s_next'], batch['u'], \
                                                                        batch['r'], batch['avail_u'],\
                                                                        batch['avail_u_next'], batch['terminated']
        mask = 1 - batch['padded'].float()

        q_evals, q_targets = self.get_q_values(batch, max_episode_len)   # episode_num, max_episode_len， n_agents，n_actions
        if self.args.cuda:
            o = o.cuda()
            o_next = o_next.cuda()
            s = s.cuda()
            s_next = s_next.cuda()
            u = u.cuda()
            r = r.cuda()
            avail_u = avail_u.cuda()
            avail_u_next = avail_u_next.cuda()
            terminated = terminated.cuda()
            mask = mask.cuda()

        # get individual q value
        q_evals = torch.gather(q_evals, dim=3, index=u)
        q_targets[avail_u_next == 0.0] = - 9999999
        q_targets = q_targets.max(dim=3)[0]
        # get total q value
        q_tot_eval = self.eval_qatten_net(q_evals, s, o)
        q_tot_target = self.target_qatten_net(q_targets, s_next, o_next)

        targets = r + self.args.gamma * q_tot_target * (1 - terminated)
        td_error = (q_tot_eval - targets.detach())
        masked_td_error = td_error * mask
        loss = (masked_td_error ** 2).sum() / mask.sum()

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.eval_parameters, self.args.grad_norm_clip)
        self.optimizer.step()

        if train_step > 0 and train_step % self.args.target_update_cycle == 0:
            self.target_rnn.load_state_dict(self.eval_rnn.state_dict())
            self.target_qatten_net.load_state_dict(self.eval_qatten_net.state_dict())


    def _get_inputs(self, batch, t_idx):
        obs, obs_next, u_onehot = batch['o'][:,t_idx], batch['o_next'][:,t_idx],\
                                  batch['u_onehot'][:]
        episode_num = obs.shape[0]
        inputs, inputs_next = [], []
        inputs.append(obs)
        inputs_next.append(obs_next)
        if self.args.last_action:
            if t_idx == 0:
                inputs.append(torch.zeros_like(u_onehot[:, t_idx]))
            else:
                inputs.append(u_onehot[:, t_idx-1])
            inputs_next.append(u_onehot[:, t_idx])
        if self.args.reuse_network:
            inputs.append(torch.eye(self.args.n_agents).unsqueeze(0).expand(episode_num, -1, -1))
            inputs_next.append(torch.eye(self.args.n_agents).unsqueeze(0).expand(episode_num, -1, -1))

        inputs = torch.cat([x.reshape(episode_num * self.args.n_agents, -1) for x in inputs], dim=1)
        inputs_next = torch.cat([x.reshape(episode_num * self.args.n_agents, -1) for x in inputs_next], dim=1)

        return inputs, inputs_next

    def get_q_values(self, batch, max_episode_len):
        episode_num = batch['o'].shape[0]
        q_evals, q_targets = [], []
        for t_idx in range(max_episode_len):
            inputs, inputs_next = self._get_inputs(batch, t_idx)
            if self.args.cuda:
                inputs = inputs.cuda()
                inputs_next = inputs_next.cuda()
                self.eval_hidden = self.eval_hidden.cuda()
                self.target_hidden = self.target_hidden.cuda()
            q_eval, self.eval_hidden = self.eval_rnn(inputs, self.eval_hidden)
            q_target, self.target_hidden = self.target_rnn(inputs_next, self.target_hidden)

            q_eval = q_eval.view(episode_num, self.n_agents, -1)
            q_target = q_target.view(episode_num, self.n_agents, -1)
            q_evals.append(q_eval)
            q_targets.append(q_target)

        q_evals = torch.stack(q_evals, dim=1)
        q_targets = torch.stack(q_targets, dim=1)
        return q_evals, q_targets


    def init_hidden(self, episode_num):
        # 为每个episode中的每个agent都初始化一个eval_hidden、target_hidden
        self.eval_hidden = torch.zeros((episode_num, self.n_agents, self.args.rnn_hidden_dim))
        self.target_hidden = torch.zeros((episode_num, self.n_agents, self.args.rnn_hidden_dim))

    def save_model(self, train_step):
        num = str(train_step // self.args.save_cycle)
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        torch.save(self.eval_qatten_net.state_dict(), self.model_dir + '/' + num + '_qatten_net_params.pkl')
        torch.save(self.eval_rnn.state_dict(), self.model_dir + '/' + num + '_rnn_net_params.pkl')




if __name__ == '__main__':
    from common.arguments import get_common_args, get_qatten_args
    from smac.env import StarCraft2Env
    args = get_common_args()
    args = get_atten_args(args)

    env = StarCraft2Env(map_name=args.map,
                        step_mul=args.step_mul,
                        difficulty=args.difficulty,
                        game_version=args.game_version)

    env_info = env.get_env_info()
    args.n_actions = env_info["n_actions"]
    args.n_agents = env_info["n_agents"]
    args.state_shape = env_info["state_shape"]
    args.obs_shape = env_info["obs_shape"]
    args.episode_limit = env_info["episode_limit"]

    qatten = Qatten(args)
