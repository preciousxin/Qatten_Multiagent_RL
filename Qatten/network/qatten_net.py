import torch
import torch.nn as nn
import torch.nn.functional as F

class QattenNet(nn.Module):
    def __init__(self, args):
        super(QattenNet, self).__init__()
        self.args = args
        self.obs_shape = self.args.obs_shape
        self.state_shape = self.args.state_shape
        self.n_actions = self.args.n_actions
        self.n_agents = self.args.n_agents
        self.num_head = self.args.num_head

        self.query = nn.Sequential(nn.Linear(self.state_shape, self.args.query_hidden_dim1),
                                   nn.ReLU(),
                                   nn.Linear(self.args.query_hidden_dim1, self.args.query_hidden_dim2))
        self.key_extractors = nn.ModuleList()
        for i in range(self.num_head):
            self.key_extractors.append(nn.Sequential(nn.Linear(self.obs_shape, self.args.key_hidden_dim)))


        self.c_value = nn.Sequential(nn.Linear(self.state_shape, self.args.c_hidden_dim1),
                                     nn.ReLU(),
                                     nn.Linear(self.args.c_hidden_dim1, self.args.c_hidden_dim2))

        if self.args.is_head_weight:
            self.head_weight = nn.Sequential(nn.Linear(self.state_shape, self.args.weight_hidden_dim1),
                                             nn.ReLU(),
                                             nn.Linear(self.weight_hidden_dim1, self.weight_hidden_dim2))

    def forward(self, q_values, states, individual_fs):
        # state (episode_num, max_episode_len， state_shape)
        # q_values (episode_num, max_episode_len, n_agents)
        # individual_fs  (episode_num, max_episode_len, n_agents, n_actions)
        episode_num = q_values.size(0)
        episode_len = q_values.size(1)
        q_values = q_values.view(-1, 1, self.n_agents)
        states = states.view(-1, self.state_shape)
        individual_fs = individual_fs.view(-1,1,self.n_agents, self.obs_shape)

        individual_fs_agents = []
        embedding_u_agents = []
        embedding_s = self.query(states)                                  # episode_num * max_episode_len, hidden_query
        Lambda_i_h = torch.zeros(episode_num * episode_len, self.n_agents, self.num_head)
        if self.args.cuda:
            Lambda_i_h = Lambda_i_h.cuda()
        for i in range(self.n_agents):
            individual_fs_agents.append(individual_fs[:,:,i,:].view(-1,self.obs_shape))  # episode_num * max_episode_len, obs_shape
            for h in range(self.num_head):
                embedding_u = self.key_extractors[h](individual_fs_agents[i])  # episode_num * max_episode_len, hidden_key
                lambd = torch.matmul(embedding_s.view(-1,1,self.args.query_hidden_dim2),embedding_u.view(-1,self.args.key_hidden_dim,1))
                lambd= lambd.squeeze(dim=1)                    # episode_num * max_episode_len, 1
                Lambda_i_h[:,i,h] += lambd.squeeze()

        Lambda_i = torch.sum(Lambda_i_h, dim=-1)               # episode_num * max_episode, n_agents
        c_value = self.c_value(states)                         # episode_num * max_episode, 1
        c_value = c_value.view(episode_num, -1, 1)

        # calculate Q-tot
        Q_tot = Lambda_i * q_values.view(-1, self.n_agents)                            # episode_num * max_episode, n_agents
        Q_tot = torch.sum(Q_tot, dim=-1)                       # (episode_num * max_episode, )
        Q_tot = Q_tot.view(episode_num, -1, 1)                 # episode_num, max_episode, 1
        Q_tot = Q_tot + c_value

        return Q_tot






