import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.nn.pytorch as dglnn


class HG_GNN(nn.Module):
    def __init__(self,device,G,config,item_num,max_seq_len=20,max_sess=50):
        super(HG_GNN, self).__init__()
        self.G = G.to(device)
        self.max_sess = max_sess
        self.hidden_size = config.hidden_size
        self.em_size = config.embed_size
        self.pos_embedding = nn.Embedding(200, self.em_size)
        self.v2e = nn.Embedding(G.number_of_nodes(), self.em_size).to(device)

        self.conv1 = dglnn.SAGEConv(self.em_size, self.em_size, 'mean')

        dropout = config.dropout
        self.emb_dropout = nn.Dropout(p=dropout)
        self.gru = nn.GRU(self.em_size, self.hidden_size, 1)
        self.max_seq_len = max_seq_len
        self.W = nn.Linear(self.em_size, self.em_size)

        # node embedding
        self.linear_one = nn.Linear(self.em_size, self.em_size, bias=True)
        self.linear_two = nn.Linear(self.em_size, self.em_size, bias=True)
        self.linear_three = nn.Linear(self.em_size, 1, bias=False)

        # gru embedding
        self.a_1 = nn.Linear(self.hidden_size, self.hidden_size)
        self.a_2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.v_t = nn.Linear(self.hidden_size, 1, bias=False)

        self.ct_dropout = nn.Dropout(dropout)

        self.user_transform = nn.Sequential(
            nn.Linear(self.em_size, self.em_size, bias=True)
        )

        self.gru_transform = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.em_size, bias=True)
        )

        self.sigmoid_concat = nn.Sequential(
            nn.Linear(self.em_size * 2, 1, bias=True),
            nn.Sigmoid()
        )

        self.w_1 = nn.Parameter(torch.Tensor(2 * self.em_size, self.em_size))
        self.w_2 = nn.Parameter(torch.Tensor(self.em_size, 1))
        self.glu1 = nn.Linear(self.em_size, self.em_size)
        self.glu2 = nn.Linear(self.em_size, self.em_size, bias=False)

        self.w_3 = nn.Parameter(torch.Tensor(self.em_size, self.em_size))
        self.w_4 = nn.Parameter(torch.Tensor(self.em_size, 1))
        self.glu3 = nn.Linear(self.em_size, self.em_size)
        self.glu4 = nn.Linear(self.em_size, self.em_size, bias=False)

        self.reset_parameters()

        self.item_num = item_num

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.em_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def compute_hidden_vector(self, hidden, mask, pos_idx):
        mask = mask.float().unsqueeze(-1)
        batch_size = hidden.shape[0]
        len = hidden.shape[1]
        tmp = torch.sum(hidden * mask, -2) / torch.sum(mask, 1)
        hs = tmp.unsqueeze(-2).repeat(1, len, 1)
        # 退化掉位置信息
        pos_emb = self.pos_embedding(pos_idx)
        nh = torch.matmul(torch.cat([pos_emb, hidden], -1), self.w_1)
        nh = torch.tanh(nh)
        # nh = hidden
        # 退化掉物品重要性信息
        nh = torch.sigmoid(self.glu1(nh) + self.glu2(hs))
        beta = torch.matmul(nh, self.w_2)
        beta = beta * mask
        select = torch.sum(beta * hidden, 1)
        # select = torch.sum(nh, 1)
        return select, tmp

    def sess_user_vector(self, user_vec, note_embeds, mask):
        """
        user_vec:
        note_embeds:
        mask:
        """
        mask = mask.float().unsqueeze(-1)
        hs = user_vec.unsqueeze(-2).repeat(1, mask.shape[1], 1)
        # hs = user_vec.repeat(1, mask.shape[1], 1)
        nh = torch.matmul(note_embeds, self.w_3)
        nh = torch.tanh(nh)
        nh = torch.sigmoid(self.glu3(nh) + self.glu4(hs))
        beta = torch.matmul(nh, self.w_4)
        beta = beta * mask
        select = torch.sum(beta * note_embeds, 1)

        return select

    def softmax(self,X):
        X_exp = torch.exp(X)
        partition = torch.sum(X_exp, dim=1, keepdim=True)
        return X_exp / partition

    # def forward(self, device, data):
    def forward(self, data,device):
        """
        seq(bs*L)
        seq: bs*L
        his_ids: bs * M
        mask:
        seq_len(bs)
        """
        user, seq, mask, seq_len, pos_idx = data.uid, data.browsed_ids, data.mask, data.seq_len, data.pos_idx
        user = user + self.item_num

        # HG-GNN
        # 并没有解耦为同质图再进行学习
        h1 = self.conv1(self.G,
                        self.emb_dropout(self.v2e(torch.arange(0, self.G.number_of_nodes()).long().to(device))))

        h1 = F.relu(h1)
        # h2 = self.conv1(self.G,h1)
        # h2 = F.relu(h2)
        # h = (h1 + h2)/2
        bs = seq.size()[0]
        L = seq.size()[1]
        node_list = seq
        item_embeds = (h1[node_list] + self.v2e(node_list)) / 2
        user_embeds = (h1[user] + self.v2e(user)) / 2
        node_embeds = item_embeds.view((bs, L, -1))
        user_embeds = user_embeds.squeeze()
        seq_embeds = user_embeds
        # print(mask.shape) torch.Size([512, 20])
        # print(user_embeds.shape) torch.Size([512, 1, 128])

        sess_vec, avg_sess = self.compute_hidden_vector(node_embeds, mask, pos_idx)

        # 退化掉 4.4.1
        # seq_embeds = avg_sess

        # 退化掉 4.4.2
        sess_user = self.sess_user_vector(user_embeds, node_embeds, mask)
        alpha = self.sigmoid_concat(torch.cat([sess_vec, sess_user], 1))
        seq_embeds += (alpha * sess_vec + (1 - alpha) * sess_user)

        # seq_embeds = sess_vec

        item_ID_embs = self.v2e.weight
        scores = torch.matmul(seq_embeds, item_ID_embs.permute(1, 0))

        return scores
