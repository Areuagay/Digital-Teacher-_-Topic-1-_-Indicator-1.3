import numpy as np
import pandas as pd
import torch
from scipy.sparse import csr_matrix
# from torch.utils.data import Dataset
from torch_geometric.data import InMemoryDataset, Data

from config import parameter_parser

opt = parameter_parser()


class SRGNNDataset(InMemoryDataset):
    """Every session is a graph."""
    def __init__(self, root, phrase, transform=None, pre_transform=None):
        assert phrase in ['train', 'val', 'test','train_val']
        self.phrase = phrase
        super(SRGNNDataset, self).__init__(root, transform, pre_transform)
        self.root = root
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return [self.phrase + '_' + str(opt.max_session) + '.csv']

    @property
    def processed_file_names(self):
        return [self.phrase + '_' + str(opt.max_session) + '_' + opt.model + '.pt']

    def download(self):
        pass

    def process(self):
        path = self.root + '/' + self.raw_file_names[0]
        data = pd.read_csv(path, header=None)

        data_list = []
        for line in range(data.shape[0]):
            if ';' in data.iloc[line,1] == False:
                x = list(map(int, data.iloc[line,1].split(',')))
                for idx in range(len(x)-1):
                    sequences = x[:idx+1]
                    y = x[idx+1]
                    data_list.append(self.processed_seq(sequences,y))
            sequences = list(map(int, data.iloc[line, 2].split(',')))
            y = data.iloc[line, 3]
            data_list.append(self.processed_seq(sequences, y))

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    def processed_seq(self,sequences,y):
        i = 0
        nodes = {}  # dict{15: 0, 16: 1, 18: 2, ...}
        senders = []
        x = []
        for node in sequences:
            if node not in nodes:
                nodes[node] = i
                x.append([node])
                i += 1
            senders.append(nodes[node])
        receivers = senders[:]
        del senders[-1]  # the last item is a receiver
        del receivers[0]  # the first item is a sender
        edge_index = torch.tensor([senders, receivers], dtype=torch.long)
        x = torch.tensor(x, dtype=torch.long)
        y = torch.tensor([y], dtype=torch.long)
        return Data(x=x, edge_index=edge_index, y=y)


class HGGNNDataset(InMemoryDataset):
    def __init__(self,root, phrase,transform=None, pre_transform=None):
        assert phrase in ['train', 'val', 'test','train_val']
        self.phrase = phrase
        super(HGGNNDataset, self).__init__(root,transform,pre_transform)
        self.root = root
        self.data, self.slices = torch.load(self.processed_paths[0])

    def get_data(self):
        data_path = self.root + '/' + self.phrase + '_' + str(opt.max_session) + '.csv'
        data = pd.read_csv(data_path, header=None).iloc[:,[0,2,3]]
        data.columns = [0,1,2]
        return data

    @property
    def raw_file_names(self):
        return [self.phrase + '_' + str(opt.max_session) + '.csv']

    @property
    def processed_file_names(self):
        return [self.phrase + '_' + str(opt.max_session) + opt.model +  '.pt']

    def download(self):
        pass


    def process(self):
        """
        data format:
        <[uid]> <[v1,v2,v3]> <label>
        """
        data = self.get_data()
        data_list = []
        for index in range(data.shape[0]):
            instance = data.iloc[index,:]
            # uid = np.array([instance[0]],dtype=np.int)
            browsed_ids = np.zeros((opt.max_length), dtype=np.int)
            data_1 = list(map(int, instance[1].split(',')))
            seq_len = len(data_1[-opt.max_length:])
            mask = np.array([1 for _ in range(seq_len)] + [0 for _ in range(opt.max_length - seq_len)], dtype=np.int)
            pos_idx=np.array([seq_len-i-1 for i in range(seq_len)]+[0 for _ in range(opt.max_length-seq_len)],dtype=np.int)
            browsed_ids[:seq_len]=np.array(data_1[-opt.max_length:])
            # seq_len = np.array(seq_len, dtype=np.int)
            # label = np.array(instance[2], dtype=np.int)
            # return uid, browsed_ids, mask, seq_len, label, pos_idx
            uid = torch.tensor([instance[0]], dtype=torch.long)
            browsed_ids= torch.tensor([browsed_ids], dtype=torch.long)
            mask= torch.tensor([mask], dtype=torch.long)
            seq_len= torch.tensor([seq_len], dtype=torch.long)
            label= torch.tensor([instance[2]], dtype=torch.long)
            pos_idx= torch.tensor([pos_idx], dtype=torch.long)
            data_list.append(Data(uid=uid, browsed_ids=browsed_ids, mask=mask,seq_len=seq_len,y=label,pos_idx=pos_idx))
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


def data_masks_DHCN(all_sessions, n_node):
    indptr, indices, data = [], [], []
    indptr.append(0)
    for j in range(len(all_sessions)):
        session = np.unique(list(map(int, all_sessions[j].split(','))))
        length = len(session)
        s = indptr[-1]
        indptr.append((s + length))
        for i in range(length):
            # indices.append(session[i]-1)
            indices.append(session[i])
            data.append(1)
    matrix = csr_matrix((data, indices, indptr), shape=(len(all_sessions), n_node))

    return matrix

def adj_DHCN(root):
    if opt.data == 'xing':
        n_node = 59121
    elif opt.data == 'reddit':
        n_node = 27453

    path = root + '/train_50_anonymous.csv'
    data = pd.read_csv(path, header=None)

    H_T = data_masks_DHCN(data[0], n_node)
    BH_T = H_T.T.multiply(1.0 / H_T.sum(axis=1).reshape(1, -1))
    BH_T = BH_T.T
    H = H_T.T
    DH = H.T.multiply(1.0 / H.sum(axis=1).reshape(1, -1))
    DH = DH.T
    DHBH_T = np.dot(DH, BH_T)

    adjacency = DHBH_T.tocoo()
    return adjacency


class DHCNDataset(InMemoryDataset):
    def __init__(self,root, phrase,transform=None, pre_transform=None):
        assert phrase in ['train', 'val', 'test','train_val']
        self.phrase = phrase
        super(DHCNDataset, self).__init__(root,transform,pre_transform)
        self.root = root
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return [self.phrase + '_' + str(opt.max_session) + '_anonymous.csv']

    @property
    def processed_file_names(self):
        return [self.phrase + '_' + str(opt.max_session) + opt.model +  '.pt']

    def download(self):
        pass

    def process(self):
        path = self.root + '/' + self.raw_file_names[0]
        raw_data = pd.read_csv(path, header=None)
        data_list = []
        for line in range(raw_data.shape[0]):
            session = list(map(int, raw_data.iloc[line,0].split(',')))
            y = raw_data.iloc[line,1]
            data_list.append(self.processed_seq(session,y))
            data, slices = self.collate(data_list)
            torch.save((data, slices), self.processed_paths[0])

    def processed_seq(self,session,y):
        session_len = len(session)
        items = session + (opt.max_session - len(session)) * [0]
        mask = [1] * len(session) + (opt.max_session - len(session)) * [0]
        reversed_sess_item = list(reversed(session)) + (opt.max_session - len(session)) * [0]
        y = torch.tensor([y], dtype=torch.long)
        session_len = torch.tensor(session_len, dtype=torch.long)
        items = torch.tensor(items, dtype=torch.long)
        reversed_sess_item = torch.tensor(reversed_sess_item, dtype=torch.long)
        mask = torch.tensor(mask, dtype=torch.long)
        return Data(y=y, session_len=session_len, sess_items=items,reversed_sess_item=reversed_sess_item,mask=mask)