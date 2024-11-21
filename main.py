import numpy as np
import torch
import time
import dgl #pip install --upgrade dgl-cu111
from torch_geometric.loader import DataLoader # conda config --add channels conda-forge // pip install torch-geometric
import torch.nn as nn

from config import parameter_parser
from data_loader import SRGNNDataset,HGGNNDataset
from backbones.SRGNNModel import SRGNNModel
from backbones.HGGNNModel import HG_GNN
from losses import SMRRLoss

opt = parameter_parser()

device = torch.device('cuda:7' if torch.cuda.is_available() else 'cpu')



org_path = 'dataset/' + opt.data
save_path = 'best_model/'

def train(model,device,train_iter,val_iter):
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.learning_rate, weight_decay=opt.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opt.lr_dc_step, gamma=opt.lr_dc)

    total_batch = 0
    last_improve = 0
    loss_list = []
    loss_epoch = []
    best_acc = 0
    tau = 0.01 #0.01

    if opt.loss == 'SMRRLoss':
        L = SMRRLoss(tau)
    else:
        L = nn.CrossEntropyLoss()

    for epoch in range(opt.epoch):
        print('Epoch [{}/{}]'.format(epoch + 1, opt.epoch))
        loss_records = []

        for i, data_batch in enumerate(train_iter):
            """
            opt beta
            """
            outputs = model(data_batch.to(device),device)
            label = data_batch.y
            loss= L(outputs, label.to(device))
            loss_list.append(loss.item())
            loss_records.append(loss.item())
            model.zero_grad()
            loss.backward()
            optimizer.step()
            STEP_SIZE = 500

            if total_batch % STEP_SIZE == 0:
                msg = 'Iter: {0:>6},  Train Loss: {1:>5.6}'
                loss_epoch.append(np.mean(loss_list))
                print(msg.format(total_batch, np.mean(loss_list)))
                loss_list = []
            total_batch += 1
        scheduler.step()

        print('preformance on val set....')
        acc, info = evaluate_topk(model, val_iter)

        if acc > best_acc:
            best_acc = acc
            last_improve = 0
        else:
            last_improve += 1
            if last_improve >= opt.patience:
                print('Early stop: No more improvement')
                break

    return model


def metrics(res, labels):
    res = np.concatenate(res)
    acc = []
    rank = []
    for i in range(len(labels)):
        if labels[i] in res[i]:
            acc.append(1)
            rank.append(np.where(res[i] == labels[i])[0][0] + 1)
        else:
            acc.append(0)
            rank.append(1)
    acc = np.array(acc)
    rank = np.array(rank)
    mrr = (acc / rank).mean()
    ndcg = (acc / np.log2(rank + 1)).mean()
    return acc.mean(), mrr, ndcg

def evaluate_topk(model,data_iter,K=20):
    hit = []
    res50 = []
    res20 = []
    res10 = []
    res5 = []
    mrr = []
    labels = []
    t0 = time.time()
    rank_all = []
    with torch.no_grad():
        for i, data_batch in enumerate(data_iter):
            scores = model(data_batch.to(device),device)
            label = data_batch.y.cpu()
            label_value = torch.diag(scores[:, label].squeeze(-1)).to(device)
            sim_diff = scores - label_value.squeeze(-1)[:, None]
            rank = torch.sum(sim_diff > 0, dim=-1) + 1
            rank_all.extend(rank.tolist())
            sub_scores = scores.topk(K)[1].cpu()
            res20.append(sub_scores)
            res10.append(scores.topk(10)[1].cpu())
            res5.append(scores.topk(5)[1].cpu())
            res50.append(scores.topk(50)[1].cpu())
            labels.append(label)
        labels = np.concatenate(labels)  # .flatten()
        acc50, mrr50, ndcg50 = metrics(res50, labels)
        acc20, mrr20, ndcg20 = metrics(res20, labels)
        acc10, mrr10, ndcg10 = metrics(res10, labels)
        acc5, mrr5, ndcg5 = metrics(res5, labels)

        print("Top20 : acc {}, mrr {}, ndcg {}".format(acc20 * 100, mrr20 * 100, ndcg20 * 100))
        print("Top10 : acc {}, mrr {}, ndcg {}".format(acc10 * 100, mrr10 * 100, ndcg10 * 100))
        print("Top5 : acc {}, mrr {}, ndcg {}".format(acc5 * 100, mrr5 * 100, ndcg5 * 100))

        pred_time = time.time() - t0
        # acc=acc.mean()
        msg = 'Top-{} acc:{:.3f}, mrr:{:.4f}, ndcg:{:.4f}, time: {:.1f}s \n'.format(50, acc50 * 100, mrr50 * 100,
                                                                                    ndcg50 * 100, pred_time)
        msg += 'Top-{} acc:{:.3f}, mrr:{:.4f}, ndcg:{:.4f} \n'.format(20, acc20 * 100, mrr20 * 100,
                                                                                    ndcg20 * 100)
        msg += 'Top-{} acc:{:.3f}, mrr:{:.4f}, ndcg:{:.4f} \n'.format(10, acc10 * 100, mrr10 * 100, ndcg10 * 100)
        msg += 'Top-{} acc:{:.3f}, mrr:{:.4f}, ndcg:{:.4f} \n'.format(5, acc5 * 100, mrr5 * 100, ndcg5 * 100)
        # msg += 'Top-{} acc:{:.3f}, mrr:{:.4f}, ndcg:{:.4f} \n'.format(50, acc50 * 100, mrr50 * 100, ndcg50 * 100)

        return acc20, msg

if __name__ == '__main__':

    data = opt.data
    model = opt.model
    print('dataset:', data,'      model:',model)

    if data == 'xing':
        n_item = 59121
        n_user = 11479
    elif data == 'reddit':
        n_item = 27453
        n_user = 18271

    if model == 'HGGNN':
        train_dataset = HGGNNDataset(org_path, phrase='train')
        train_loader = DataLoader(train_dataset, batch_size=opt.batch_size,shuffle=True)
        val_dataset = HGGNNDataset(org_path, phrase='val')
        val_loader = DataLoader(val_dataset, batch_size=opt.batch_size, shuffle=False)
        test_dataset = HGGNNDataset(org_path, phrase='test')
        test_loader = DataLoader(test_dataset, batch_size=opt.batch_size, shuffle=False)
        g = dgl.load_graphs(org_path +'/'+'graph_' + str( opt.sample_size) + '_' + str(opt.max_length) + '.dgl')[0][0]
        model = HG_GNN(device, g, opt, n_item, opt.max_length).to(device)

    if model == 'SRGNN':
        train_dataset = SRGNNDataset(org_path, phrase='train')
        train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True)
        val_dataset = SRGNNDataset(org_path, phrase='val')
        val_loader = DataLoader(val_dataset, batch_size=opt.batch_size, shuffle=False)
        test_dataset = SRGNNDataset(org_path, phrase='test')
        test_loader = DataLoader(test_dataset, batch_size=opt.batch_size, shuffle=False)
        model = SRGNNModel(hidden_size=opt.hidden_size, n_node=n_item).to(device)

    if opt.no_train:
        model_opt = torch.load(save_path + opt.data + '_' + opt.model + '_' + opt.loss)
        print('preformance on test set....')
        acc, info = evaluate_topk(model_opt, test_loader)
        # print(info)

    else:
        model_opt = train(model,device,train_loader,val_loader)
        print('preformance on test set....')
        acc, info = evaluate_topk(model_opt, test_loader)
        # print(info)
        torch.save(model_opt, save_path + opt.data + '_' + opt.model + '_' + opt.loss)
        print('Model saved!')





