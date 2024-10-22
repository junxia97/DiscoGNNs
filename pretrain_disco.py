from unittest import loader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse
from loader import MoleculeDataset
from dataloader import DataLoaderMasking, DataLoaderMaskingPred #, DataListLoader
from torch_geometric.data import DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import copy
from tqdm import tqdm
import numpy as np
from model import GNN
from sklearn.metrics import roc_auc_score
from splitters import scaffold_split, random_split, random_scaffold_split
import pandas as pd
from util import MaskAtom, ReplaceAtom
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool
import timeit
from tensorboardX import SummaryWriter

criterion = nn.CrossEntropyLoss(reduction = 'none')
criterion_dis = nn.BCELoss()
m = nn.Sigmoid()
n = nn.Softmax()

def compute_accuracy(pred, target):
    return float(torch.sum(torch.max(pred.detach(), dim = 1)[1] == target).cpu().item())/len(pred)

def js_div(net_1_logits, net_2_logits):
    net_1_probs = F.softmax(net_1_logits, dim=0)
    net_2_probs = F.softmax(net_2_logits, dim=0)  
    total_m = 0.5 * (net_1_probs + net_1_probs)
    loss = 0.0
    loss += F.kl_div(F.log_softmax(net_1_logits, dim=0), total_m, reduction="batchmean") 
    loss += F.kl_div(F.log_softmax(net_2_logits, dim=0), total_m, reduction="batchmean") 
    return 0.5 * loss

class graphcl(nn.Module):
    def __init__(self, gnn):
        super(graphcl, self).__init__()
        self.gnn = gnn
        self.pool = global_mean_pool
        self.projection_head = nn.Sequential(nn.Linear(300, 300), nn.ReLU(inplace=True), nn.Linear(300, 300))

    def forward_cl(self, x, edge_index, edge_attr, batch):
        x_node = self.gnn(x, edge_index, edge_attr)
        x = self.pool(x_node, batch)
        x = self.projection_head(x)
        return x_node, x

    def loss_cl(self, x1, x2):
        T = 0.1
        batch_size, _ = x1.size()
        x1_abs = x1.norm(dim=1)
        x2_abs = x2.norm(dim=1)
        sim_matrix = torch.einsum('ik,jk->ij', x1, x2) / torch.einsum('i,j->ij', x1_abs, x2_abs)
        loss_con = js_div(sim_matrix, sim_matrix.t())
        sim_matrix = torch.exp(sim_matrix / T)
        pos_sim = sim_matrix[range(batch_size), range(batch_size)]
        loss_cl = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
        loss_cl = -torch.log(loss_cl).mean()        
        return loss_cl, loss_con

def log(t, eps=1e-9):
    return torch.log(t + eps)
    
def gumbel_noise(t):
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -log(-log(noise))

def gumbel_sample(t, temperature = 1.):
    return ((t / temperature) + gumbel_noise(t)).argmax(dim=-1)

def train(args, train_inter, train_acc_inter, epoch, model_list, dataset, optimizer_list, device):
    model, linear_pred_atoms, linear_pred_bonds, linear_dis_atoms, linear_dis_bonds = model_list
    optimizer_model, optimizer_linear_pred_atoms, optimizer_linear_pred_bonds, optimizer_linear_dis_atoms, optimizer_linear_dis_bonds = optimizer_list
    dataset.aug = "none"
    dataset1 = dataset.shuffle()
    dataset2 = copy.deepcopy(dataset1)
    if args.rep_edge: # Replacement
        dataset1.aug, dataset1.aug_ratio = "RepNE", args.aug_ratio1 
    else:
        dataset1.aug, dataset1.aug_ratio = "RepN", args.aug_ratio1 
    dataset2.aug, dataset2.aug_ratio = args.aug2, args.aug_ratio2 #Subgraph Augmentation.
    loader1 = DataLoaderMasking(dataset1, batch_size=args.batch_size, num_workers = args.num_workers, shuffle=False)
    loader2 = DataLoader(dataset2, batch_size=args.batch_size, num_workers = args.num_workers, shuffle=False)
    model.train()
    linear_pred_atoms.train()
    linear_pred_bonds.train()
    linear_dis_atoms.train()
    linear_dis_bonds.train()
    loss_accum = 0
    acc_node_accum = 0
    acc_edge_accum = 0
    epoch_iter = tqdm(zip(loader1, loader2), desc="Iteration")
    for step, batch in enumerate(epoch_iter):
        optimizer_model.zero_grad()
        optimizer_linear_pred_atoms.zero_grad()
        optimizer_linear_pred_bonds.zero_grad()
        optimizer_linear_dis_atoms.zero_grad()
        optimizer_linear_dis_bonds.zero_grad()
        
        batch, batch_aug = batch
        batch = batch.to(device)
        batch_aug = batch_aug.to(device)
        node_rep, graph_rep = model.forward_cl(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        
        node_dis_prob = m(linear_dis_atoms(node_rep)) # probability of the same atom
        dis_labels = torch.ones(batch.x.size(0)).to(device)
        dis_labels[batch.masked_atom_indices] = (batch.mask_node_label[:, 0] == batch.x[batch.masked_atom_indices, 0]).float()
        dis_loss = criterion_dis(node_dis_prob, dis_labels.unsqueeze(1))
        
        node_cls_prob = linear_pred_atoms(node_rep[batch.masked_atom_indices]) # N * C
        label_onehot = torch.eye(119)[batch.mask_node_label[:,0],:].to(device)
        copy_prob = node_dis_prob[batch.masked_atom_indices]
        prob = n(node_cls_prob) * label_onehot
        cls_loss = -log(dis_labels[batch.masked_atom_indices] * copy_prob + (1 - copy_prob) * prob.sum(1)).mean()

        if step % 100 == 0:
            acc_node = compute_accuracy(node_cls_prob, batch.mask_node_label[:,0])
            acc_node_accum += acc_node

        if args.rep_edge:
            masked_edge_index = batch.edge_index[:, batch.connected_edge_indices]
            edge_rep = node_rep[batch.edge_index[0]] + node_rep[batch.edge_index[1]]
            pred_edge_prob = m(linear_dis_bonds(edge_rep))
            dis_labels_edge = torch.ones(batch.edge_index.size(1)).to(device)
            dis_labels_edge[batch.connected_edge_indices] = (batch.mask_edge_label[:,0] == batch.edge_attr[batch.connected_edge_indices, 0]).float()
            dis_loss += criterion_dis(pred_edge_prob, dis_labels_edge.unsqueeze(1))    
            edge_rep_cls = node_rep[masked_edge_index[0]] + node_rep[masked_edge_index[1]]
            edge_cls_prob = linear_pred_bonds(edge_rep_cls)
            label_onehot = torch.eye(4)[batch.mask_edge_label[:,0],:].to(device)
            copy_prob = pred_edge_prob[batch.connected_edge_indices]
            prob = n(edge_cls_prob) * label_onehot
            cls_loss += -log(dis_labels_edge[batch.connected_edge_indices] * copy_prob + (1 - copy_prob) * prob.sum(1)).mean()
            acc_edge = compute_accuracy(edge_cls_prob, batch.mask_edge_label[:,0])
            acc_edge_accum += acc_edge

        loss_dis = args.weight_dis * dis_loss + cls_loss       
        _, graph_rep_aug = model.forward_cl(batch_aug.x, batch_aug.edge_index, batch_aug.edge_attr, batch_aug.batch)
        loss_xent, loss_con = model.loss_cl(graph_rep_aug, graph_rep)
        loss_cl = loss_xent + loss_con * args.weight_ent
        loss = loss_dis + loss_cl
        loss.backward()
        optimizer_model.step()
        optimizer_linear_pred_atoms.step()
        optimizer_linear_pred_bonds.step()
        optimizer_linear_dis_atoms.step()
        optimizer_linear_dis_bonds.step()
        loss_accum += float(loss.cpu().item())
        epoch_iter.set_description(f"Epoch: {epoch} disloss: {dis_loss:.4f} clsloss: {cls_loss:.4f} tacc: {acc_node:.4f}")

    return loss_accum/step, acc_node_accum/step, acc_edge_accum/step

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch implementation of pre-training of graph neural networks')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='input batch size for training (default: 256)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate (default: 0.001)')
    parser.add_argument('--decay', type=float, default=0,
                        help='weight decay (default: 0)')
    parser.add_argument('--num_layer', type=int, default=5,
                        help='number of GNN message passing layers (default: 5).')
    parser.add_argument('--emb_dim', type=int, default=300,
                        help='embedding dimensions (default: 300)')
    parser.add_argument('--dropout_ratio', type=float, default=0,
                        help='dropout ratio (default: 0)')
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='temperature for gumbel sampling')                       
    parser.add_argument('--JK', type=str, default="last",
                        help='how the node features are combined across layers. last, sum, max or concat')
    parser.add_argument('--dataset', type=str, default = 'zinc_standard_agent', help='root directory of dataset for pretraining')
    parser.add_argument('--input_model_file', type=str, default = '', help='filename to output the model')
    parser.add_argument('--output_model_file', type=str, default = '', help='filename to output the model')
    parser.add_argument('--gnn_type', type=str, default="gin") 
    parser.add_argument('--seed', type=int, default=0, help = "Seed for splitting dataset.")
    parser.add_argument('--num_workers', type=int, default = 8, help='number of workers for dataset loading')
    parser.add_argument('--rep_edge', type=int, default=1, help='whether to replace edges or not together with atoms')
    parser.add_argument('--weight_dis', type=float, default = 0.30) 
    parser.add_argument('--weight_ent', type=float, default = 0.10)                    
    parser.add_argument('--aug_ratio1', type=float, default = 0.15)
    parser.add_argument('--aug2', type=str, default = 'subgraph')
    parser.add_argument('--aug_ratio2', type=float, default = 0.20)
    args = parser.parse_args()

    torch.manual_seed(0)
    np.random.seed(0)
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    print("num layer: %d replace rate: %f replace edge: %d" %(args.num_layer, args.aug_ratio1, args.rep_edge))
    dataset = MoleculeDataset("./dataset/" + args.dataset, dataset=args.dataset)
    gnn = GNN(args.num_layer, args.emb_dim, JK = args.JK, drop_ratio = args.dropout_ratio, gnn_type = args.gnn_type).to(device)
    model = graphcl(gnn).to(device)
    if not args.input_model_file == "":
        model.from_pretrained(args.input_model_file)
    linear_dis_atoms = torch.nn.Linear(args.emb_dim, 1).to(device)
    linear_dis_bonds = torch.nn.Linear(args.emb_dim, 1).to(device)
    linear_pred_atoms = torch.nn.Linear(args.emb_dim, 119).to(device)
    linear_pred_bonds = torch.nn.Linear(args.emb_dim, 4).to(device)
    model_list = [model, linear_pred_atoms, linear_pred_bonds, linear_dis_atoms, linear_dis_bonds]
    #set up optimizers
    optimizer_model = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.decay)
    optimizer_linear_pred_atoms = optim.Adam(linear_pred_atoms.parameters(), lr=args.lr, weight_decay=args.decay)
    optimizer_linear_pred_bonds = optim.Adam(linear_pred_bonds.parameters(), lr=args.lr, weight_decay=args.decay)
    optimizer_linear_dis_atoms = optim.Adam(linear_dis_atoms.parameters(), lr=args.lr, weight_decay=args.decay)
    optimizer_linear_dis_bonds = optim.Adam(linear_dis_bonds.parameters(), lr=args.lr, weight_decay=args.decay)
    optimizer_list = [optimizer_model, optimizer_linear_pred_atoms, optimizer_linear_pred_bonds, optimizer_linear_dis_atoms, optimizer_linear_dis_bonds]
    train_acc_list = []
    train_loss_list = []
    train_acc_inter = []
    train_inter = []
    for epoch in range(1, args.epochs+1):
        print("====epoch " + str(epoch))
        train_loss, train_acc_atom, train_acc_bond = train(args, train_inter, train_acc_inter, epoch, model_list, dataset, optimizer_list, device)
        print(train_loss, train_acc_atom, train_acc_bond)
        train_loss_list.append(train_loss)
        train_acc_list.append(train_acc_atom)    
    df = pd.DataFrame({'train_acc':train_acc_list,'train_loss':train_loss_list}) 
    if not args.output_model_file == "":
        torch.save(model.gnn.state_dict(), args.output_model_file + f"disco_{args.rep_edge}_{args.weight_dis}_{args.weight_ent}_{args.aug2}_{args.aug_ratio2}v6.pth")
    df.to_csv(f'./logs/disco_{args.rep_edge}_{args.weight_dis}_{args.weight_ent}_{args.aug2}_{args.aug_ratio2}v6.csv')  
if __name__ == "__main__":
    main()
