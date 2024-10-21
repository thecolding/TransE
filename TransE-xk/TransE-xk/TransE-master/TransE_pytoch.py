import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from sklearn.model_selection import train_test_split


# 定义数据集类
class TripleDataset(Dataset):
    def __init__(self, triples, entity2id, relation2id):
        self.triples = triples
        self.entity2id = entity2id
        self.relation2id = relation2id

    def __len__(self):
        return len(self.triples)

    def __getitem__(self, idx):
        head, relation, tail = self.triples[idx]
        return self.entity2id[head], self.relation2id[relation], self.entity2id[tail]

# 定义TransE模型
class TransE(nn.Module):
    def __init__(self, num_entities, num_relations, embedding_dim, margin, norm):
        super(TransE, self).__init__()
        self.entity_embeddings = nn.Embedding(num_entities, embedding_dim)
        self.relation_embeddings = nn.Embedding(num_relations, embedding_dim)
        self.margin = margin
        self.norm = norm

        # 初始化参数
        nn.init.xavier_uniform_(self.entity_embeddings.weight)
        nn.init.xavier_uniform_(self.relation_embeddings.weight)

    def forward(self, positive_sample, negative_sample):
        pos_h, pos_r, pos_t = positive_sample
        neg_h, neg_r, neg_t = negative_sample

        # 嵌入
        pos_h = self.entity_embeddings(pos_h)
        pos_r = self.relation_embeddings(pos_r)
        pos_t = self.entity_embeddings(pos_t)
        neg_h = self.entity_embeddings(neg_h)
        neg_r = self.relation_embeddings(neg_r)
        neg_t = self.entity_embeddings(neg_t)

        # 计算距离
        pos_dist = torch.norm(pos_h + pos_r - pos_t, p=self.norm, dim=1)
        neg_dist = torch.norm(neg_h + neg_r - neg_t, p=self.norm, dim=1)

        return pos_dist, neg_dist

    def loss(self, pos_dist, neg_dist):
        y = torch.ones_like(pos_dist)
        loss_fn = nn.MarginRankingLoss(margin=self.margin)
        return loss_fn(pos_dist, neg_dist, y)

# 训练函数
def train_transE(triples, entity2id, relation2id, embedding_dim, lr, epochs, batch_size, margin=6.0, norm=1):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 构建数据集
    dataset = TripleDataset(triples, entity2id, relation2id)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=4)

    # 初始化模型
    num_entities = len(entity2id)
    num_relations = len(relation2id)
    model = TransE(num_entities, num_relations, embedding_dim, margin, norm).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)

    # 训练过程
    for epoch in range(epochs):
        total_loss = 0
        for batch in dataloader:
            positive_sample = batch
            negative_sample = (positive_sample[0].clone(),  # 头实体
                               positive_sample[1],  # 关系不变
                               torch.randint(0, num_entities, (positive_sample[2].size(0),), device=device))  # 生成负样本尾实体

            positive_sample = [tensor.to(device) for tensor in positive_sample]
            negative_sample = [tensor.to(device) for tensor in negative_sample]

            pos_dist, neg_dist = model(positive_sample, negative_sample)
            loss = model.loss(pos_dist, neg_dist)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        if epoch % 10 == 0:
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss}")

    return model  # 返回训练后的模型

# 加载三元组数据并划分数据集
def load_and_split_triples(file_path):
    triples = []
    entity_set = set()
    relation_set = set()

    with open(file_path, 'r', encoding='utf-8') as f:
        next(f)  # 跳过表头
        for line in f:
            columns = line.strip().split(',')
            if len(columns) >= 3:  # 确保至少有三列数据
                head, relation, tail = columns[:3]  # 只取前面三列
                triples.append((head, relation, tail))
                entity_set.add(head)
                entity_set.add(tail)
                relation_set.add(relation)

    entity2id = {entity: idx for idx, entity in enumerate(entity_set)}
    relation2id = {relation: idx for idx, relation in enumerate(relation_set)}

    # 划分数据集：80%训练集，10%验证集，10%测试集
    train_triples, temp_triples = train_test_split(triples, test_size=0.2, random_state=None)
    val_triples, test_triples = train_test_split(temp_triples, test_size=0.5, random_state=None)

    return train_triples, val_triples, test_triples, entity2id, relation2id

def evaluate_model(model, triples, entity2id, relation2id, k=10):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    hit_at_k = 0
    total_rank = 0
    total_reciprocal_rank = 0
    total = 0

    for head, relation, tail in triples:
        head_id = entity2id[head]
        relation_id = relation2id[relation]
        tail_id = entity2id[tail]

        head_vec = model.entity_embeddings(torch.tensor([head_id], device=device)).squeeze(0)
        relation_vec = model.relation_embeddings(torch.tensor([relation_id], device=device)).squeeze(0)

        score = torch.norm(head_vec + relation_vec - model.entity_embeddings.weight, p=1, dim=1)
        ranking = torch.argsort(score)
        rank = (ranking == tail_id).nonzero(as_tuple=True)[0].item()

        total_rank += (rank + 1)
        total_reciprocal_rank += 1 / (rank + 1)
        if rank < k:
            hit_at_k += 1
        total += 1

    print(f'Hit@{k}: {hit_at_k / total}')
    print(f'Mean Rank: {total_rank / total}')
    print(f'MRR: {total_reciprocal_rank / total}')

if __name__ == '__main__':
    # 加载三元组数据并划分数据集
    file_path = 'filtered_data.csv'  # 你的文件路径
    train_triples, val_triples, test_triples, entity2id, relation2id = load_and_split_triples(file_path)

    # 训练TransE模型并返回模型
    model = train_transE(train_triples, entity2id, relation2id, embedding_dim=100, lr=0.003, epochs=200, batch_size=1024)

    # 评估模型在验证集和测试集上的性能
    print("Evaluating on validation set:")
    evaluate_model(model, val_triples, entity2id, relation2id, k=10)

    print("Evaluating on test set:")
    evaluate_model(model, test_triples, entity2id, relation2id, k=10)

