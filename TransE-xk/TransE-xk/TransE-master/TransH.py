import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
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

# 定义TransH模型
class TransH(nn.Module):
    def __init__(self, num_entities, num_relations, embedding_dim):
        super(TransH, self).__init__()
        self.entity_embeddings = nn.Embedding(num_entities, embedding_dim)
        self.relation_embeddings = nn.Embedding(num_relations, embedding_dim * 2)
        self.embedding_dim = embedding_dim

        # 初始化参数
        nn.init.xavier_uniform_(self.entity_embeddings.weight)
        nn.init.xavier_uniform_(self.relation_embeddings.weight)

    def forward(self, positive_sample, negative_sample):
        pos_h, pos_r, pos_t = positive_sample
        neg_h, neg_r, neg_t = negative_sample

        # 获取实体和关系的嵌入向量
        pos_h_emb = self.entity_embeddings(pos_h)
        pos_r_emb = self.relation_embeddings(pos_r).view(-1, 2, self.embedding_dim)
        pos_t_emb = self.entity_embeddings(pos_t)
        neg_h_emb = self.entity_embeddings(neg_h)
        neg_r_emb = self.relation_embeddings(neg_r).view(-1, 2, self.embedding_dim)
        neg_t_emb = self.entity_embeddings(neg_t)

        # 计算投影
        pos_h_proj = pos_h_emb + pos_r_emb[:, 0, :]
        pos_t_proj = pos_t_emb - pos_r_emb[:, 1, :]
        neg_h_proj = neg_h_emb + neg_r_emb[:, 0, :]
        neg_t_proj = neg_t_emb - neg_r_emb[:, 1, :]

        # 计算距离
        pos_dist = torch.norm(pos_h_proj - pos_t_proj, p=2, dim=1)
        neg_dist = torch.norm(neg_h_proj - neg_t_proj, p=2, dim=1)

        return pos_dist, neg_dist

    def loss(self, pos_dist, neg_dist):
        y = torch.ones_like(pos_dist)
        loss_fn = nn.MarginRankingLoss(margin=1.0)
        return loss_fn(pos_dist, neg_dist, y)

# 训练函数
def train_transH(triples, entity2id, relation2id, embedding_dim, lr, epochs, batch_size, margin=1.0, norm=2):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 构建数据集
    dataset = TripleDataset(triples, entity2id, relation2id)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=4)

    # 初始化模型
    num_entities = len(entity2id)
    num_relations = len(relation2id)
    model = TransH(num_entities, num_relations, embedding_dim).to(device)

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
    model.eval()  # 将模型设置为评估模式
    hit_at_k = 0
    total_rank = 0
    total_reciprocal_rank = 0
    total = 0

    with torch.no_grad():  # 在评估阶段，不需要计算梯度
        for head, relation, tail in triples:
            head_id = entity2id[head]
            relation_id = relation2id[relation]
            tail_id = entity2id[tail]

            # 获取实体和关系的嵌入向量
            head_vec = model.entity_embeddings(torch.tensor([head_id], device=device)).squeeze(0)
            relation_vec = model.relation_embeddings(torch.tensor([relation_id], device=device))
            w = relation_vec[:, :model.embedding_dim]  # 使用model.embedding_dim
            b = relation_vec[:, model.embedding_dim:]  # 使用model.embedding_dim

            tail_vec = model.entity_embeddings(torch.tensor([tail_id], device=device)).squeeze(0)

            # 计算投影
            head_proj = head_vec + w
            tail_proj = tail_vec - b

            # 计算所有实体的得分
            all_entity_vecs = model.entity_embeddings.weight
            all_scores = torch.norm(all_entity_vecs + w - head_proj, p=2, dim=1)
            all_scores = -all_scores  # 我们希望正样本的得分更高

            # 获取排名
            ranking = torch.argsort(all_scores)
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

    # 训练TransH模型并返回模型
    model = train_transH(train_triples, entity2id, relation2id, embedding_dim=150, lr=0.001, epochs=500, batch_size=1024)

    # 评估模型在验证集和测试集上的性能
    print("Evaluating on validation set:")
    evaluate_model(model, val_triples, entity2id, relation2id, k=10)

    print("Evaluating on test set:")
    evaluate_model(model, test_triples, entity2id, relation2id, k=10)

