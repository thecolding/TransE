import codecs
import numpy as np
import copy
import time
import random

entities2id = {}
relations2id = {}


def dataloader(file1):
    print("load file...")

    entity = []
    relation = []
    triple_list = []

    with codecs.open(file1, 'r', encoding='utf-8') as f:
        content = f.readlines()
        for line in content:
            triple = line.strip().split(",")
            if len(triple) != 3:
                continue

            if triple[0] not in entities2id:
                entities2id[triple[0]] = len(entities2id)
                entity.append(entities2id[triple[0]])

            if triple[2] not in entities2id:
                entities2id[triple[2]] = len(entities2id)
                entity.append(entities2id[triple[2]])

            if triple[1] not in relations2id:
                relations2id[triple[1]] = len(relations2id)
                relation.append(relations2id[triple[1]])

            triple_list.append([entities2id[triple[0]], relations2id[triple[1]], entities2id[triple[2]]])

    print("Complete load. entity : %d , relation : %d , triple : %d" % (
        len(entity), len(relation), len(triple_list)))

    return entity, relation, triple_list


def norm_l1(h, r, t):
    return np.sum(np.fabs(h + r - t))


def norm_l2(h, r, t):
    return np.sum(np.square(h + r - t))


class TransE:
    def __init__(self, entities, relations, triples, embedding_dim=100, lr=0.01, margin=1.0, norm=1):
        self.entities = entities
        self.relations = relations
        self.triples = triples
        self.dimension = embedding_dim
        self.learning_rate = lr
        self.margin = margin
        self.norm = norm
        self.loss = 0.0

    def data_initialise(self):
        entity_vector_list = {}
        relation_vector_list = {}
        for entity in self.entities:
            entity_vector = np.random.uniform(-6.0 / np.sqrt(self.dimension), 6.0 / np.sqrt(self.dimension),
                                              self.dimension)
            entity_vector_list[entity] = entity_vector

        for relation in self.relations:
            relation_vector = np.random.uniform(-6.0 / np.sqrt(self.dimension), 6.0 / np.sqrt(self.dimension),
                                                self.dimension)
            relation_vector = self.normalization(relation_vector)
            relation_vector_list[relation] = relation_vector

        self.entities = entity_vector_list
        self.relations = relation_vector_list

    def normalization(self, vector):
        return vector / np.linalg.norm(vector)

    def training_run(self, epochs=1, nbatches=100, out_file_title=''):

        batch_size = int(len(self.triples) / nbatches)
        print("batch size: ", batch_size)
        for epoch in range(epochs):
            start = time.time()
            self.loss = 0.0
            # Normalise the embedding of the entities to 1
            for entity in self.entities.keys():
                self.entities[entity] = self.normalization(self.entities[entity])

            for batch in range(nbatches):
                batch_samples = random.sample(self.triples, batch_size)

                Tbatch = []
                for sample in batch_samples:
                    corrupted_sample = copy.deepcopy(sample)
                    pr = np.random.random(1)[0]
                    if pr > 0.5:
                        # change the head entity
                        corrupted_sample[0] = random.sample(list(self.entities.keys()), 1)[0]
                        while corrupted_sample[0] == sample[0]:
                            corrupted_sample[0] = random.sample(list(self.entities.keys()), 1)[0]
                    else:
                        # change the tail entity
                        corrupted_sample[2] = random.sample(list(self.entities.keys()), 1)[0]
                        while corrupted_sample[2] == sample[2]:
                            corrupted_sample[2] = random.sample(list(self.entities.keys()), 1)[0]

                    Tbatch.append((sample, corrupted_sample))

                self.update_triple_embedding(Tbatch)
            end = time.time()
            print("epoch: ", epoch, "cost time: %s" % (round((end - start), 3)))
            print("running loss: ", self.loss)

        with codecs.open(out_file_title + "TransE_entity_" + str(self.dimension) + "dim_batch" + str(batch_size), "w") as f1:
            for e in self.entities.keys():
                f1.write(str(e) + "\t")
                f1.write(str(list(self.entities[e])))
                f1.write("\n")

        with codecs.open(out_file_title + "TransE_relation_" + str(self.dimension) + "dim_batch" + str(batch_size), "w") as f2:
            for r in self.relations.keys():
                f2.write(str(r) + "\t")
                f2.write(str(list(self.relations[r])))
                f2.write("\n")

    def update_triple_embedding(self, Tbatch):
        copy_entity = copy.deepcopy(self.entities)
        copy_relation = copy.deepcopy(self.relations)

        for correct_sample, corrupted_sample in Tbatch:

            correct_head = copy_entity[correct_sample[0]]
            correct_tail = copy_entity[correct_sample[2]]
            relation = copy_relation[correct_sample[1]]

            corrupted_head = copy_entity[corrupted_sample[0]]
            corrupted_tail = copy_entity[corrupted_sample[2]]

            if self.norm == 1:
                correct_distance = norm_l1(correct_head, relation, correct_tail)
                corrupted_distance = norm_l1(corrupted_head, relation, corrupted_tail)
            else:
                correct_distance = norm_l2(correct_head, relation, correct_tail)
                corrupted_distance = norm_l2(corrupted_head, relation, corrupted_tail)

            loss = self.margin + correct_distance - corrupted_distance
            if loss > 0:
                self.loss += loss

                correct_gradient = 2 * (correct_head + relation - correct_tail)
                corrupted_gradient = 2 * (corrupted_head + relation - corrupted_tail)

                if self.norm == 1:
                    for i in range(len(correct_gradient)):
                        correct_gradient[i] = 1 if correct_gradient[i] > 0 else -1
                        corrupted_gradient[i] = 1 if corrupted_gradient[i] > 0 else -1

                copy_entity[correct_sample[0]] -= self.learning_rate * correct_gradient
                copy_relation[correct_sample[1]] -= self.learning_rate * correct_gradient
                copy_entity[correct_sample[2]] += self.learning_rate * correct_gradient

                copy_relation[correct_sample[1]] += self.learning_rate * corrupted_gradient
                copy_entity[corrupted_sample[0]] += self.learning_rate * corrupted_gradient
                copy_entity[corrupted_sample[2]] -= self.learning_rate * corrupted_gradient

        self.entities = copy_entity
        self.relations = copy_relation


if __name__ == '__main__':
    file1 = "filtered_data.csv"
    entity_set, relation_set, triple_list = dataloader(file1)

    transE = TransE(entity_set, relation_set, triple_list, embedding_dim=100, lr=0.01, margin=1.0, norm=2)
    transE.data_initialise()
    transE.training_run(out_file_title="filtered_data_")
