import numpy as np
from numpy import linalg as la
from utils import dictionary
import pickle

def load_embedding():
    with open('word-embedding.mat', 'rb') as f:
        return pickle.load(f)

def read_questions():
    questions = []
    with open("questions-words.txt", "r") as fi:
        for line in fi:
            if line.startswith(':'):
                continue
            words = line.strip().lower().split(' ')
            ids = [dictionary.get(word) for word in words]
            if not all(ids):
                continue
            questions.append(ids)
    return np.array(questions)

# embedding: [vocabulary_size, embedding_size]
def predict(embedding, ws):
    # normalize each embedding vector
    # so that cosine distance can be computed easier
    nemb = embedding / np.maximum(la.norm(embedding, axis=1), 1e-12).reshape([embedding.shape[0],1])
    a = nemb[ws[0]]
    b = nemb[ws[1]]
    c = nemb[ws[2]]
    d = b - a + c
    dis = np.matmul(d, np.transpose(nemb))
    return np.argpartition(dis, -4)[-4:]

def eval(embedding, questions):
    correct = 0
    total = questions.shape[0]
    for ind in range(total):
      print('\r  {}/{}'.format(ind, total), end = '\r')
      qu = questions[ind]
      ps = predict(embedding, qu[:3])
      for w in ps:
        if w == qu[3]:
            correct += 1
            break
    print("Eval %4d/%d accuracy = %4.1f%%" % (correct, total, correct * 100.0 / total))