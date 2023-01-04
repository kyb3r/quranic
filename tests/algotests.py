from corpus import Corpus, Quran, HadithCollection
import torch
import numpy as np

c = Corpus("x")
c.load_embeddings("quran-embeddings")
# c.save_embeddings("quran-embeddings")


c.documents = Quran().verses # HadithCollection("bukhari").hadiths

print(type(c.doc_embeddings))

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
 



def np_compute_sim(a, b):
    results = []
    for _a in a:
        for _b in b:
            results.append([cosine_similarity(_a, _b)])
    return torch.tensor(results)


def sim_matrix(a, b, eps=1e-8):
    """
    added eps for numerical stability
    """
    a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
    a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
    b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
    sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
    return sim_mt

import time

start = time.time()

embeddings = c.doc_embeddings
e = embeddings

x = torch.tensor([embeddings[0].tolist()])

result = sim_matrix(e, x)
print(result)

zipped = list(zip(result.tolist(), c.documents))
s1 = sorted(zipped, key=lambda x: x[0], reverse=True)[:3]
print(s1)


print(time.time()-start)


start = time.time()
result = np_compute_sim(e, x)


zipped = list(zip(result.tolist(), c.documents))
s2 = sorted(zipped, key=lambda x: x[0], reverse=True)[:3]
print(s2)

print(time.time()-start)


print(all([x[1] == y[1] for x, y in zip(s1, s2)]))