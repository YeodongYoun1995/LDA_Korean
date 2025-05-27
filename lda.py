# lda.py
import numpy as np
import random
from collections import defaultdict
from tqdm import tqdm
import time

class LDAModel:
    def __init__(self, num_topics=8, alpha=0.1, beta=0.01, ngram=1):
        self.num_topics = num_topics
        self.alpha = alpha
        self.beta = beta
        self.ngram = ngram

    def _tokenize(self, text):
        stopwords = set(['있다', '하다', '되다', '으로', '에서', '이다', '그', '저', '이', '를', '에', '의', '가', '은', '는', '과', '도'])
        words = [w for w in text.split() if w not in stopwords and len(w) > 1]
        if self.ngram == 1:
            return words
        else:
            return ['_'.join(words[i:i+self.ngram]) for i in range(len(words)-self.ngram+1)]

    def fit(self, documents, iterations=100):
        start_time = time.time()

        self.documents = [self._tokenize(doc) for doc in documents]
        self.V = list(set(word for doc in self.documents for word in doc))
        self.W2ID = {w: i for i, w in enumerate(self.V)}
        self.ID2W = {i: w for w, i in self.W2ID.items()}
        self.D = len(documents)
        self.K = self.num_topics
        self.z_dn = []
        self.log_per_iter = []  # 수렴 추적용 로그 리스트

        self.n_dk = np.zeros((self.D, self.K), dtype=int)
        self.n_kw = np.zeros((self.K, len(self.V)), dtype=int)
        self.n_k = np.zeros(self.K, dtype=int)

        for d, doc in enumerate(self.documents):
            z_n = []
            for w in doc:
                w_id = self.W2ID[w]
                z = random.randint(0, self.K - 1)
                z_n.append(z)
                self.n_dk[d][z] += 1
                self.n_kw[z][w_id] += 1
                self.n_k[z] += 1
            self.z_dn.append(z_n)

        for it in tqdm(range(iterations), desc="LDA 학습 중 (iterations)"):
            prev_kw = self.n_kw.copy()

            for d, doc in enumerate(self.documents):
                for n, w in enumerate(doc):
                    z = self.z_dn[d][n]
                    w_id = self.W2ID[w]
                    self.n_dk[d][z] -= 1
                    self.n_kw[z][w_id] -= 1
                    self.n_k[z] -= 1

                    p_z = (self.n_kw[:, w_id] + self.beta) * (self.n_dk[d] + self.alpha) / (self.n_k + len(self.V) * self.beta)
                    p_z /= np.sum(p_z)
                    new_z = np.random.choice(self.K, p=p_z)

                    self.z_dn[d][n] = new_z
                    self.n_dk[d][new_z] += 1
                    self.n_kw[new_z][w_id] += 1
                    self.n_k[new_z] += 1

            delta = np.abs(self.n_kw - prev_kw).mean()
            self.log_per_iter.append(delta)

        self.train_time = time.time() - start_time

    def infer_theta(self, new_documents, iterations=20):
        new_docs_tokenized = [self._tokenize(doc) for doc in new_documents]
        V = len(self.V)

        n_kw = self.n_kw
        n_k = self.n_k
        val_z_dn = []

        for doc in new_docs_tokenized:
            z_n = []
            n_dk = np.zeros(self.K, dtype=int)
            word_ids = []

            for w in doc:
                if w not in self.W2ID:
                    continue
                w_id = self.W2ID[w]
                z = random.randint(0, self.K - 1)
                z_n.append(z)
                word_ids.append(w_id)
                n_dk[z] += 1

            if not z_n:
                val_z_dn.append([])  # 빈 시퀀스는 빈 리스트로 추가
                continue

            for _ in range(iterations):
                for i, z in enumerate(z_n):
                    w_id = word_ids[i]
                    n_dk[z] -= 1

                    p_z = (n_kw[:, w_id] + self.beta) * (n_dk + self.alpha)
                    p_z = np.maximum(p_z, 0)
                    p_z_sum = np.sum(p_z)
                    if p_z_sum == 0:
                        p_z = np.ones(self.K) / self.K
                    else:
                        p_z /= p_z_sum

                    new_z = np.random.choice(self.K, p=p_z)
                    z_n[i] = new_z
                    n_dk[new_z] += 1

            val_z_dn.append(z_n)

        return val_z_dn

    def print_top_words(self, top_n=10):
        for k in range(self.K):
            word_ids = np.argsort(self.n_kw[k])[::-1][:top_n]
            top_words = [self.ID2W[i] for i in word_ids]
            print(f"[Topic {k+1}] {' '.join(top_words)}")