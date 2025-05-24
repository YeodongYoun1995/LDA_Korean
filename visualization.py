# visualization.py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import os
import math
import pandas as pd

# 각 토픽의 상위 단어 분포를 막대그래프로 시각화
# 각 토픽에 대해 top_n개의 단어를 확률 기준으로 시각화하여 저장
def plot_topic_word_distribution(lda_model, top_n=10, save_path=None):
    for k in range(lda_model.K):
        word_ids = np.argsort(lda_model.n_kw[k])[::-1][:top_n]
        words = [lda_model.ID2W[i] for i in word_ids]
        probs = lda_model.n_kw[k][word_ids] / np.sum(lda_model.n_kw[k])

        plt.figure(figsize=(8, 4))
        sns.barplot(x=probs, y=words)
        plt.title(f"Topic {k} Top {top_n} Words")
        plt.xlabel("Probability")
        plt.tight_layout()
        if save_path:
            plt.savefig(os.path.join(save_path, f"topic_{k}_words.png"))
        plt.close()

# 학습 문서의 문서-토픽 분포를 히트맵으로 시각화
# 상위 50개의 문서에 대해 각 토픽이 얼마나 할당되었는지를 보여줌
def plot_document_topic_distribution(lda_model, save_path=None):
    doc_topic_matrix = lda_model.n_dk / lda_model.n_dk.sum(axis=1, keepdims=True)
    plt.figure(figsize=(10, 6))
    sns.heatmap(doc_topic_matrix[:50], cmap="Blues", cbar=True)
    plt.title("Document-Topic Distribution (First 50 Docs)")
    plt.xlabel("Topic")
    plt.ylabel("Document")
    plt.tight_layout()
    if save_path:
        plt.savefig(os.path.join(save_path, "doc_topic_heatmap.png"))
    plt.close()

# 각 토픽에 가장 많이 매핑된 라벨 분포를 막대그래프로 시각화
# 주어진 topic_assignments에서 가장 많이 등장한 topic을 기준으로 라벨 분포 생성
def plot_topic_label_distribution(labels, topic_assignments, num_topics, save_path=None):
    topic_label_map = {i: Counter() for i in range(num_topics)}

    for label, topic_seq in zip(labels, topic_assignments):
        most_common_topic = Counter(topic_seq).most_common(1)[0][0]
        topic_label_map[most_common_topic][label] += 1

    for topic_idx, label_dist in topic_label_map.items():
        labels_sorted = [l for l, _ in label_dist.most_common()]
        values = [label_dist[l] for l in labels_sorted]

        plt.figure(figsize=(8, 4))
        sns.barplot(x=values, y=labels_sorted)
        plt.title(f"Topic {topic_idx} Label Distribution")
        plt.xlabel("Count")
        plt.tight_layout()
        if save_path:
            plt.savefig(os.path.join(save_path, f"topic_{topic_idx}_labels.png"))
        plt.close()

# 실제 라벨과 예측된 가장 높은 토픽을 비교한 confusion matrix 시각화
def plot_confusion_matrix(labels, topic_assignments, num_topics, save_path=None, prefix=""):
    y_true = labels
    y_pred = [Counter(topic_seq).most_common(1)[0][0] for topic_seq in topic_assignments]
    label_set = sorted(list(set(labels)))
    label_to_idx = {label: idx for idx, label in enumerate(label_set)}
    y_true_idx = [label_to_idx[l] for l in y_true]

    cm = confusion_matrix(y_true_idx, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_set)
    disp.plot(xticks_rotation=45, cmap='Blues')
    if save_path:
        fname = f"confusion_matrix_{prefix}.png" if prefix else "confusion_matrix.png"
        plt.savefig(os.path.join(save_path, fname))
    plt.close()

# UMass, UCI, PMI 기반의 토픽 일관성(Coherence) 점수 계산
def compute_coherence_scores(lda_model):
    umass, uci, pmi = 0.0, 0.0, 0.0
    epsilon = 1e-12
    V = lda_model.n_kw.shape[1]
    D = lda_model.D

    doc_word_matrix = np.zeros((D, V))
    for d, doc in enumerate(lda_model.documents):
        for w in doc:
            doc_word_matrix[d][lda_model.W2ID[w]] = 1

    word_doc_freq = doc_word_matrix.sum(axis=0)

    for k in range(lda_model.K):
        top_words = np.argsort(lda_model.n_kw[k])[::-1][:10]
        for i in range(len(top_words)):
            for j in range(i + 1, len(top_words)):
                wi, wj = top_words[i], top_words[j]
                D_wi = word_doc_freq[wi]
                D_wj = word_doc_freq[wj]
                D_wij = np.sum(doc_word_matrix[:, wi] * doc_word_matrix[:, wj])

                if D_wij > 0:
                    umass += math.log((D_wij + epsilon) / D_wj)

                p_wi = D_wi / D
                p_wj = D_wj / D
                p_wij = D_wij / D
                if p_wi > 0 and p_wj > 0 and p_wij > 0:
                    uci += math.log(p_wij / (p_wi * p_wj))

                if p_wij > 0:
                    pmi += max(0, math.log(p_wij / (p_wi * p_wj)))

    return {
        'UMass': umass,
        'UCI': uci,
        'PMI': pmi
    }

# validation 데이터에서 레이블별 평균 토픽 분포를 히트맵으로 시각화
def plot_label_topic_heatmap(val_theta, val_labels, save_path=None):
    label_set = sorted(list(set(val_labels)))
    label_to_idx = {label: idx for idx, label in enumerate(label_set)}
    K = val_theta.shape[1]

    avg_topic_by_label = np.zeros((len(label_set), K))
    count_by_label = np.zeros(len(label_set))

    for theta, label in zip(val_theta, val_labels):
        idx = label_to_idx[label]
        avg_topic_by_label[idx] += theta
        count_by_label[idx] += 1

    avg_topic_by_label /= count_by_label[:, None]

    df = pd.DataFrame(avg_topic_by_label, index=label_set, columns=[f"Topic {k}" for k in range(K)])
    plt.figure(figsize=(10, 6))
    sns.heatmap(df, annot=True, fmt=".2f", cmap="Blues")
    plt.title("Label-wise Average Topic Distribution (Validation)")
    plt.tight_layout()
    if save_path:
        plt.savefig(os.path.join(save_path, "val_label_topic_heatmap.png"))
    plt.close()

# validation 문서에서 각 라벨별로 가장 많이 등장한 dominant topic 분포 시각화
def plot_label_dominant_topic_hist(val_theta, val_labels, save_path=None):
    label_set = sorted(set(val_labels))
    dominant_topic_by_label = {label: [] for label in label_set}

    for theta, label in zip(val_theta, val_labels):
        top_topic = int(np.argmax(theta))
        dominant_topic_by_label[label].append(top_topic)

    for label in label_set:
        counter = Counter(dominant_topic_by_label[label])
        topics = list(counter.keys())
        counts = list(counter.values())

        plt.figure(figsize=(8, 4))
        sns.barplot(x=topics, y=counts)
        plt.title(f"Dominant Topic Distribution for Label: {label}")
        plt.xlabel("Topic")
        plt.ylabel("Count")
        if save_path:
            plt.savefig(os.path.join(save_path, f"val_dominant_topic_label_{label}.png"))
        plt.close()

# LDA 학습 중 각 iteration마다 토픽 분포 변화량 기록한 로그 시각화 (수렴 유무 확인)       
def plot_topic_convergence_log(lda_model, save_path=None):
    if not hasattr(lda_model, "log_per_iter"):
        return
    plt.figure(figsize=(8, 4))
    plt.plot(lda_model.log_per_iter)
    plt.xlabel("Iteration")
    plt.ylabel("Avg Topic Word Change")
    plt.title("Topic Distribution Convergence")
    plt.tight_layout()
    if save_path:
        plt.savefig(os.path.join(save_path, "topic_convergence_log.png"))
    plt.close()
