import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import platform
import seaborn as sns
from collections import Counter
import os
import math
import pandas as pd
import unicodedata
import warnings

# FutureWarning 무시
warnings.simplefilter(action='ignore', category=FutureWarning)

# 한글 폰트 설정
if platform.system() == 'Darwin':  # macOS
    matplotlib.rc('font', family='AppleGothic')
elif platform.system() == 'Windows':
    matplotlib.rc('font', family='Malgun Gothic')
matplotlib.rcParams['axes.unicode_minus'] = False

# 시각화용 색상 팔레트
PALETTE = sns.color_palette('Set2', 10)

def normalize_labels(labels):
    return [unicodedata.normalize('NFC', l) for l in labels]

def plot_topic_word_distribution(lda_model, top_n=10, save_path=None):
    for k in range(lda_model.K):
        word_ids = np.argsort(lda_model.n_kw[k])[::-1][:top_n]
        words = [lda_model.ID2W[i] for i in word_ids]
        probs = lda_model.n_kw[k][word_ids] / np.sum(lda_model.n_kw[k])
        plt.figure(figsize=(8, 4))
        sns.barplot(x=probs, y=words, palette=PALETTE[:len(words)])
        plt.title(f"Topic {k+1} Top {top_n} Words")
        plt.xlabel("Probability")
        plt.tight_layout()
        if save_path:
            plt.savefig(os.path.join(save_path, f"topic_{k+1}_words.png"))
        plt.close()

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

def plot_topic_label_distribution(labels, topic_assignments, num_topics, save_path=None):
    labels = normalize_labels(labels)
    topic_label_map = {i: Counter() for i in range(num_topics)}
    for label, topic_seq in zip(labels, topic_assignments):
        most_common_topic = Counter(topic_seq).most_common(1)[0][0]
        topic_label_map[most_common_topic][label] += 1
    for topic_idx, label_dist in topic_label_map.items():
        labels_sorted = [l for l, _ in label_dist.most_common()]
        values = [label_dist[l] for l in labels_sorted]
        plt.figure(figsize=(8, 4))
        sns.barplot(x=values, y=labels_sorted, palette=PALETTE[:len(values)])
        plt.title(f"Topic {topic_idx+1} Label Distribution")
        plt.xlabel("Count")
        plt.tight_layout()
        if save_path:
            plt.savefig(os.path.join(save_path, f"topic_{topic_idx+1}_labels.png"))
        plt.close()

def plot_confusion_matrix(labels, topic_assignments, num_topics, save_path=None, prefix=""):
    y_true = normalize_labels(labels)

    # 빈 시퀀스 처리 (For Validation Set)
    y_pred = []
    for topic_seq in topic_assignments:
        if topic_seq:
            most_common = Counter(topic_seq).most_common(1)[0][0]
        else:
            most_common = -1  # 잘못된 값으로 지정 (추후 필터링)
        y_pred.append(most_common)

    # 유효한 인덱스만 필터링
    valid_idx = [i for i, pred in enumerate(y_pred) if pred != -1]
    y_true = [y_true[i] for i in valid_idx]
    y_pred = [y_pred[i] for i in valid_idx]

    label_set = sorted(set(y_true))
    label_to_idx = {label: idx for idx, label in enumerate(label_set)}
    y_true_idx = [label_to_idx[label] for label in y_true]

    cm = np.zeros((len(label_set), num_topics), dtype=int)
    for true_idx, pred_topic in zip(y_true_idx, y_pred):
        cm[true_idx][pred_topic] += 1

    plt.figure(figsize=(10, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=[f"Topic {i+1}" for i in range(num_topics)],
                yticklabels=label_set)
    plt.xlabel("Predicted Topic")
    plt.ylabel("True Label")
    plt.title(f"Confusion Matrix ({prefix})" if prefix else "Confusion Matrix")
    plt.tight_layout()
    if save_path:
        fname = f"confusion_matrix_{prefix}.png" if prefix else "confusion_matrix.png"
        plt.savefig(os.path.join(save_path, fname))
    plt.close()

def compute_coherence_scores(lda_model, top_n_words=10, max_pairs=30, min_df=5):
    umass, uci, pmi = 0.0, 0.0, 0.0
    epsilon = 1e-12
    V = lda_model.n_kw.shape[1]
    D = lda_model.D

    doc_word_matrix = np.zeros((D, V))
    word_doc_freq = np.zeros(V)
    for d, doc in enumerate(lda_model.documents):
        for w in doc:
            wid = lda_model.W2ID[w]
            doc_word_matrix[d][wid] = 1
            word_doc_freq[wid] += 1

    valid_words = set(np.where(word_doc_freq >= min_df)[0])

    for k in range(lda_model.K):
        top_words = [wid for wid in np.argsort(lda_model.n_kw[k])[::-1] if wid in valid_words][:top_n_words]
        pairs_checked = 0
        for i in range(len(top_words)):
            for j in range(i + 1, len(top_words)):
                if pairs_checked >= max_pairs:
                    break
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
                    pmi += max(0, math.log(p_wij / (p_wi * p_wj)))
                pairs_checked += 1

    return {
        'UMass': umass,
        'UCI': uci,
        'PMI': pmi
    }

def plot_label_topic_heatmap(val_theta, val_labels, save_path=None):
    val_labels = normalize_labels(val_labels)
    label_set = sorted(set(val_labels))
    label_to_idx = {label: idx for idx, label in enumerate(label_set)}
    K = val_theta.shape[1]
    avg_topic_by_label = np.zeros((len(label_set), K))
    count_by_label = np.zeros(len(label_set))
    for theta, label in zip(val_theta, val_labels):
        idx = label_to_idx[label]
        avg_topic_by_label[idx] += theta
        count_by_label[idx] += 1
    avg_topic_by_label /= count_by_label[:, None]
    df = pd.DataFrame(avg_topic_by_label, index=label_set, columns=[f"Topic {k+1}" for k in range(K)])
    plt.figure(figsize=(10, 6))
    sns.heatmap(df, annot=True, fmt=".2f", cmap="Blues")
    plt.title("Label-wise Average Topic Distribution (Validation)")
    plt.tight_layout()
    if save_path:
        plt.savefig(os.path.join(save_path, "val_label_topic_heatmap.png"))
    plt.close()

def plot_label_dominant_topic_hist(val_theta, val_labels, save_path=None):
    val_labels = normalize_labels(val_labels)
    label_set = sorted(set(val_labels))
    dominant_topic_by_label = {label: [] for label in label_set}
    for theta, label in zip(val_theta, val_labels):
        top_topic = int(np.argmax(theta))
        dominant_topic_by_label[label].append(top_topic)
    for label in label_set:
        counter = Counter(dominant_topic_by_label[label])
        topics = [t + 1 for t in counter.keys()]
        counts = list(counter.values())
        plt.figure(figsize=(8, 4))
        sns.barplot(x=topics, y=counts, palette=PALETTE[:len(topics)])
        plt.title(f"Dominant Topic Distribution for Label: {label}")
        plt.xlabel("Topic")
        plt.ylabel("Count")
        if save_path:
            plt.savefig(os.path.join(save_path, f"val_dominant_topic_label_{label}.png"))
        plt.close()

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