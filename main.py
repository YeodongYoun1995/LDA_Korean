# main.py
import argparse
import os
import json
import random
import numpy as np
from dataloader import load_sentences_by_split
from lda import LDAModel
from visualization import (
    plot_topic_word_distribution,
    plot_document_topic_distribution,
    plot_topic_label_distribution,
    plot_confusion_matrix,
    compute_coherence_scores,
    plot_topic_convergence_log
)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--category', choices=['뉴스', '위키'], required=True, help="데이터 카테고리: 뉴스 또는 위키")
    parser.add_argument('--num_topics', type=int, default=8, help="토픽 개수")
    parser.add_argument('--iterations', type=int, default=100, help="Gibbs 샘플링 반복 횟수")
    parser.add_argument('--top_n', type=int, default=10, help="각 토픽별 상위 단어 수")
    parser.add_argument('--seed', type=int, default=42, help="랜덤 시드")
    parser.add_argument('--ngram', type=int, default=1, help="n-gram 설정 (1이면 unigram, 2면 bigram 등)")
    args = parser.parse_args()

    ngram_str = {1: "unigram", 2: "bigram", 3: "trigram"}.get(args.ngram, f"{args.ngram}gram")
    output_path = f"output/{args.category}/{args.seed}/{ngram_str}"
    os.makedirs(output_path, exist_ok=True)

    print("[INFO] 하이퍼파라미터 설정:")
    print(f"  Category      : {args.category}")
    print(f"  Num Topics    : {args.num_topics}")
    print(f"  Iterations    : {args.iterations}")
    print(f"  Top N Words   : {args.top_n}")
    print(f"  Random Seed   : {args.seed}")
    print(f"  N-gram        : {args.ngram} ({ngram_str})")

    print("[INFO] 데이터 로딩 중...")
    train_docs, train_labels, val_docs, val_labels = load_sentences_by_split('data', args.category)
    print(f"  Training docs : {len(train_docs)}")
    print(f"  Validation docs: {len(val_docs)}")

    print("[INFO] 랜덤 시드 설정...")
    random.seed(args.seed)
    np.random.seed(args.seed)

    print("[INFO] LDA 모델 학습 시작...")
    lda = LDAModel(num_topics=args.num_topics, ngram=args.ngram)
    lda.fit(train_docs, iterations=args.iterations)

    print(f"[INFO] 학습 시간: {lda.train_time:.2f}초")
    print("[INFO] 토픽별 주요 단어 출력:")
    lda.print_top_words(top_n=args.top_n)

    print("[INFO] 시각화 및 저장 시작 (Training)...")
    plot_topic_word_distribution(lda, top_n=args.top_n, save_path=output_path)
    plot_document_topic_distribution(lda, save_path=output_path)
    plot_topic_label_distribution(train_labels, lda.z_dn, lda.num_topics, save_path=output_path)
    plot_confusion_matrix(train_labels, lda.z_dn, lda.num_topics, save_path=output_path, prefix="train")
    plot_topic_convergence_log(lda, save_path=output_path)

    print("[INFO] Coherence Score 계산 중 (Training)...")
    coherence_scores = compute_coherence_scores(lda)
    with open(os.path.join(output_path, 'coherence_scores_train.json'), 'w', encoding='utf-8') as f:
        json.dump(coherence_scores, f, ensure_ascii=False, indent=2)
    print(f"[INFO] Training Coherence Scores: {coherence_scores}")

    print("[INFO] Validation 데이터 추론 중...")
    val_z_dn = lda.infer_theta(val_docs, iterations=20)
    plot_confusion_matrix(val_labels, val_z_dn, lda.num_topics, save_path=output_path, prefix="val")

if __name__ == '__main__':
    main()