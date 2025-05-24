# 📚 LDA Topic Modeling (NumPy Only) for Korean Text

이 프로젝트는 **NumPy, Pandas, SciPy 등만 사용**하여 한국어 뉴스/위키 데이터를 대상으로 **LDA (Latent Dirichlet Allocation)** 를 수행하는 모듈입니다.  
딥러닝 프레임워크 없이 순수한 Gibbs 샘플링 기반으로 구현되며, 다양한 시각화 및 Coherence Metric을 포함합니다.

## 🗂️ 디렉토리 구조

```text
├── data/                          # 원천 데이터 (뉴스/위키, 정치/경제 등)
├── output/                        # 결과물 저장 (category/seed/ngram)
├── dataloader.py                  # 학습/검증 데이터 로딩
├── lda.py                         # NumPy 기반 LDA 모델
├── main.py                        # 전체 실행 흐름 제어
├── visualization.py               # 시각화 및 평가 함수들
└── shell/
    ├── unigram.sh
    ├── bigram.sh
    └── trigram.sh
```

## 📌 실행 방법

```bash
python main.py \
  --category 뉴스 \
  --num_topics 8 \
  --iterations 100 \
  --top_n 10 \
  --seed 42 \
  --ngram 1
```

* --category: 뉴스 또는 위키
* --num_topics: 설정할 토픽 개수
* --iterations: Gibbs 샘플링 반복 수
* --top_n: 토픽별 출력 단어 수
* --seed: 재현을 위한 랜덤 시드
* --ngram: 1 (unigram), 2 (bigram), 3 (trigram)

## 🧩 주요 구성 파일 설명

dataloader.py
* category (뉴스 또는 위키)에 따라 학습/검증 데이터를 JSON에서 불러옵니다.
* load_sentences_by_split() 함수로 train/val을 분리해서 반환합니다.

lda.py
* NumPy 기반 LDA 클래스 LDAModel 구현
* Gibbs 샘플링 기반으로 학습
* 수렴 로그 (log_per_iter) 저장
* n-gram tokenizer 포함
* infer_theta()를 통해 validation 문서에 대한 토픽 분포 추론 가능

visualization.py (분석/시각화 함수)
* plot_topic_word_distribution() : 토픽별 주요 단어 막대그래프
* plot_document_topic_distribution() : 문서-토픽 히트맵
* plot_confusion_matrix() : 실제 label과 토픽 예측의 Confusion Matrix
* plot_label_topic_heatmap() : validation 문서의 label별 평균 토픽 분포 히트맵
* plot_label_dominant_topic_hist() : label별 dominant topic histogram
* plot_topic_convergence_log() : 학습 수렴 로그 그래프
* compute_coherence_scores() : UMass, UCI, PMI 기반 coherence metric 계산

main.py
* 전체 실행 로직을 담당하며 다음을 수행:
* 데이터 로딩 및 시드 설정
* LDA 학습
* 결과 시각화 저장
* coherence score 산출
* validation 분석 수행 및 저장

## Credit
이 프로젝트는 연구 및 교육용 목적으로 제작되었습니다.