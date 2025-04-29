# AI Generated Text Detection using Clustering

이 프로젝트는 Kaggle의 "LLM Detect AI Generated Text" 대회 데이터를 사용하여 AI가 생성한 텍스트와 인간이 작성한 텍스트를 클러스터링 기법으로 분석합니다.

## 프로젝트 구조

```
Clustering/
├── code/
│   └── Clustering.py      # 메인 분석 코드
├── data/
│   ├── train_essays.csv   # 학습용 에세이 데이터
│   ├── train_prompts.csv  # 프롬프트 데이터
│   ├── test_essays.csv    # 테스트용 에세이 데이터
│   └── sample_submission.csv  # 제출 양식
├── results/
│   ├── labels_pca_2d.png  # PCA 시각화 (실제 레이블)
│   ├── clusters_pca_2d.png  # PCA 시각화 (클러스터)
│   ├── labels_tsne_2d.png  # t-SNE 시각화 (실제 레이블)
│   └── clusters_tsne_2d.png  # t-SNE 시각화 (클러스터)
└── requirements.txt       # 필요한 패키지 목록
```

## 설치 방법

1. 저장소 클론:
```bash
git clone https://github.com/[your-username]/ai-text-clustering.git
cd ai-text-clustering
```

2. 필요한 패키지 설치:
```bash
pip install -r requirements.txt
```

## 데이터

- **데이터 출처**: [Kaggle - LLM Detect AI Generated Text](https://www.kaggle.com/competitions/llm-detect-ai-generated-text/data)
- **데이터 구성**:
  - train_essays.csv: 학습용 에세이 데이터 (4.2MB)
  - train_prompts.csv: 에세이 작성용 프롬프트 (26KB)
  - test_essays.csv: 테스트용 에세이 데이터 (90B)

## 분석 방법

1. **데이터 전처리**
   - 소문자화
   - HTML 태그 및 특수문자 제거
   - 불용어(stopwords) 제거
   - 영어 토크나이저 기반 단어 토큰화

2. **특징 추출**
   - TF-IDF 벡터화 (최대 특징 수: 10,000)

3. **차원 축소**
   - PCA (n_components=50)
   - 설명된 분산 비율: 32.04%

4. **군집화**
   - K-means 클러스터링 (n_clusters=2)
   - Silhouette Score: 0.3349

## 결과

### 클러스터링 성능
- **데이터셋 크기**: 1,378개 샘플
- **클러스터 크기**:
  - Cluster 0: 660개
  - Cluster 1: 718개
- **클러스터별 레이블 분포**:
  - Cluster 0: 인간 작성 99.70%, AI 생성 0.30%
  - Cluster 1: 인간 작성 99.86%, AI 생성 0.14%
- **잠재 분류 정확도**: 99.78%

### 클러스터별 특징적 단어
- **Cluster 0** (선거/투표 주제):
  - electoral, vote, college, president, electors, votes, states, popular, election, voters
- **Cluster 1** (자동차/교통 주제):
  - car, cars, usage, people, pollution, driving, limiting, smog, air, transportation

### 시각화 결과
- PCA와 t-SNE를 사용한 2차원 시각화
- 주제별로 군집이 잘 분리됨
- AI/인간 텍스트 구분보다는 주제별 구분이 더 뚜렷함

## 결론

- 클러스터링은 텍스트의 주제를 기준으로 성공적으로 이루어짐
- 데이터셋의 AI 생성 텍스트 비율이 매우 낮아(1% 미만) 비지도 학습으로는 AI/인간 텍스트 구분이 어려움
- t-SNE가 PCA보다 클러스터 구조를 더 선명하게 보여줌

## 라이선스

이 프로젝트는 MIT 라이선스를 따릅니다.

## 참고자료

- [Kaggle - LLM Detect AI Generated Text](https://www.kaggle.com/competitions/llm-detect-ai-generated-text)
- scikit-learn 문서: [K-means Clustering](https://scikit-learn.org/stable/modules/clustering.html#k-means)
- scikit-learn 문서: [PCA](https://scikit-learn.org/stable/modules/decomposition.html#pca)
- scikit-learn 문서: [t-SNE](https://scikit-learn.org/stable/modules/manifold.html#t-sne) 