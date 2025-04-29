import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE
import re
import nltk
from nltk.corpus import stopwords
import matplotlib.font_manager as fm

# NLTK 패키지 다운로드
nltk.download('stopwords')
nltk.download('punkt')

# 한글 폰트 설정
# 시스템에 설치된 폰트 중 한글을 지원하는 첫 번째 폰트를 사용
fonts = [f.name for f in fm.fontManager.ttflist if 'Gothic' in f.name or 'Gulim' in f.name or 'Batang' in f.name]
if fonts:
    plt.rcParams['font.family'] = fonts[0]
else:
    print("Warning: 한글 폰트를 찾을 수 없습니다.")
plt.rcParams['axes.unicode_minus'] = False

# 데이터 로드 (Kaggle "LLM Detect AI Generated Text" 대회 데이터셋)
# 링크: https://www.kaggle.com/competitions/llm-detect-ai-generated-text/data
data = pd.read_csv('train_essays.csv')
print("데이터셋 크기:", data.shape)
print(data.head())

# 1. 데이터 전처리 함수
def preprocess_text(text):
    """텍스트 전처리 함수: 소문자화, HTML 태그 제거, 특수문자 제거, 불용어 제거"""
    # 소문자화
    text = text.lower()
    
    # HTML 태그 제거
    text = re.sub(r'<.*?>', '', text)
    
    # 특수문자 제거 (영어, 숫자, 띄어쓰기만 유지)
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    
    # 불용어 제거
    stop_words = set(stopwords.words('english'))
    words = text.split()
    filtered_words = [word for word in words if word not in stop_words]
    
    return ' '.join(filtered_words)

# 전처리 적용
print("텍스트 전처리 중...")
data['processed_text'] = data['text'].apply(preprocess_text)
print("전처리 완료!")

# 2. TF-IDF 벡터화
print("TF-IDF 벡터화 중...")
vectorizer = TfidfVectorizer(max_features=10000)
X_tfidf = vectorizer.fit_transform(data['processed_text'])
print(f"TF-IDF 행렬 형태: {X_tfidf.shape}")

# 3. PCA로 차원 축소
print("PCA로 차원 축소 중...")
pca = PCA(n_components=50, random_state=42)
X_pca = pca.fit_transform(X_tfidf.toarray())
print(f"PCA 결과 형태: {X_pca.shape}")
print(f"PCA 설명된 분산 비율: {sum(pca.explained_variance_ratio_):.4f}")

# 4. K-Means 클러스터링 적용
print("K-Means 클러스터링 중...")
kmeans = KMeans(n_clusters=2, init='k-means++', max_iter=300, random_state=42)
clusters = kmeans.fit_predict(X_pca)
data['cluster'] = clusters

# 5. 군집 품질 평가
silhouette_avg = silhouette_score(X_pca, clusters)
print(f"Silhouette Score: {silhouette_avg:.4f}")

# 각 클러스터의 크기 확인
cluster_sizes = data['cluster'].value_counts()
print("클러스터 크기:")
print(cluster_sizes)

# 실제 레이블과 클러스터 비교
cluster_label_dist = pd.crosstab(data['cluster'], data['generated'], normalize='index') * 100
print("\n클러스터별 레이블 분포 (%):")
print(cluster_label_dist)

# 다수 레이블을 기준으로 클러스터 매핑
cluster_to_label = {}
for cluster in [0, 1]:
    if cluster_label_dist.loc[cluster, 0] > cluster_label_dist.loc[cluster, 1]:
        cluster_to_label[cluster] = 0  # 인간 작성
    else:
        cluster_to_label[cluster] = 1  # AI 생성

print("\n클러스터 매핑:")
for cluster, label in cluster_to_label.items():
    print(f"Cluster {cluster} -> {'AI 생성' if label == 1 else '인간 작성'}")

# 매핑된 레이블 기준 정확도 계산
data['predicted_label'] = data['cluster'].map(cluster_to_label)
accuracy = (data['predicted_label'] == data['generated']).mean()
print(f"\n잠재적 분류 정확도: {accuracy:.4f} ({accuracy*100:.2f}%)")

# 6. 데이터 시각화
# PCA로 2차원 투영
print("시각화를 위한 PCA 2차원 투영 중...")
pca_viz = PCA(n_components=2, random_state=42)
X_pca_2d = pca_viz.fit_transform(X_pca)

# 시각화 (클러스터 기준)
plt.figure(figsize=(10, 8))
sns.scatterplot(x=X_pca_2d[:, 0], y=X_pca_2d[:, 1], hue=data['cluster'], palette='Set1', alpha=0.6)
plt.title('PCA 2D Projection - Clusters')
plt.legend(title='Cluster')
plt.savefig('clusters_pca_2d.png')
plt.close()

# 시각화 (실제 레이블 기준)
plt.figure(figsize=(10, 8))
sns.scatterplot(x=X_pca_2d[:, 0], y=X_pca_2d[:, 1], hue=data['generated'], palette='Set2', alpha=0.6)
plt.title('PCA 2D Projection - Actual Labels')
plt.legend(title='Label', labels=['Human Written', 'AI Generated'])
plt.savefig('labels_pca_2d.png')
plt.close()

# t-SNE 시각화 (선택적으로 수행 - 계산 비용이 높음)
print("t-SNE 2차원 투영 중... (시간이 소요될 수 있습니다)")
# 계산 시간 단축을 위해 데이터 샘플링
sample_size = min(5000, len(data))
sample_indices = np.random.choice(len(data), sample_size, replace=False)

tsne = TSNE(n_components=2, random_state=42, perplexity=30)
X_tsne = tsne.fit_transform(X_pca[sample_indices])

# t-SNE 시각화 (클러스터 기준)
plt.figure(figsize=(10, 8))
sns.scatterplot(x=X_tsne[:, 0], y=X_tsne[:, 1], hue=data['cluster'].iloc[sample_indices], palette='Set1', alpha=0.6)
plt.title('t-SNE 2D Projection - Clusters')
plt.legend(title='Cluster')
plt.savefig('clusters_tsne_2d.png')
plt.close()

# t-SNE 시각화 (실제 레이블 기준)
plt.figure(figsize=(10, 8))
sns.scatterplot(x=X_tsne[:, 0], y=X_tsne[:, 1], hue=data['generated'].iloc[sample_indices], palette='Set2', alpha=0.6)
plt.title('t-SNE 2D Projection - Actual Labels')
plt.legend(title='Label', labels=['Human Written', 'AI Generated'])
plt.savefig('labels_tsne_2d.png')
plt.close()

# 클러스터 특성 분석 - 각 클러스터에서 중요한 단어 추출
def get_important_features(tfidf_vectorizer, kmeans_model, n_features=20):
    """클러스터 중심에서 가장 특징적인 단어 추출"""
    feature_names = tfidf_vectorizer.get_feature_names_out()
    centroids = kmeans_model.cluster_centers_
    
    important_features = {}
    for i, centroid in enumerate(centroids):
        # 센트로이드 벡터에서 가장 값이 큰 단어 인덱스
        sorted_indices = centroid.argsort()[::-1]
        top_indices = sorted_indices[:n_features]
        top_features = [feature_names[idx] for idx in top_indices]
        important_features[i] = top_features
    
    return important_features

# 클러스터별 중요 단어 추출
print("\n클러스터별 특징적인 단어:")
# PCA로 변환된 데이터를 KMeans에 적용했으므로, 원래 TF-IDF 벡터에서 특징 추출
kmeans_tfidf = KMeans(n_clusters=2, init='k-means++', max_iter=300, random_state=42)
kmeans_tfidf.fit(X_tfidf)
important_words = get_important_features(vectorizer, kmeans_tfidf)

for cluster, words in important_words.items():
    predicted_type = "인간 작성" if cluster_to_label[cluster] == 0 else "AI 생성"
    print(f"Cluster {cluster} ({predicted_type})의 특징적인 단어: {', '.join(words)}")

print("\n분석 완료!")