# AI Generated Text Detection using Clustering

This project analyzes AI-generated and human-written text using clustering techniques, based on the Kaggle competition "LLM Detect AI Generated Text".

## Installation

1. Clone the repository:
```bash
git clone https://github.com/hxngo/ai-text-clustering.git
cd ai-text-clustering
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

## Data

- **Data Source**: [Kaggle - LLM Detect AI Generated Text](https://www.kaggle.com/competitions/llm-detect-ai-generated-text/data)
- **Data Composition**:
  - train_essays.csv: Training essay data (4.2MB)
  - train_prompts.csv: Essay writing prompts (26KB)
  - test_essays.csv: Test essay data (90B)

## Analysis Method

1. **Data Preprocessing**
   - Lowercase conversion
   - HTML tags and special characters removal
   - Stopwords removal
   - English tokenizer-based word tokenization

2. **Feature Extraction**
   - TF-IDF vectorization (max features: 10,000)

3. **Dimensionality Reduction**
   - PCA (n_components=50)
   - Explained variance ratio: 32.04%

4. **Clustering**
   - K-means clustering (n_clusters=2)
   - Silhouette Score: 0.3349

## Results

### Clustering Performance
- **Dataset Size**: 1,378 samples
- **Cluster Sizes**:
  - Cluster 0: 660 samples
  - Cluster 1: 718 samples
- **Label Distribution by Cluster**:
  - Cluster 0: Human-written 99.70%, AI-generated 0.30%
  - Cluster 1: Human-written 99.86%, AI-generated 0.14%
- **Potential Classification Accuracy**: 99.78%

### Characteristic Words by Cluster
- **Cluster 0** (Election/Voting Topics):
  - electoral, vote, college, president, electors, votes, states, popular, election, voters
- **Cluster 1** (Automotive/Transportation Topics):
  - car, cars, usage, people, pollution, driving, limiting, smog, air, transportation

### Visualization Results
- 2D visualization using PCA and t-SNE
- Clear separation by topics
- Topic-based clustering is more distinct than AI/human text separation

## Conclusion

- Clustering successfully separates texts based on topics
- Unsupervised learning is challenging for AI/human text separation due to very low AI-generated text ratio (<1%)
- t-SNE shows clearer cluster structure compared to PCA

## License

This project is licensed under the MIT License.

## References

- [Kaggle - LLM Detect AI Generated Text](https://www.kaggle.com/competitions/llm-detect-ai-generated-text)
- scikit-learn documentation: [K-means Clustering](https://scikit-learn.org/stable/modules/clustering.html#k-means)
- scikit-learn documentation: [PCA](https://scikit-learn.org/stable/modules/decomposition.html#pca)
- scikit-learn documentation: [t-SNE](https://scikit-learn.org/stable/modules/manifold.html#t-sne)