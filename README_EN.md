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