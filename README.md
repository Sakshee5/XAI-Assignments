# Assignment 8

#### Visualize the embedding space of an embedding model on the MTEB leaderboard using tSNE, PCA, and UMAP. Compare/contrast the approaches.


## Dataset Chosen to visualize
**Dataset Name**: GoEmotions

**Description**: The GoEmotions dataset consists of 58,000 Reddit comments annotated with 27 different emotion categorie.

**Reason for Choosing**: By visualizing the embeddings derived from different models, I can analyze how well these models differentiate between various emotions and identify potential clusters that represent semantic similarities.Moreover, easy to validate and visualize wuth true labels. And lastly, it is easy to load!


## Embedding model chosen

1. **CardiffNLP Twitter RoBERTa Base Sentiment (cardiffnlp/twitter-roberta-base-sentiment)**:

**Reason for Choice**: This model is specifically fine-tuned for sentiment analysis on Twitter data. So i figured, it would be interesting to see how it performs on "emotions".  

Number of parameters: 125 million (same as RoBERTa base). <br>
Embedding dimension: 768 dimensions (same as RoBERTa base).

**MTEB Status**: roberta-base models are present on the MTEB leaderboard. I am using a specific twitter-sentiment model since it aligns with the chosen dataset.

2. **NLPTown BERT Multilingual Sentiment (nlptown/bert-base-multilingual-uncased-sentiment):**

**Reason for Choice**: Another sentiment-focused model that can handle multilingual input and provides sentiment scores.

168 Million Parameter model with embedding dimension - 768

**MTEB Status**: Number 250 on the MTEB leaderboard.

3. **all-mpnet-base-v2**:

**Reason for Choice**: This is general-purpose embedding model. I wanted to cluster embeddings of sentiment data on a general model so as to create a baseline for comparison against sentiment-specific models.

Number of parameters: 110 million. <br>
Embedding dimension: 768 dimensions.

**MTEB Status**: Number 200 on the MTEB leaderboard.


