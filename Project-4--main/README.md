# Part 1: Risk Tolerance Prediction with DistilBERT, Sentence Transformers, and Portfolio Recommendation

This project uses machine learning models (DistilBERT and Sentence Transformers) to predict the risk tolerance of investors based on their responses to a set of questions. It also recommends an investment portfolio and visualizes the allocation based on the investor's risk tolerance.

## Features

- **Synthetic Data Generation**: Generates a dataset of 200 investors with responses to 10 predefined questions, each related to their investment preferences based on their risk tolerance.
- **Risk Tolerance Prediction**: Classifies an investor’s risk tolerance into five levels based on their responses to a series of questions.
- **Enhanced Risk Scoring**: Uses sentiment analysis combined with keyword-based semantic similarity to provide a more precise classification of risk tolerance.
- **Portfolio Recommendation**: Based on the investor's risk tolerance, recommends an investment portfolio with varying levels of risk (ranging from conservative to aggressive).
- **Portfolio Visualization**: Displays the recommended portfolio in a pie chart, allowing the investor to visualize their asset allocation. 

## Requirements

- Python 3.x
- PyTorch
- scikit-learn
- Hugging Face Transformers 
- Sentence-Transformers
- pandas
- numpy
- matplotlib

Install the required packages using the following command:

```bash
pip install torch transformers sentence-transformers scikit-learn pandas numpy matplotlib
```

## How It Works

### 1. **Data Generation**

The `templates` dictionary defines sample responses for five different risk tolerance levels. These responses are randomly assigned to 200 synthetic investors, with each investor having 10 responses (one for each question). The data is structured as follows:

- **10 Questions** (`Q1` to `Q10`): Text responses based on the investor's risk level.
- **Risk Tolerance Label**: An integer representing the risk tolerance level (1-5).

### 2. **Data Preprocessing**

The dataset is split into:
- **Features (X)**: Responses to the 10 questions.
- **Labels (y)**: Risk tolerance levels (1-5).

The data is split into 90% for training and 10% for testing using `train_test_split` from `sklearn`.

### 3. **Tokenization**

The responses are tokenized using the DistilBERT tokenizer, which converts text data into tokens (sub-word units) that the model can process.

### 4. **Model Training**

A pre-trained DistilBERT model (`distilbert-base-uncased`) is used for sequence classification. The model is fine-tuned on the training dataset for 10 epochs. The following training parameters are used:

- **Learning rate**: 2e-5
- **Batch size**: 16 (both for training and evaluation)
- **Weight decay**: 0.01 (for regularization)
- **Save strategy**: Save model after each epoch
- **Logging**: Logs every 10 steps
- **Evaluation**: Evaluate at the end of each epoch

### 5. **Enhanced Risk Scoring**

The risk tolerance is predicted by analyzing the sentiment of the user's responses and comparing them to pre-defined keyword combinations. The highest matching risk tolerance level (with a similarity threshold) is selected.

```python
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util
import numpy as np

# Initialize models for sentiment analysis and semantic similarity
sentiment_analyzer = pipeline("sentiment-analysis")
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# Example keyword combinations for risk levels (1 to 5)
keywords = { ... }  # Predefined keywords for each risk score

# Function to classify risk tolerance
def classify_risk_tolerance(response):
    sentiment = sentiment_analyzer(response)[0]
    polarity = sentiment['label']
    response_embedding = embedder.encode(response, convert_to_tensor=True)
    similarity_scores = {}

    for score, embeddings in keyword_embeddings.items():
        cosine_scores = util.cos_sim(response_embedding, embeddings)
        similarity_scores[score] = np.max(cosine_scores.cpu().numpy())

    best_score = max(similarity_scores, key=similarity_scores.get)
    if similarity_scores[best_score] > 0.75:
        return best_score
    if polarity == "NEGATIVE":
        return 1
    elif polarity == "NEUTRAL":
        return 3
    else:
        return 5
```

### 6. **Portfolio Recommendation**

Based on the investor’s risk tolerance level, a diversified portfolio is recommended with asset allocation. The portfolio is adjusted depending on the risk level, ranging from more conservative (with higher bond and cash allocations) to more aggressive (with more equity and cryptocurrency investments).

```python
import matplotlib.pyplot as plt
import numpy as np

# Example diversified portfolio for different risk levels (1: Conservative, 5: Aggressive)
portfolio_distribution = {
    1: {'ETFs': 60, 'Bonds': 30, 'Cash': 10},
    2: {'ETFs': 50, 'Bonds': 30, 'Stocks': 10, 'Crypto': 10},
    3: {'ETFs': 40, 'Bonds': 20, 'Stocks': 20, 'Crypto': 20},
    4: {'ETFs': 30, 'Stocks': 30, 'Crypto': 20, 'Real Estate': 10, 'Bonds': 10},
    5: {'Stocks': 40, 'Crypto': 40, 'ETFs': 10, 'Bonds': 5, 'Cash': 5}
}

# Normalize portfolio allocation to sum to 100%
def recommend_portfolio(risk_level):
    portfolio = portfolio_distribution[risk_level]
    total_percentage = sum(portfolio.values())
    normalized_portfolio = {key: value / total_percentage * 100 for key, value in portfolio.items()}
    return normalized_portfolio

# Visualize the portfolio allocation as a pie chart
def visualize_portfolio(portfolio):
    labels = portfolio.keys()
    sizes = portfolio.values()
    plt.figure(figsize=(8, 8))
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
    plt.title('Recommended Investment Portfolio')
    plt.axis('equal')
    plt.show()
```

### 7. **User Interaction**

The user interacts with the chatbot, providing responses to questions about their investment preferences. Based on the total score from the responses, the chatbot determines the investor’s risk level and recommends a portfolio.

```python
# Function to interact with the user and evaluate responses
def chat_with_user():
    print("Welcome to the Investor Risk Tolerance Chatbot!\n")
    total_score = 0

    # Ask risk tolerance questions
    for question in questions:
        print(question)
        response = input("Your answer: ")
        score = classify_risk_tolerance(response)
        total_score += score
        print(f"Your response has been scored: {score} (Risk Tolerance Level)")

    # Calculate risk level and recommend portfolio
    risk_level = calculate_risk_level(total_score)
    print(f"\nYour risk tolerance level is: {risk_level}")
    portfolio = recommend_portfolio(risk_level)

    print("\nYour recommended portfolio is:")
    for asset, allocation in portfolio.items():
        print(f"{asset}: {allocation:.2f}%")

    visualize_portfolio(portfolio)

# Run the chatbot
if __name__ == "__main__":
    chat_with_user()
```

### 8. **Portfolio Visualization**

Based on the investor's risk tolerance, the chatbot generates a pie chart showing the asset allocation in their recommended portfolio.

---

### Conclusion

This enhanced version of the chatbot not only predicts an investor's risk tolerance but also provides a personalized investment portfolio recommendation. It leverages advanced natural language processing techniques to assess risk and visualizes the asset distribution in an easy-to-understand format. The tool is ideal for investors who want to make informed decisions based on their individual risk profiles.


**Continue to [Part 2](READMEML2.md)**
