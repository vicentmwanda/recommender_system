#  Machine Learning At Scale: Design,Implementation and Evaluation of A Movie Recommender System

This project is a **scalable movie recommender system** developed as part of the **Machine Learning at Scale** course at **AIMS South Africa**. 
The system predicts user preferences for movies using **matrix factorization** and handles large-scale datasets efficiently.  

GitHub repository: [https://github.com/vicentmwanda/recommender_system]

##  Project Overview

- **Objective**: Predict ratings that a user would assign to movies and provide personalized recommendations.  
- **Dataset**: MovieLens 32M dataset, containing:
  - 32,000,204 ratings
  - 200,948 users
  - 87,585 movies
  - 20 genres  
- **Challenges addressed**:
  - **Cold start problem** for new users and movies
  - Efficient training on large-scale datasets
  - Long-tail distribution in user activity and movie popularity

Link to the dataset: [https://files.grouplens.org/datasets/movielens/ml-32m.zip]


## Methodology

The system is based on **Alternating Least Squares (ALS)** with  the following experiments:

  **1.Bias-Only Model**: Captures global user and item biases.
  
  **2.Bias + Latent Factors**: Incorporates user and item embeddings (trait vectors) to model interactions.
  
  **3.Feature-Enhanced Model**: Adds genre embeddings to handle cold-start items and improve recommendations for unrated movies.

**Train-Test Splits**: Experiments with three strategies: random, per-user, and per-item.
**Optimization**: Parallelized and vectorized computations using Numba to enable scalable training.

**Cold-start strategies**:
- **User selects any movies they like in database**
- **Mean pooling of genre embeddings**  
- **Feature-based ALS embedding of genres**

##  Project Demo
You can access a demo of the recommender system using the following link:
[https://recommender-demo.streamlit.app/]
