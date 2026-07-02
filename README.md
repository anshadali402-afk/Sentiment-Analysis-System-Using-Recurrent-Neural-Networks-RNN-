# Sentiment-Analysis-System-Using-Recurrent-Neural-Networks-RNN-

# Sentiment Analysis System Using Recurrent Neural Networks (RNN)

## Overview

This project implements a Sentiment Analysis System using a Recurrent Neural Network (RNN) to classify IMDb movie reviews as either positive or negative. The model is trained on the IMDb movie review dataset provided by TensorFlow/Keras and learns sequential relationships between words to predict the sentiment of unseen reviews.

The project also includes a Streamlit web application that allows users to enter a movie review and receive a real-time sentiment prediction.

## Objectives

* Build a sentiment analysis model using a Recurrent Neural Network (RNN).
* Train the model on the IMDb movie review dataset.
* Classify movie reviews as positive or negative.
* Evaluate model performance using standard classification metrics.
* Develop an interactive web application using Streamlit.

## Features

* Binary sentiment classification.
* Automatic text preprocessing.
* Sequence padding for variable-length reviews.
* RNN-based deep learning model.
* Interactive Streamlit user interface.
* Real-time sentiment prediction.
* Model evaluation with accuracy, precision, recall, F1-score, and confusion matrix.

## Technologies Used

* Python
* TensorFlow
* Keras
* NumPy
* Matplotlib
* Scikit-learn
* Streamlit

## Dataset

The project uses the IMDb Movie Review Dataset available through TensorFlow/Keras.

Dataset Information:

* 50,000 movie reviews
* 25,000 training reviews
* 25,000 testing reviews
* Binary sentiment labels:

  * Positive (1)
  * Negative (0)

## Project Structure

```text
Sentiment_Analysis/
│
├── app.py
├── sentiment_rnn_model.h5
├── Sentiment_Analysis_RNN.ipynb
├── requirements.txt
├── README.md
```

## Installation

Clone the repository:

```bash
git clone https://github.com/your-username/sentiment-analysis-rnn.git
```

Navigate to the project directory:

```bash
cd sentiment-analysis-rnn
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Running the Application

Launch the Streamlit application using:

```bash
streamlit run app.py
```

The application will open in your default web browser.

## Model Architecture

The RNN model consists of the following layers:

* Embedding Layer
* SimpleRNN Layer
* Dense Hidden Layer
* Output Layer with Sigmoid Activation

## Model Evaluation

The model is evaluated using the following metrics:

* Accuracy
* Precision
* Recall
* F1-Score
* Confusion Matrix

## Workflow

1. Import required libraries.
2. Load the IMDb dataset.
3. Preprocess the text data.
4. Pad review sequences.
5. Build the RNN model.
6. Compile the model.
7. Train the model.
8. Evaluate the model.
9. Save the trained model.
10. Deploy the model with Streamlit.

## Future Improvements

* Replace SimpleRNN with LSTM or GRU.
* Add support for multi-class sentiment analysis.
* Use pre-trained word embeddings such as GloVe or Word2Vec.
* Deploy the application on Streamlit Community Cloud.
* Improve text preprocessing with advanced Natural Language Processing techniques.

## Learning Outcomes

Through this project, the following concepts are demonstrated:

* Natural Language Processing (NLP)
* Deep Learning
* Recurrent Neural Networks (RNN)
* Text Classification
* Binary Classification
* Model Evaluation
* Streamlit Application Development

## Requirements

* Python 3.10 or later
* TensorFlow
* Streamlit
* NumPy
* Matplotlib
* Scikit-learn

Install all dependencies using:

```bash
pip install -r requirements.txt
```

## License

This project is intended for educational and academic purposes.

If you'd like, I can also provide a more professional GitHub README with sections for screenshots, demo, badges, and installation instructions.

