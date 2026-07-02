"""
Sentiment Analysis System Using Recurrent Neural Networks (RNN)
Streamlit app version of the RNN.ipynb notebook.

Run with:
    streamlit run app.py
"""

import re
import functools

import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
)

# ----------------------------------------------------------------------------
# Page config
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="RNN Sentiment Analysis",
    page_icon="🎬",
    layout="wide",
)

VOCAB_SIZE = 10000
MAX_LENGTH = 200


# ----------------------------------------------------------------------------
# Cached data / model loading
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_data():
    old_load = np.load
    np.load = functools.partial(old_load, allow_pickle=True)
    try:
        (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=VOCAB_SIZE)
    finally:
        np.load = old_load
    x_train = pad_sequences(x_train, maxlen=MAX_LENGTH, padding="post")
    x_test = pad_sequences(x_test, maxlen=MAX_LENGTH, padding="post")
    return x_train, y_train, x_test, y_test


@st.cache_resource(show_spinner=False)
def get_word_index():
    word_index = imdb.get_word_index()
    # IMDb dataset reserves the first few indices
    word_index = {k: (v + 3) for k, v in word_index.items()}
    word_index["<PAD>"] = 0
    word_index["<START>"] = 1
    word_index["<UNK>"] = 2
    word_index["<UNUSED>"] = 3
    return word_index


@st.cache_resource(show_spinner=False)
def train_model(epochs, batch_size):
    x_train, y_train, x_test, y_test = load_data()

    model = Sequential()
    model.add(Embedding(input_dim=VOCAB_SIZE, output_dim=64))
    model.add(SimpleRNN(units=64, activation="tanh"))
    model.add(Dense(32, activation="relu"))
    model.add(Dense(1, activation="sigmoid"))

    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    history = model.fit(
        x_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2,
        verbose=0,
    )

    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)

    preds_prob = model.predict(x_test, verbose=0)
    preds = (preds_prob > 0.5).astype(int)

    acc = accuracy_score(y_test, preds)
    cm = confusion_matrix(y_test, preds)
    report = classification_report(y_test, preds)

    return {
        "model": model,
        "history": history.history,
        "test_loss": loss,
        "test_accuracy": accuracy,
        "acc_score": acc,
        "confusion_matrix": cm,
        "classification_report": report,
    }


def encode_review(text, word_index, max_length=MAX_LENGTH):
    text = text.lower()
    tokens = re.findall(r"[a-z']+", text)
    encoded = [1]  # <START>
    for token in tokens:
        idx = word_index.get(token, 2)  # <UNK> = 2
        if idx >= VOCAB_SIZE:
            idx = 2
        encoded.append(idx)
    padded = pad_sequences([encoded], maxlen=max_length, padding="post")
    return padded


# ----------------------------------------------------------------------------
# Sidebar - training configuration
# ----------------------------------------------------------------------------
st.sidebar.header("⚙️ Model Settings")
epochs = st.sidebar.slider("Epochs", min_value=1, max_value=10, value=5)
batch_size = st.sidebar.selectbox("Batch size", [32, 64, 128, 256], index=2)
train_button = st.sidebar.button("🚀 Train / Retrain Model", use_container_width=True)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "This app trains a **SimpleRNN** model on the IMDb movie review dataset "
    "for binary sentiment classification (positive / negative)."
)

# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
st.title("🎬 Sentiment Analysis System Using RNN")
st.caption("A SimpleRNN model trained on the IMDb dataset (top 10,000 words, sequences padded to 200 tokens).")

if train_button:
    train_model.clear()

with st.spinner("Training model... this can take a minute or two the first time."):
    results = train_model(epochs, batch_size)

tab1, tab2, tab3 = st.tabs(["📝 Try a Review", "📊 Training Results", "🔍 Dataset Explorer"])

# ----------------------------------------------------------------------------
# Tab 1: Try a review
# ----------------------------------------------------------------------------
with tab1:
    st.subheader("Predict Sentiment for Your Own Review")

    default_text = (
        "This movie was absolutely wonderful, the acting was great and the "
        "story kept me engaged from start to finish."
    )
    user_text = st.text_area("Enter a movie review:", value=default_text, height=150)

    if st.button("Predict Sentiment"):
        word_index = get_word_index()
        encoded = encode_review(user_text, word_index)
        prob = float(results["model"].predict(encoded, verbose=0)[0][0])

        sentiment = "Positive 😀" if prob > 0.5 else "Negative 😞"
        st.metric("Predicted Sentiment", sentiment, f"{prob:.2%} confidence")
        st.progress(min(max(prob, 0.0), 1.0))

    st.markdown("---")
    st.subheader("Or Try a Sample from the Test Set")
    x_train, y_train, x_test, y_test = load_data()
    sample_idx = st.number_input(
        "Test set index (0–24999)", min_value=0, max_value=len(x_test) - 1, value=10, step=1
    )
    if st.button("Predict Sample Review"):
        sample_review = x_test[sample_idx].reshape(1, -1)
        prob = float(results["model"].predict(sample_review, verbose=0)[0][0])
        actual = "Positive" if y_test[sample_idx] == 1 else "Negative"
        predicted = "Positive 😀" if prob > 0.5 else "Negative 😞"

        col1, col2 = st.columns(2)
        col1.metric("Actual Label", actual)
        col2.metric("Predicted", predicted, f"{prob:.2%} confidence")

# ----------------------------------------------------------------------------
# Tab 2: Training results
# ----------------------------------------------------------------------------
with tab2:
    st.subheader("Model Performance")

    col1, col2, col3 = st.columns(3)
    col1.metric("Test Loss", f"{results['test_loss']:.4f}")
    col2.metric("Test Accuracy", f"{results['test_accuracy']:.2%}")
    col3.metric("Accuracy (sklearn)", f"{results['acc_score']:.2%}")

    hist = results["history"]

    fig_col1, fig_col2 = st.columns(2)

    with fig_col1:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(hist["accuracy"], label="Train")
        ax.plot(hist["val_accuracy"], label="Validation")
        ax.set_title("Training Accuracy")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy")
        ax.legend()
        st.pyplot(fig)

    with fig_col2:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(hist["loss"], label="Train")
        ax.plot(hist["val_loss"], label="Validation")
        ax.set_title("Training Loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend()
        st.pyplot(fig)

    st.subheader("Confusion Matrix")
    cm = results["confusion_matrix"]
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Negative", "Positive"])
    ax.set_yticklabels(["Negative", "Positive"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")
    st.pyplot(fig)

    st.subheader("Classification Report")
    st.code(results["classification_report"])

# ----------------------------------------------------------------------------
# Tab 3: Dataset explorer
# ----------------------------------------------------------------------------
with tab3:
    st.subheader("IMDb Dataset Overview")
    x_train, y_train, x_test, y_test = load_data()

    col1, col2 = st.columns(2)
    col1.metric("Training Samples", len(x_train))
    col2.metric("Testing Samples", len(x_test))

    st.write(f"Vocabulary size: **{VOCAB_SIZE:,}** | Sequence length: **{MAX_LENGTH}**")

    idx = st.number_input(
        "View encoded training sample index", min_value=0, max_value=len(x_train) - 1, value=0, step=1
    )
    st.write("Encoded review (padded):")
    st.code(str(x_train[idx].tolist()))
    st.write(f"Label: **{'Positive' if y_train[idx] == 1 else 'Negative'}**")
