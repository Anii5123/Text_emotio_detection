import streamlit as st
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import numpy as np
import pandas as pd
import altair as alt

# Load model and tokenizer
@st.cache_resource
def load_model_and_tokenizer():
    model = BertForSequenceClassification.from_pretrained("./goemotions_ekman_model")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    return model, tokenizer

model, tokenizer = load_model_and_tokenizer()
model.eval()  # Set model to evaluation mode

# Emotion ID to label mapping (Ekman categories)
id_to_emotion = {
    0: "anger",
    1: "disgust",
    2: "fear",
    3: "joy",
    4: "sadness",
    5: "surprise",
    6: "neutral"
}

# Emoji dictionary
emotions_emoji_dict = {
    "anger": "😠",
    "disgust": "🤮",
    "fear": "😨😱",
    "joy": "🤗",
    "neutral": "😐",
    "sadness": "😔",
    "surprise": "😮"
}

# Emotion prediction function
def predict_emotions(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=1).cpu().numpy()[0]
        predicted_id = int(np.argmax(probabilities))
    return id_to_emotion[predicted_id], probabilities

# Main Streamlit app
def main():
    st.title("🧠 Text Emotion Detection (BERT - GoEmotions)")
    st.subheader("Detect Emotions In Text Using a Fine-tuned BERT Model")

    with st.form(key='emotion_form'):
        raw_text = st.text_area("Enter your text here:")
        submit_button = st.form_submit_button(label="Analyze")

    if submit_button and raw_text.strip():
        col1, col2 = st.columns(2)

        # Prediction
        predicted_emotion, probabilities = predict_emotions(raw_text)

        with col1:
            st.success("Original Text")
            st.write(raw_text)

            st.success("Predicted Emotion")
            emoji_icon = emotions_emoji_dict.get(predicted_emotion, "❓")
            st.write(f"{predicted_emotion.capitalize()} {emoji_icon}")
            st.write("Confidence: {:.2f}%".format(np.max(probabilities) * 100))

        with col2:
            st.success("Prediction Probability Distribution")
            proba_df = pd.DataFrame({
                "emotion": list(id_to_emotion.values()),
                "probability": probabilities
            })
            chart = alt.Chart(proba_df).mark_bar().encode(
                x='emotion',
                y='probability',
                color='emotion'
            )
            st.altair_chart(chart, use_container_width=True)

    else:
        st.info("👈 Enter some text and click Analyze to see the emotion prediction.")

if __name__ == '__main__':
    main()
