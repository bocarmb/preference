import streamlit as st
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torch.nn.functional import softmax


# Charger le modèle et le tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")
model = BertForSequenceClassification.from_pretrained("bert-base-multilingual-cased", num_labels=2)

# Dictionnaire de mots toxiques (exemple, à compléter)
toxic_words = {
    "insultes": ["idiot", "stupide", "abruti", "imbécile","domaram","domouharam","sa thiotou ndeye","doul rek","ya meun fén","mena féne"],
    "haine": ["déteste", "haineux", "raciste", "sexiste"],
    "violence": ["tuer", "frapper", "agresser", "massacrer","déloger","détruire","tuer","tuerie","massacre","massacrer","massacrer",],
}

# Fonction pour détecter la présence de mots toxiques
def detect_toxic_words(comment):
    detected_words = []
    for category, words in toxic_words.items():
        for word in words:
            if word in comment.lower():
                detected_words.append((word, category))
    return detected_words




# Fonction de prédiction
def predict_comment(comments):
    inputs = tokenizer(comments, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
    probabilities = softmax(outputs.logits, dim=1).squeeze().tolist()
    classes = ["Toxique", "Non toxique"]
    result = {classes[i]: round(probabilities[i] * 100, 2) for i in range(len(classes))}
    return result

# Interface Streamlit
st.title("Modération Automatique des Commentaires")
st.write("Entrez un commentaire pour analyser son niveau de toxicité.")

user_input = st.text_area("Votre commentaire")
if st.button("Analyser"):
    if user_input.strip():
        prediction = predict_comment(user_input)
        st.write("### Résultats de l'analyse :")
        for label, score in prediction.items():
            st.write(f"**{label}** : {score}%")
    else:
        st.warning("Veuillez entrer un commentaire.")

