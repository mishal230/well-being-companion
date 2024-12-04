import os
import gradio as gr
import nltk
import numpy as np
import json
import pickle
from nltk.tokenize import word_tokenize
from nltk.stem.lancaster import LancasterStemmer
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# Suppress TensorFlow warnings
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# NLTK Setup
nltk.download("punkt")
stemmer = LancasterStemmer()

# Load data
with open("intents.json") as file:
    intents_data = json.load(file)

with open("data.pickle", "rb") as f:
    words, labels, training, output = pickle.load(f)

# Hugging Face models for Well-Being Companion
tokenizer_sentiment = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
model_sentiment = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
tokenizer_emotion = AutoTokenizer.from_pretrained("j-hartmann/emotion-english-distilroberta-base")
model_emotion = AutoModelForSequenceClassification.from_pretrained("j-hartmann/emotion-english-distilroberta-base")

# Helper Functions
def bag_of_words(s, words):
    bag = [0] * len(words)
    s_words = word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words if word.isalnum()]
    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1
    return np.array(bag)
def analyze_sentiment(user_input):
    """Analyze sentiment and map to emojis."""
    inputs = tokenizer_sentiment(user_input, return_tensors="pt")
    with torch.no_grad():
        outputs = model_sentiment(**inputs)
    sentiment_class = torch.argmax(outputs.logits, dim=1).item()
    sentiment_map = ["Negative 😔", "Neutral 😐", "Positive 😊"]
    return f"Sentiment: {sentiment_map[sentiment_class]}"

def detect_emotion(user_input):
    """Detect emotions based on input."""
    pipe = pipeline("text-classification", model=model_emotion, tokenizer=tokenizer_emotion)
    result = pipe(user_input)
    emotion = result[0]["label"].lower().strip()
    emotion_map = {
        "joy": "Joy 😊",
        "anger": "Anger 😠",
        "sadness": "Sadness 😢",
        "fear": "Fear 😨",
        "surprise": "Surprise 😲",
        "neutral": "Neutral 😐",
    }
    # Return only the formatted emotion string
    return emotion_map.get(emotion, "Unknown 🤔")

def generate_suggestions(emotion):
    """Return relevant suggestions based on detected emotions."""
    emotion_key = emotion.lower()
    suggestions = {
        "joy": [
            ["Relaxation Techniques", "https://www.helpguide.org/mental-health/meditation/mindful-breathing-meditation"],
            ["Dealing with Stress", "https://www.helpguide.org/mental-health/anxiety/tips-for-dealing-with-anxiety"],
            ["Emotional Wellness Toolkit", "https://www.nih.gov/health-information/emotional-wellness-toolkit"],
            ["Relaxation Video", "https://youtu.be/m1vaUGtyo-A"],
        ],
        "anger": [
            ["Emotional Wellness Toolkit", "https://www.nih.gov/health-information/emotional-wellness-toolkit"],
            ["Stress Management Tips", "https://www.health.harvard.edu/health-a-to-z"],
            ["Dealing with Anger", "https://www.helpguide.org/mental-health/anxiety/tips-for-dealing-with-anxiety"],
            ["Relaxation Video", "https://youtu.be/MIc299Flibs"],
        ],
        "fear": [
            ["Mindfulness Practices", "https://www.helpguide.org/mental-health/meditation/mindful-breathing-meditation"],
            ["Coping with Anxiety", "https://www.helpguide.org/mental-health/anxiety/tips-for-dealing-with-anxiety"],
            ["Emotional Wellness Toolkit", "https://www.nih.gov/health-information/emotional-wellness-toolkit"],
            ["Relaxation Video", "https://youtu.be/yGKKz185M5o"],
        ],
        "sadness": [
            ["Emotional Wellness Toolkit", "https://www.nih.gov/health-information/emotional-wellness-toolkit"],
            ["Dealing with Anxiety", "https://www.helpguide.org/mental-health/anxiety/tips-for-dealing-with-anxiety"],
            ["Relaxation Video", "https://youtu.be/-e-4Kx5px_I"],
        ],
        "surprise": [
            ["Managing Stress", "https://www.health.harvard.edu/health-a-to-z"],
            ["Coping Strategies", "https://www.helpguide.org/mental-health/anxiety/tips-for-dealing-with-anxiety"],
            ["Relaxation Video", "https://youtu.be/m1vaUGtyo-A"],
        ],
    }
    return suggestions.get(emotion_key, [["No specific suggestions available.", "#"]])

def get_health_professionals_and_map(location, query):
    """Search nearby healthcare professionals using Google Maps API."""
    try:
        if not location or not query:
            return [], ""  # Return empty list if inputs are missing

        geo_location = gmaps.geocode(location)
        if geo_location:
            lat, lng = geo_location[0]["geometry"]["location"].values()
            places_result = gmaps.places_nearby(location=(lat, lng), radius=10000, keyword=query)["results"]
            professionals = []
            map_ = folium.Map(location=(lat, lng), zoom_start=13)
            for place in places_result:
                # Use a list of values to append each professional
                professionals.append([place['name'], place.get('vicinity', 'No address provided')])
                folium.Marker(
                    location=[place["geometry"]["location"]["lat"], place["geometry"]["location"]["lng"]],
                    popup=f"{place['name']}"
                ).add_to(map_)
            return professionals, map_._repr_html_()

        return [], ""  # Return empty list if no professionals found
    except Exception as e:
        return [], ""  # Return empty list on exception


# Chronic Disease Prediction Functions
def load_data():
    df = pd.read_csv("Training.csv")
    tr = pd.read_csv("Testing.csv")
    disease_dict = {  'Fungal infection': 0, 'Allergy': 1, 'GERD': 2, 'Chronic cholestasis': 3, 'Drug Reaction': 4,
        'Peptic ulcer diseae': 5, 'AIDS': 6, 'Diabetes ': 7, 'Gastroenteritis': 8, 'Bronchial Asthma': 9,
        'Hypertension ': 10, 'Migraine': 11, 'Cervical spondylosis': 12, 'Paralysis (brain hemorrhage)': 13,
        'Jaundice': 14, 'Malaria': 15, 'Chicken pox': 16, 'Dengue': 17, 'Typhoid': 18, 'hepatitis A': 19,
        'Hepatitis B': 20, 'Hepatitis C': 21, 'Hepatitis D': 22, 'Hepatitis E': 23, 'Alcoholic hepatitis': 24,
        'Tuberculosis': 25, 'Common Cold': 26, 'Pneumonia': 27, 'Dimorphic hemmorhoids(piles)': 28,
        'Heart attack': 29, 'Varicose veins': 30, 'Hypothyroidism': 31, 'Hyperthyroidism': 32,
        'Hypoglycemia': 33, 'Osteoarthristis': 34, 'Arthritis': 35,
        '(vertigo) Paroymsal  Positional Vertigo': 36, 'Acne': 37, 'Urinary tract infection': 38,
        'Psoriasis': 39, 'Impetigo': 40
    } # Same logic.
    df.replace({'prognosis': disease_dict}, inplace=True)
    return df, tr, disease_dict

df, tr, disease_dict = load_data()
l1 = list(df.columns[:-1])
X = df[l1]
y = df['prognosis']
X_test = tr[l1]
y_test = tr['prognosis']

def train_models():
    models = {
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier(),
        "Naive Bayes": GaussianNB()
    }
    trained_models = {}
    for model_name, model_obj in models.items():
        model_obj.fit(X, y)
        acc = accuracy_score(y_test, model_obj.predict(X_test))
        trained_models[model_name] = (model_obj, acc)
    return trained_models

trained_models = train_models()

disease_to_professional = {
    'Fungal infection': ["Dermatologist", "Family Doctor"],
    'Allergy': ["Allergist", "Family Doctor"],
    'GERD': ["Gastroenterologist", "Family Doctor"],
    'Chronic cholestasis': ["Gastroenterologist", "Family Doctor"],
    'Drug Reaction': ["Dermatologist", "Family Doctor"],
    'Peptic ulcer disease': ["Gastroenterologist", "Family Doctor"],
    'AIDS': ["Infectious Disease Specialist", "Family Doctor"],
    'Diabetes ': ["Endocrinologist", "Family Doctor"],
    'Gastroenteritis': ["Gastroenterologist", "Family Doctor"],
    'Bronchial Asthma': ["Pulmonologist", "Family Doctor"],
    'Hypertension ': ["Cardiologist", "Family Doctor"],
    'Migraine': ["Neurologist", "Family Doctor"],
    'Cervical spondylosis': ["Orthopedist", "Family Doctor"],
    'Paralysis (brain hemorrhage)': ["Neurologist", "Family Doctor"],
    'Jaundice': ["Hepatologist", "Family Doctor"],
    'Malaria': ["Infectious Disease Specialist", "Family Doctor"],
    'Chicken pox': ["Pediatrician", "Family Doctor"],
    'Dengue': ["Infectious Disease Specialist", "Family Doctor"],
    'Typhoid': ["Infectious Disease Specialist", "Family Doctor"],
    'hepatitis A': ["Hepatologist", "Family Doctor"],
    'Hepatitis B': ["Hepatologist", "Family Doctor"],
    'Hepatitis C': ["Hepatologist", "Family Doctor"],
    'Hepatitis D': ["Hepatologist", "Family Doctor"],
    'Hepatitis E': ["Hepatologist", "Family Doctor"],
    'Alcoholic hepatitis': ["Hepatologist", "Family Doctor"],
    'Tuberculosis': ["Pulmonologist", "Family Doctor"],
    'Common Cold': ["General Practitioner"],
    'Pneumonia': ["Pulmonologist", "Family Doctor"],
    'Dimorphic hemorrhoids(piles)': ["Gastroenterologist", "Family Doctor"],
    'Heart attack': ["Cardiologist"],
    'Varicose veins': ["Vascular Surgeon", "Family Doctor"],
    'Hypothyroidism': ["Endocrinologist", "Family Doctor"],
    'Hyperthyroidism': ["Endocrinologist", "Family Doctor"],
    'Hypoglycemia': ["Endocrinologist", "Family Doctor"],
    'Osteoarthritis': ["Orthopedist", "Family Doctor"],
    'Arthritis': ["Rheumatologist", "Family Doctor"],
    '(vertigo) Paroxysmal Positional Vertigo': ["Neurologist", "Family Doctor"],
    'Acne': ["Cosmetic Dermatologist", "Family Doctor"],
    'Urinary tract infection': ["Urologist", "Family Doctor"],
    'Psoriasis': ["Dermatologist", "Family Doctor"],
    'Impetigo': ["Dermatologist", "Family Doctor"]
}
def disease_predictor(symptoms):
    results = []
    
    for model_name, (model, acc) in trained_models.items():
        disease = predict_disease(model, symptoms)
        pro = disease_to_professional.get(disease, ["No Recommendations Available"])
        
        # Convert the list of recommended professionals into a string without commas
        pro_str = " and ".join(pro) if isinstance(pro, list) else pro
        
        results.append(f"Model: {model_name}\nPredicted Disease: {disease}\nRecommended Professionals: {pro_str}\n")
    
    return "\n".join(results)


    
def predict_disease_button(symptom1, symptom2, symptom3, symptom4, symptom5):
    # Filter out "None" values and pass selected symptoms
    selected_symptoms = [s for s in [symptom1, symptom2, symptom3, symptom4, symptom5] if s != "None"]
    
    # Check if at least 3 symptoms are selected
    if len(selected_symptoms) < 3:
        return "Please select at least three symptoms."
    else:
        return disease_predictor(selected_symptoms)

# Gradio App 
with gr.Blocks() as app:

    with gr.Tab("Well-Being Companion"):
        gr.Markdown("<h1>🌟 Well-Being Companion</h1><p>Track your health, mood, and more!</p>")
        
        with gr.Row():
            user_input = gr.Textbox(label="Describe Your Current Feeling or Concern:", placeholder="How are you feeling today?")
            location = gr.Textbox(label="Location", placeholder="e.g., New York")
            query = gr.Textbox(label="Search for Professionals or Services", placeholder="e.g., therapist, dietitian.")
        
        with gr.Row():
            sentiment_btn = gr.Button("Analyze Sentiment")
            sentiment_result = gr.Textbox(label="Sentiment Analysis")
            emotion_btn = gr.Button("Detect Emotion")
            emotion_result = gr.Textbox(label="Emotion Detection")

        # Set up click functionality
        sentiment_btn.click(analyze_sentiment, inputs=user_input, outputs=sentiment_result)
        emotion_btn.click(detect_emotion, inputs=user_input, outputs=emotion_result)
        
        gr.Markdown("### Suggestions", elem_id="suggestions-title")

        # Table to display suggestions
        suggestions_table = gr.DataFrame(headers=["Title", "Link"])

        # New 'Get Suggestion' button
        with gr.Row():
            suggestion_btn = gr.Button("Get Suggestion")
            suggestion_btn.click(generate_suggestions, inputs=emotion_result, outputs=suggestions_table)

        with gr.Row():
            nearby_btn = gr.Button("Find Nearby Professionals")
            professionals_output = gr.Textbox(label="Professionals")

        nearby_btn.click(get_health_professionals_and_map, inputs=[location, query], outputs=professionals_output)

    with gr.Tab("Chat History"):
        gr.Markdown("<h3>Chat History:</h3>")
        chat_history = gr.Textbox(label="Chat Logs", placeholder="Conversation history will appear here.")

    with gr.Tab("Chronic Disease Prediction"):
        gr.Markdown("<h1>🩺 Chronic Disease Prediction</h1><p>Enter your symptoms to get a prediction.</p>")

        symptom1 = gr.Dropdown(["None"] + l1, label="Symptom 1")
        symptom2 = gr.Dropdown(["None"] + l1, label="Symptom 2")
        symptom3 = gr.Dropdown(["None"] + l1, label="Symptom 3")
        symptom4 = gr.Dropdown(["None"] + l1, label="Symptom 4")
        symptom5 = gr.Dropdown(["None"] + l1, label="Symptom 5")

        predict_button = gr.Button("Predict Disease")
        prediction_result = gr.Textbox(label="Prediction Result")

        predict_button.click(
            fn=predict_disease_button,
            inputs=[symptom1, symptom2, symptom3, symptom4, symptom5],
            outputs=prediction_result
        )

# Launch the app
app.launch()

    
