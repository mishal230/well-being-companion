import os
import gradio as gr
import nltk
import numpy as np
import tflearn
import random
import json
import pickle
from nltk.tokenize import word_tokenize
from nltk.stem.lancaster import LancasterStemmer
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import googlemaps
import folium
import torch

# Suppress TensorFlow warnings
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Download necessary NLTK resources
nltk.download("punkt")
stemmer = LancasterStemmer()

# Load intents and chatbot training data
with open("intents.json") as file:
    intents_data = json.load(file)

with open("data.pickle", "rb") as f:
    words, labels, training, output = pickle.load(f)

# Build the chatbot model
net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)
chatbot_model = tflearn.DNN(net)
chatbot_model.load("MentalHealthChatBotmodel.tflearn")

# Hugging Face sentiment and emotion models
tokenizer_sentiment = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
model_sentiment = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
tokenizer_emotion = AutoTokenizer.from_pretrained("j-hartmann/emotion-english-distilroberta-base")
model_emotion = AutoModelForSequenceClassification.from_pretrained("j-hartmann/emotion-english-distilroberta-base")

# Google Maps API Client
gmaps = googlemaps.Client(key=os.getenv("GOOGLE_API_KEY"))

# Helper Functions
def bag_of_words(s, words):
    """Convert user input to bag-of-words vector."""
    bag = [0] * len(words)
    s_words = word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words if word.isalnum()]
    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1
    return np.array(bag)

def generate_chatbot_response(message, history):
    """Generate chatbot response and maintain conversation history."""
    history = history or []
    try:
        result = chatbot_model.predict([bag_of_words(message, words)])
        tag = labels[np.argmax(result)]
        response = "I'm sorry, I didn't understand that. 🤔"
        for intent in intents_data["intents"]:
            if intent["tag"] == tag:
                response = random.choice(intent["responses"])
                break
    except Exception as e:
        response = f"Error: {e}"
    history.append((message, response))
    return history, response

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
    return emotion_map.get(emotion, "Unknown 🤔"), emotion

def generate_suggestions(emotion):
    """Return relevant suggestions based on detected emotions."""
    emotion_key = emotion.lower()
    suggestions = {
        "joy": [
            ["Relaxation Techniques", '<a href="https://www.helpguide.org/mental-health/meditation/mindful-breathing-meditation" target="_blank">Visit</a>'],
            ["Dealing with Stress", '<a href="https://www.helpguide.org/mental-health/anxiety/tips-for-dealing-with-anxiety" target="_blank">Visit</a>'],
            ["Emotional Wellness Toolkit", '<a href="https://www.nih.gov/health-information/emotional-wellness-toolkit" target="_blank">Visit</a>'],
            ["Relaxation Video", '<a href="https://youtu.be/m1vaUGtyo-A" target="_blank">Watch</a>'],
        ],
        "anger": [
            ["Emotional Wellness Toolkit", '<a href="https://www.nih.gov/health-information/emotional-wellness-toolkit" target="_blank">Visit</a>'],
            ["Stress Management Tips", '<a href="https://www.health.harvard.edu/health-a-to-z" target="_blank">Visit</a>'],
            ["Dealing with Anger", '<a href="https://www.helpguide.org/mental-health/anxiety/tips-for-dealing-with-anxiety" target="_blank">Visit</a>'],
            ["Relaxation Video", '<a href="https://youtu.be/MIc299Flibs" target="_blank">Watch</a>'],
        ],
        "fear": [
            ["Mindfulness Practices", '<a href="https://www.helpguide.org/mental-health/meditation/mindful-breathing-meditation" target="_blank">Visit</a>'],
            ["Coping with Anxiety", '<a href="https://www.helpguide.org/mental-health/anxiety/tips-for-dealing-with-anxiety" target="_blank">Visit</a>'],
            ["Emotional Wellness Toolkit", '<a href="https://www.nih.gov/health-information/emotional-wellness-toolkit" target="_blank">Visit</a>'],
            ["Relaxation Video", '<a href="https://youtu.be/yGKKz185M5o" target="_blank">Watch</a>'],
        ],
        "sadness": [
            ["Emotional Wellness Toolkit", '<a href="https://www.nih.gov/health-information/emotional-wellness-toolkit" target="_blank">Visit</a>'],
            ["Dealing with Anxiety", '<a href="https://www.helpguide.org/mental-health/anxiety/tips-for-dealing-with-anxiety" target="_blank">Visit</a>'],
            ["Relaxation Video", '<a href="https://youtu.be/-e-4Kx5px_I" target="_blank">Watch</a>'],
        ],
        "surprise": [
            ["Managing Stress", '<a href="https://www.health.harvard.edu/health-a-to-z" target="_blank">Visit</a>'],
            ["Coping Strategies", '<a href="https://www.helpguide.org/mental-health/anxiety/tips-for-dealing-with-anxiety" target="_blank">Visit</a>'],
            ["Relaxation Video", '<a href="https://youtu.be/m1vaUGtyo-A" target="_blank">Watch</a>'],
        ],
    }
    return suggestions.get(emotion_key, [["No specific suggestions available.", ""]])

def get_health_professionals_and_map(location, query):
    """Search nearby healthcare professionals using Google Maps API."""
    try:
        if not location or not query:
            return ["Please provide both location and query."], ""

        geo_location = gmaps.geocode(location)
        if geo_location:
            lat, lng = geo_location[0]["geometry"]["location"].values()
            places_result = gmaps.places_nearby(location=(lat, lng), radius=10000, keyword=query)["results"]
            professionals = []
            map_ = folium.Map(location=(lat, lng), zoom_start=13)
            for place in places_result:
                professionals.append(f"{place['name']} - {place.get('vicinity', 'No address provided')}")
                folium.Marker(
                    location=[place["geometry"]["location"]["lat"], place["geometry"]["location"]["lng"]],
                    popup=f"{place['name']}"
                ).add_to(map_)
            return professionals, map_._repr_html_()

        return ["No professionals found for the given location."], ""
    except Exception as e:
        return [f"An error occurred: {e}"], ""

# Main Application Logic
def app_function(user_input, location, query, history):
    chatbot_history, _ = generate_chatbot_response(user_input, history)
    sentiment_result = analyze_sentiment(user_input)
    emotion_result, cleaned_emotion = detect_emotion(user_input)
    suggestions = generate_suggestions(cleaned_emotion)
    professionals, map_html = get_health_professionals_and_map(location, query)
    return chatbot_history, sentiment_result, emotion_result, suggestions, professionals, map_html

# CSS Styling
custom_css = """
body {
    font-family: 'Roboto', sans-serif;
    background: linear-gradient(135deg, #0d0d0d, #ff5722);
    color: white;
}

h1 {
    background: #ffffff;
    color: #000000;
    border-radius: 8px;
    padding: 10px;
    font-weight: bold;
    text-align: center;
    font-size: 2.5rem;
}

textarea, input {
    background: transparent;
    color: black;
    border: 2px solid orange;
    padding: 8px;
    font-size: 1rem;
    caret-color: black;
    outline: none;
    border-radius: 8px;
}

textarea:focus, input:focus {
    background: transparent;
    color: black;
    border: 2px solid orange;
    outline: none;
}

textarea:hover, input:hover {
    background: transparent;
    color: black;
    border: 2px solid orange;
}

.df-container {
    background: white;
    color: black;
    border: 2px solid orange;
    border-radius: 10px;
    padding: 10px;
    font-size: 14px;
    max-height: 400px;
    height: auto;
    overflow-y: auto;
}

#suggestions-title {
    text-align: center;
    font-weight: bold;
    color: white;
    font-size: 2.2rem;
    margin-bottom: 20px;
}
"""

# Gradio Application
with gr.Blocks(css=custom_css) as app:
    gr.HTML("<h1>🌟 Well-Being Companion</h1>")
    with gr.Row():
        user_input = gr.Textbox(label="Your Message")
        location = gr.Textbox(label="Your Location")
        query = gr.Textbox(label="Search Query")
    chatbot = gr.Chatbot(label="Chat History")
    sentiment = gr.Textbox(label="Detected Sentiment")
    emotion = gr.Textbox(label="Detected Emotion")
    
    # Adding Suggestions Title with Styled Markdown (Centered and Bold)
    gr.Markdown("Suggestions", elem_id="suggestions-title")
    
    suggestions = gr.DataFrame(headers=["Title", "Link"])  # Table for suggestions
    professionals = gr.Textbox(label="Nearby Professionals", lines=6)
    map_html = gr.HTML(label="Interactive Map")
    submit = gr.Button(value="Submit", variant="primary")

    submit.click(
        app_function,
        inputs=[user_input, location, query, chatbot],
        outputs=[chatbot, sentiment, emotion, suggestions, professionals, map_html],
    )

app.launch()
