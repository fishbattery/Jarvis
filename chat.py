import random
import json

import torch

from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

import pyttsx3
import speech_recognition as sr

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('intents.json', 'r') as json_data:
    intents = json.load(json_data)
    

FILE = "data.pth"
data = torch.load(FILE)

def speak(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

def get_audio():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        audio = r.listen(source)
        said = ""

        try:
            said = r.recognize_google(audio)
        except Exception as e:
            print('Exception: ' + str(e))

    return said.lower()

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "Jarvis"
print("Let's chat! (type 'quit' to exit)")
while True:
    WAKE = "hey jarvis", "ok jarvis", "hi jarvis", "sup jarvis"
    sentence = get_audio()
    # sentence = "do you use credit cards?"
    for phrase in WAKE:
        sentence = get_audio()

        sentence = tokenize(sentence)
        X = bag_of_words(sentence, all_words)
        X = X.reshape(1, X.shape[0])
        X = torch.from_numpy(X).to(device)

        output = model(X)
        _, predicted = torch.max(output, dim=1)

        tag = tags[predicted.item()]

        probs = torch.softmax(output, dim=1)
        prob = probs[0][predicted.item()]
        if prob.item() > 0.75:
            for intent in intents['intents']:
                if tag == intent["tag"]:

                    response = random.choice(intent['responses'])

                    print(f"{bot_name}: {response}")
                    speak(f"{response}")
        else:
            print(f"{bot_name}: I do not understand...")
            speak(f"I do not understand")