from __future__ import print_function
import random
import json
import requests

import datetime
import time
import webbrowser
import random
from bs4 import BeautifulSoup

import tkinter as tk
from tkinter import filedialog, Text
from tkinter.constants import YES

import os
import os.path
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials 

from selenium import  webdriver

import torch

from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

import pytz
import pyttsx3
import speech_recognition as sr

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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('intents.json', 'r') as json_data:
    intents = json.load(json_data)
    

FILE = "data.pth"
data = torch.load(FILE)



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
print("talk")    

while True:
    WAKE = "hey jarvis", "ok jarvis", "hi jarvis", "sup jarvis"
    text = get_audio()
    for phrase in WAKE:
        if phrase in text:
            print(f"{bot_name}: Ready as always sir")
            speak("ready as always sir")
            text = get_audio()

            sentence = tokenize(text)
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
                        speak(response)


                        if response == "What is the city name":
                            headers = {
                            	'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}

                            def weather(city):
                                city = city.replace(" ", "+")
                                res = requests.get(
                                    f'https://www.google.com/search?q={city}&oq={city}&aqs=chrome.0.35i39l2j0l4j46j69i60.6128j1j7&sourceid=chrome&ie=UTF-8', headers=headers)
                                print("Searching...\n")
                                soup = BeautifulSoup(res.text, 'html.parser')
                                location = soup.select('#wob_loc')[0].getText().strip()
                                time = soup.select('#wob_dts')[0].getText().strip()
                                info = soup.select('#wob_dc')[0].getText().strip()
                                weather = soup.select('#wob_tm')[0].getText().strip()
                                print(location)
                                print(time)
                                print(info)
                                print(weather+"°C")


                            city = get_audio()
                            city = city + " weather"
                            print(f"{bot_name}: {city}")
                            speak(city)

            print(f"{bot_name}: I do not understand")
            speak(f"I do not understand")
