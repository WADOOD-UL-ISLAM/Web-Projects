import tkinter as tk
from tkinter import scrolledtext
import random
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model

# Load the required files and model
lemmatizer = WordNetLemmatizer()
intents = json.load(open('intents.json'))
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatbot_model.keras')

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25

    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)

    return_list = []
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})

    return return_list

def get_response(intents_list, intents_json):
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']

    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break

    return result

def send_message():
    user_message = user_input.get()
    if user_message:
        chat_log.config(state=tk.NORMAL)
        chat_log.insert(tk.END, f"You: {user_message}\n")
        
        # Get bot response
        ints = predict_class(user_message)
        response = get_response(ints, intents)
        
        chat_log.insert(tk.END, f"Bot: {response}\n")
        chat_log.config(state=tk.DISABLED)
        
        user_input.delete(0, tk.END)
        chat_log.yview(tk.END)  # Scroll to the end

# Set up the main application window
root = tk.Tk()
root.title("Chat Application")

# Create the chat log area
chat_log = scrolledtext.ScrolledText(root, state=tk.DISABLED, wrap=tk.WORD, height=20, width=50)
chat_log.grid(row=0, column=0, columnspan=2, padx=10, pady=10)

# Create the user input field
user_input = tk.Entry(root, width=40)
user_input.grid(row=1, column=0, padx=10, pady=10)

# Create the send button
send_button = tk.Button(root, text="Send", command=send_message)
send_button.grid(row=1, column=1, padx=10, pady=10)

# Run the application
root.mainloop()
