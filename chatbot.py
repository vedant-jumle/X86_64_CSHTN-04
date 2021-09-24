import numpy as np
import tensorflow as tf
import pandas as pd
import pickle
import json
from tensorflow import keras
import random
import re
import spacy
import nltk
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

def check_presence(df, col, item):
    bool_list = []
    item = clear_query(item)
    for entry in df[col]:
        bool_list.append(str(item).lower() in str(entry).lower())
    return bool_list

def clear_query(token):
        stop_words_with_numbers = [
            ("gb", ""),
            ("k", "000"),
            ("hr", ""),
            ("wh", ""),
            ("watts", ""),
            ("hrs", ""),
            ("watthr", "")
        ]
        stop_words_without_numbers = [
            ("core", ""),
            ("quad", '4'),
            ("hexa", "6"),
            ("octa", "8"),
            (" ", ""),
            (".", ""),
            (",", ""),
            ("-", ""),
            ("working", "work")
        ]

        # remove non-numbered stopwords
        for word, replacement in stop_words_without_numbers:
            token = token.replace(word, replacement).strip()
        
        # remove numbered stopwords
        for word, replacement in stop_words_with_numbers:
            if len(re.findall(r"\d+" + word, token)) > 0:
                token = token.replace(word, replacement).strip()
        
        return lemmatizer.lemmatize(token)

class Chatbot:
    def __init__(self, intent_bot_path="./models/bot_V1.model", ner_bot_path="./models/spacy_V2/model-best", words_path="words.pkl", classes_path="classes.pkl", intents_path="intents.json"):
        print("loading models")
        self.intent_classifier = keras.models.load_model(intent_bot_path)
        self.nlp = spacy.load(ner_bot_path)
        self.words = pickle.load(open(words_path, "rb"))
        self.classes = pickle.load(open(classes_path, "rb"))
        self.intents = json.load(open(intents_path, "r"))
        self.lemmatizer = WordNetLemmatizer()
        self.ner_labels = [
            "MANUFACTURER",
            "MODEL",
            "CPU_BRAND",
            "CPU_SERIES",
            "CPU_MODEL",
            "CORE_COUNT",
            "CPU_GEN",
            "RAM",
            "GPU_BRAND",
            "GPU",
            "GPU_TECH",
            "GPU_SERIES",
            "VRAM",
            "SCREEN_SIZE",
            "TYPE",
            "PRICE",
            "STORAGE_TYPE",
            "STORAGE_SIZE",
            "BAT_CAP",
            "MISC",
            "PRICE_TYPE",
            "SCREEN_RESO",
            "IO",
            "OS"
        ]
        self.dataframe = pd.read_csv("./tables/products.csv")
        print("models loaded")
    
    def preprocess(self, item):
        tokens = nltk.word_tokenize(item)
        tokens = [self.lemmatizer.lemmatize(word) for word in tokens]
        return tokens

    def bag_of_words(self, sentence):
        sentence_words = self.preprocess(sentence)
        bag = [0] * len(self.words)
        for w in sentence_words:
            for i, word in enumerate(self.words):
                if word == w:
                    bag[i] = 1
        
        return np.array(bag)

    def predict_intent(self, sentence, tolerance=0.7):
        bow = self.bag_of_words(sentence)
        prediction = self.intent_classifier(np.array([bow]))[0]
        max_value = max(prediction).numpy()
        res = np.where(prediction.numpy() == max_value)[0][0]

        selected_intent = self.intents["intents"][res]

        message = selected_intent["responses"][random.randint(0, len(selected_intent["responses"])-1)]
        if max_value >= tolerance:
            return {
                "tag" : selected_intent["tag"],
                "message" : message
            }
        else:
            selected_intent = self.intents["intents"][4]
            return {
                "tag" : "abigious",
                "message" : selected_intent["responses"][random.randint(0, len(selected_intent["responses"])-1)]
            }

    def ner(self, sentence):
        doc = self.nlp(sentence)
        output = {}

        for label in self.ner_labels:
            output[label] = []

        for ent in doc.ents:
            output[ent.label_].append(ent.text)
        
        return output

    def query(self, sentence):
        intent = self.predict_intent(sentence)
        product_list = []

        if intent["tag"] == "query":
            labels = self.ner(sentence)
            df = self.dataframe.copy(deep=True)
            response = pd.DataFrame(columns=df.columns)
            
            for label in self.ner_labels:
                if len(labels[label]) > 0:
                    check = df if not len(response) > 0 else response
                    empty = pd.DataFrame(columns=df.columns)
                    for item in labels[label]:
                        temp = check.loc[check[label].apply(str).apply(str.lower).apply(clear_query).str.contains(clear_query(item.lower()))] 
                        empty = empty.append(temp)
                    response = empty.copy(deep=True)

            product_list = [response.iloc[i].map(str).to_dict() for i in range(len(response))]

        return {
            "tag" : intent["tag"],
            "message" : intent["message"],
            "product_list" : product_list
        }
    
    