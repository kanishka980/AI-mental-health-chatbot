from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import nltk
from nltk.stem import WordNetLemmatizer
import pickle
import json
import random

app = Flask(__name__)


nltk.download('popular')
nltk.download('punkt')
lemmatizer = WordNetLemmatizer()


BOTS = {
    "sage": {
        "model_path": "C:\\Users\\bhavy\\PycharmProjects\\FlaskProject4\\models\\sage\\model.keras",
        "intents_path": "C:\\Users\\bhavy\\PycharmProjects\\FlaskProject4\\data\\sagefile.json",
        "words_path": "models/sage/texts.pkl",
        "classes_path": "models/sage/labels.pkl",
    },
    "captain": {
        "model_path": "C:\\Users\\bhavy\\PycharmProjects\\FlaskProject4\\models\\captain\\capmodel.keras",
        "intents_path": "C:\\Users\\bhavy\\PycharmProjects\\FlaskProject4\\data\\captainfile.json",
        "words_path": "models/captain/texts.pkl",
        "classes_path": "models/captain/labels.pkl",
    },
    "friend": {
        "model_path": "C:\\Users\\bhavy\\PycharmProjects\\FlaskProject4\\models\\friend\\friendmodel.keras",
        "intents_path": "C:\\Users\\bhavy\\PycharmProjects\\FlaskProject4\\data\\friendfile.json",
        "words_path": "models/friend/texts.pkl ",
        "classes_path": "models/friend/labels.pkl",
    },
    "mentor": {
        "model_path": "C:\\Users\\bhavy\\PycharmProjects\\FlaskProject4\\models\\mentor\\mentormodel.keras",
        "intents_path": "C:\\Users\\bhavy\\PycharmProjects\\FlaskProject4\\data\\mentorfile.json",
        "words_path": "models/mentor/texts.pkl",
        "classes_path": "models/mentor/labels.pkl",
    },
}


bots_data = {}
for bot_name, bot_config in BOTS.items():

    with open(bot_config["intents_path"]) as file:
        intents = json.load(file)


    words = pickle.load(open(bot_config["words_path"], "rb"))
    classes = pickle.load(open(bot_config["classes_path"], "rb"))


    model = tf.keras.models.load_model(bot_config["model_path"])


    bots_data[bot_name] = {
        "intents": intents,
        "words": words,
        "classes": classes,
        "model": model,
    }


def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words


def bow(sentence, words, show_details=True):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
                if show_details:
                    print("found in bag: %s" % w)
    return np.array(bag)


def predict_class(sentence, model, words, classes):
    p = bow(sentence, words, show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list


def getResponse(ints, intents_json):
    if ints:

        tag = ints[0]['intent']

        list_of_intents = intents_json['intents']
        for i in list_of_intents:
            if i['tag'] == tag:

                result = random.choice(i['responses'])
                break
        return result
    else:
        return "Sorry, I didn't understand that."


# Routes
@app.route('/')
def home():
    return render_template('index.html')


@app.route('/chat')
def chat():
    return render_template('chat.html')


@app.route('/chat/<bot>')
def chat_with_bot(bot):
    if bot in bots_data:
        return render_template('chatting.html', bot=bot, get_tagline=get_tagline)
    else:
        return "Bot not found", 404


def get_tagline(bot):
    taglines = {
        "sage": "Find wisdom and inner peace.",
        "captain": "Navigate your emotions with strength.",
        "friend": "A listening ear, always here.",
        "mentor": "Guidance for a brighter path.",
    }
    return taglines.get(bot, "Let's talk!")


@app.route('/chat/<bot>/message', methods=['POST'])
def chat_message(bot):
    try:
        if bot not in bots_data:
            return jsonify({"response": "Bot not found"}), 404

        data = request.get_json()
        user_message = data.get("message", "")

        if not user_message:
            return jsonify({"response": "Please enter a message!"})


        print("User Message Received:", user_message)

        # Get bot-specific data
        bot_data = bots_data[bot]
        model = bot_data["model"]
        words = bot_data["words"]
        classes = bot_data["classes"]
        intents = bot_data["intents"]

        ints = predict_class(user_message, model, words, classes)

        response_text = getResponse(ints, intents)
        print("Predicted Response:", response_text)

        return jsonify({"response": response_text})

    except Exception as e:
        print("Error:", str(e))
        return jsonify({"response": "An error occurred on the server."}), 500

if __name__ == '__main__':
    app.run(port=8000, debug=True)