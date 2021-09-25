from flask import Flask, render_template, request, jsonify, redirect
from flask_socketio import SocketIO, send, emit
import pandas as pd
import random
import json
from chatbot import Chatbot
from context import  Context

app = Flask(__name__)
dataframe = pd.read_csv("./tables/products.csv")
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app)

context_set = Context()

post_messages = json.load(open("post_messages.json", "r"))

chatbot = Chatbot()

def log_query(query):
    # open the file to append the query
    with open("./user_logs/user_logs.txt", 'a') as logs_file:
        logs_file.write(query+"\n")

@app.route("/")
def load_page():
    context_set.reset()
    return render_template("chatbot.html")

@app.route("/product")
def load_product_page():
    return render_template("product.html")

@app.route("/database/")
def database():
    # get the id value from the query string
    id = request.args.get("id")
    data = {}

    # search for the product id in the database
    item = dataframe.loc[dataframe["ID"] == int(id)]

    # create a dict out of the product table
    for column in list(item.columns):
        data[str(column)] = str(item[str(column)].values[0])
    
    # print for debugging
    print(data)

    # return a response to the server
    return jsonify(data)

@socketio.on("connect")
def connection():
    print("Connected")
    print(context_set.get_data())
    emit("response", {
        "message" : "I can help you find the most suitable laptop from our database",
        "product_list" : []
    }, json=True)

    emit("response", {
        "message" : "Please tell me what are you looking for?",
        "product_list" : []
    }, json=True)

    emit("response", {
        "message" : "You can ask me to look for certain brands, configuration or even what type of work you want to do on the device.",
        "product_list" : []
    }, json=True)

@socketio.on("query")
def handle_query(data):
    print(". ".join(context_set.get_data() + [data["data"]]))

    response = {
        "message":"",
        "product_list": []
    }

    temp_res = chatbot.predict_intent(data["data"], tolerance=0.5)

    if temp_res["tag"] == "query":
        response = chatbot.query(". ".join(context_set.get_data() + [data["data"]]))
        context_set.add_data(data["data"])
        log_query(data["data"])
        emit("response", response, json=True)

        if len(response["product_list"]) > 0:
            emit("response", {
                "message" : post_messages[random.randint(0, len(post_messages)-1)],
                "product_list" : []
            }, json=True)
        else:
            context_set.purge(data["data"])
            emit("response", {
                "message" : "Hmm.. couldn't find anything that matches what you told me, you may have made a mistake.",
                "product_list" : []
            }, json=True)
    else:
        temp_res["product_list"] = []
        emit("response", temp_res, json=True)

if __name__ == '__main__':
    socketio.run(app)