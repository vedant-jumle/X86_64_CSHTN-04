from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, send, emit
import pandas as pd
from chatbot import Chatbot

app = Flask(__name__)
dataframe = pd.read_csv("./tables/products.csv")
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app)

context_set = []

chatbot = Chatbot()

def log_query(query):
    # open the file to append the query
    with open("./user_logs/user_logs.txt", 'a') as logs_file:
        logs_file.write(query+"\n")

@app.route("/")
def load_page():
    return render_template("chatbot.html")

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
    context_set = []
    print("Connected")

@socketio.on("query")
def handle_query(data):
    print(".".join(context_set + [data["data"]]))
    response = chatbot.query(". ".join(context_set + [data["data"]]))
    if response["tag"] == "query":
        context_set.append(data["data"])
        log_query(data["data"])
    emit("response", response, json=True)

if __name__ == '__main__':
    socketio.run(app)