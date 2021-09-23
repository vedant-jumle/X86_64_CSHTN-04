from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, send, emit
import pandas as pd
from chatbot import Chatbot

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app)

@app.route("/")
def load_page():
    return render_template("sockets_test.html")

@socketio.on("connect")
def connection():
    print("Connected")

@socketio.on("query")
def handle_query(data):
    print(data)

    socketio.emit("response", data)

if __name__ == '__main__':
    socketio.run(app)