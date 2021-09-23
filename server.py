from flask import Flask, render_template, request, jsonify
import pandas as pd
from chatbot import Chatbot

chatbot = Chatbot()
dataframe = pd.read_csv("./tables/products.csv")
print(dataframe.head())
app = Flask(__name__)

def log_query(query):
    # open the file to append the query
    with open("./user_logs/user_logs.txt", 'a') as logs_file:
        logs_file.write(query+"\n")

@app.route('/')
def home():
    # render the chatbot page
    return render_template("chatbot.html")

@app.route('/query')
def query():
    message = request.args.get("query")
    print(message)
    data = []

    # query the chatbot generating a response
    products = chatbot.query(message)

    # log the query for future training
    log_query(message)

    # return a response to the user
    return jsonify(products[0])

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

if __name__ == '__main__':
    app.run(debug=True)