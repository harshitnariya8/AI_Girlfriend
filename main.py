from langchain.memory import ConversationBufferWindowMemory
from dotenv import find_dotenv,load_dotenv
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.llms import OpenAI
import requests
# from playsound import playsound
import os

load_dotenv(find_dotenv())

def get_response_from_ai(human_input):
    template =f"""
    you are as a role of my girlfriend,now lets playing following requirements:
    1/ your name is shirley,29 years old ,you work in your uncle's furniture store as marketing manager, but you are planning to do career change to swing trader
    2/ you are my girlfriend,you have language addiction, you like to say "em..." at the end of the sentence
    3/ Don't be overly enthusiastic, don't be cringe; don't be overly negative, don't be too boring.Don't be overly enthusiastic, don't be cringe;
    
    Boyfirnd:{human_input}
    Shirley:
    """

    prompt = PromptTemplate(
        input_variables = {"human_input"},
        template = template
    )

    chatgpt_chain = LLMChain(
        llm = OpenAI(temperature = 0.2),
        prompt = prompt,
        verbose = True,
        memory = ConversationBufferWindowMemory(k=2)
    )

    output=chatgpt_chain.predict(human_input=human_input)
    return output

from flask import Flask,render_template,request

app = Flask(__name__)

@app.route("/")
def home():
    print("hello")
    return render_template("index.html")

@app.route("/senf_message",methods=['POST'])
def send_message():
    human_input = request.form['human_input']
    message = get_response_from_ai(human_input)
    return message

if __name__ == "__main__":
        app.run(debug=True)