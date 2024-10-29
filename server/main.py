from flask import Flask, request, jsonify
from flask_cors import CORS

import pandas as pd
import numpy as np
import os


import key_param
from QA import chain as chain
from QA import QA_chain

os.environ["OPENAI_API_KEY"] = key_param.OPENAI_API_KEY

app = Flask(__name__)
CORS(app)


@app.route('/api/qa', methods=['POST'])
def process_message():
    
    data = request.get_json()
    message = data.get('message', '')

    answer = QA_chain(message)
    
    processed_message = answer.replace('\n', "<br>")
    processed_message = processed_message.replace('#', "")
    
    return jsonify({'response': processed_message})

if __name__ == '__main__':
    app.run(debug=True)