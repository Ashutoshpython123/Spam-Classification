from flask import Flask
from flask import request
from flask import render_template

import numpy as np
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer


from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.layers import Embedding
from tensorflow.keras.models import load_model


app = Flask(__name__)



def pred_method(z):
    model = None
    nltk.download('stopwords')
    ps = PorterStemmer()

    def process_data(test_sen):
        test_sen = test_sen.lower()
        test_sen = test_sen.split()
        test_sen = [ps.stem(word) for word in test_sen if not word in stopwords.words('english')]
        test_sen = ' '.join(test_sen)
        return test_sen

    test_sen = z
    processed = process_data(test_sen)
    oh = one_hot(processed,1000)
    sent_len = 20
    embedded_docs = pad_sequences([oh], padding='pre', maxlen=sent_len)
    X = np.array(embedded_docs)
    
    model = load_model('spam_classification.h5')
    pred = model.predict_classes(X)[0][0]                                                                                                                                                                                                 

    return pred

@app.route('/')
def hello():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        msg = request.form['message']
        prediction = pred_method(msg)
    return render_template('result.html', prediction = prediction)


if __name__ == '__main__':
    app.run(debug=True)
    