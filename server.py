from flask import Flask, make_response, request, jsonify
import os
import numpy
from tensorflow.keras.models import load_model

app = Flask(__name__)

#load the model
cnnModel = load_model('cnnModel.keras')

def predicting(features):
    classes = ['bacterial_leaf_blight', 'bacterial_leaf_streak', 
               'bacterial_panicle_blight', 'blast', 'brown_spot', 
               'dead_heart', 'downy_mildew', 'hispa', 'normal', 'tungro']
    imgArr = numpy.array(features)
    prediction = cnnModel.predict(imgArr)[0]
    index = numpy.argmax(prediction)
    confidence = int(prediction[index]*100)
    print(prediction)
    print(index)
    disease = classes[index]
    return confidence, disease

@app.route('/riceDiseasePrediction', methods = ['POST'])
def getFeatures():
    if request.method == "POST":
        data = request.get_json()
        features = data['image']
        if len(features) != 0:
            confidence, disease = predicting(features)
            response = jsonify({'confidence': confidence, 'disease': disease})
            return response
        else:
            return f"Input doesn't meet prerequisite, Input length is {len(features)}"

if __name__ == '__main__':
    app.run(host = '0.0.0.0', port = 105)
