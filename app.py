import numpy as np
from flask import Flask,request,jsonify,render_template
import pickle
from flask_cors import CORS

#creation of flask app
app=Flask(__name__)
CORS(app, resources={r"/*": {"origins": "http://localhost:3000"}})

#Loading pickel model
model = pickle.load(open("model.pkl","rb"))

@app.route('/', methods=['POST'])
def predict():
    data = request.json
    features=np.array([data])
    array_as_list = features.tolist()
    output1 = model.predict(array_as_list)
    output=output1.tolist()
    return jsonify(output)




if __name__ == '__main__':
    app.run(debug=True)
