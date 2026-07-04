from flask import Flask, render_template, request
import joblib 
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

app = Flask(__name__)

# loading the saved model
pipeline = joblib.load("models/mnb_model.joblib")

stop_words = set(stopwords.words('english'))

# this function cleans the user input
def clean_msg(sen):
    sen = sen.lower()
    sen = re.sub(r'[^a-z0-9]', ' ', sen)
    tokens = word_tokenize(sen)
    filtered_token = [word for word in tokens if word not in stop_words]
    filtered_sen = ' '.join(filtered_token)
    return filtered_sen

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict',methods=["POST"])
def predict():
    if request.method == "POST":
        user_input = request.form['email_text']
        cleaned_input = clean_msg(user_input)
        
        result = pipeline.predict([cleaned_input])[0]

        if result == 1:
            color = "red"
            prediction = "It's a Spam Message"
        else:
            color = "green"
            prediction = "It's not a Spam Message"
    
        return render_template("index.html", original_text=user_input,  prediction= prediction, text_color=color)
       

if __name__ == "__main__":
    app.run(debug=True)