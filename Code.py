from flask import Flask, request, jsonify
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Sample Course Data
courses = pd.read_csv("courses.csv")
vectorizer = TfidfVectorizer(stop_words='english')
course_matrix = vectorizer.fit_transform(courses['description'])

@app.route('/recommend', methods=['POST'])
def recommend():
    user_input = request.json['preferences']
    user_vector = vectorizer.transform([user_input])
    similarities = cosine_similarity(user_vector, course_matrix).flatten()
    recommended = courses.iloc[similarities.argsort()[-5:][::-1]]
    return jsonify(recommended[['title', 'description']].to_dict(orient='records'))

if __name__ == '__main__':
    app.run(debug=True)
