from flask import Flask, request, render_template_string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Sample data (you can expand this)
documents = [
    "How to fix a broken phone screen",
    "Best ways to lose weight fast",
    "Learn Python programming step by step",
    "How to repair a cracked mobile display",
    "Top healthy diet plans",
    "Python coding tutorials for beginners",
    "AI and machine learning basics",
    "How to build web applications using Flask",
    "Data science and analytics introduction"
]

# Convert documents into vectors
vectorizer = TfidfVectorizer()
doc_vectors = vectorizer.fit_transform(documents)

# HTML Template (embedded)
HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Semantic Search App</title>
    <style>
        body { font-family: Arial; text-align: center; margin-top: 50px; }
        input { width: 300px; padding: 10px; }
        button { padding: 10px 20px; }
        .result { margin-top: 20px; }
    </style>
</head>
<body>
    <h1>🔍 Semantic Search Engine</h1>
    <form method="POST">
        <input type="text" name="query" placeholder="Enter your search..." required>
        <button type="submit">Search</button>
    </form>

    {% if results %}
    <div class="result">
        <h2>Top Results:</h2>
        <ul>
            {% for r in results %}
                <li>{{r}}</li>
            {% endfor %}
        </ul>
    </div>
    {% endif %}
</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def home():
    results = []
    
    if request.method == "POST":
        query = request.form["query"]
        
        # Convert query into vector
        query_vec = vectorizer.transform([query])
        
        # Compute similarity
        similarity = cosine_similarity(query_vec, doc_vectors)[0]
        
        # Sort results
        ranked = sorted(enumerate(similarity), key=lambda x: x[1], reverse=True)
        
        # Get top 3
        results = [documents[i] for i, _ in ranked[:3]]
    
    return render_template_string(HTML, results=results)

if __name__ == "__main__":
    print("🚀 Starting Semantic Search Server on http://127.0.0.1:5000")
    print("📱 Open your browser to test it!")
    print("Press Ctrl+C to stop")
    app.run(debug=True, host='127.0.0.1', port=5000)
