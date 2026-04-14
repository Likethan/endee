# 🔍 Semantic Search Web App

A simple and efficient **Semantic Search Engine** built using **Flask** and **TF-IDF (Machine Learning)**.  
This app allows users to search for information based on **meaning**, not just exact keywords.

---

## 🚀 Features

- 🔎 Search based on **semantic similarity**
- ⚡ Fast and lightweight
- 🌐 Simple web interface using Flask
- 📄 No separate frontend files (HTML embedded)
- 🧠 Uses TF-IDF and Cosine Similarity

---

## 🛠️ Tech Stack

- Python 🐍
- Flask 🌐
- Scikit-learn 🤖

---

## 📦 Installation

1. Clone the repository:

```bash
git clone https://github.com/your-username/semantic-search-app.git
cd semantic-search-app




Projext Structure 

semantic-search-app/
│── app.py
│── requirements.txt
│── README.md
│
├── templates/
│   └── index.html
│
└── static/
    └── style.css


Liscence 


---

# 💡 Small tip (important)
Before uploading:
- Replace `your-username` with your GitHub username  
- Add your name in **Author section**

---

If you want, I can also:
- Make a **GitHub description + tags**
- Create a **cool project banner**
- Or upgrade this into a **resume-level project**

Just tell me 👍


Step 1: Document Storage
A predefined list of documents is stored inside the application. These documents act as the dataset from which results are retrieved.

Step 2: Text Vectorization (TF-IDF)
All documents are converted into numerical vectors using TF-IDF (Term Frequency – Inverse Document Frequency).
In this process, important words get higher weight, while common words get lower weight. Each sentence is transformed into a numerical representation.

Step 3: User Query Input
The user enters a search query through the web interface, such as "repair phone".

Step 4: Query Vectorization
The user’s query is also converted into a vector using the same TF-IDF model so that it can be compared with the document vectors.

Step 5: Similarity Calculation
The system calculates cosine similarity between the query vector and all document vectors.
The similarity score ranges from 0 to 1, where a higher score indicates a more relevant match.

Step 6: Ranking Results
All documents are ranked based on their similarity scores in descending order, so the most relevant results appear first.

Step 7: Display Results
The top matching results are displayed to the user in the web interface.