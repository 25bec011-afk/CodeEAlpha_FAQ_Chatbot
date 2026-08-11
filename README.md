# 🤖 CodeAlpha FAQ Chatbot

## 🌐 Live Demo

👉 [Open the Live FAQ Chatbot](https://codeealpha-faq-chatbot.onrender.com)

## 📌 CodeAlpha Artificial Intelligence Internship — Task 2

An AI-powered FAQ chatbot using NLP preprocessing, TF-IDF vectorization and cosine similarity.# 🤖 FAQBot — CodeAlpha AI Internship Task 2

A complete FAQ chatbot that uses **NLP preprocessing, TF-IDF vectorization and cosine similarity** to match user questions with the most relevant FAQ and return an answer.

## 🎯 Task requirements covered

- ✅ Collect FAQs related to a topic/product
- ✅ Preprocess user questions using NLP-style text cleaning
- ✅ Match user questions using cosine similarity
- ✅ Display the best matching answer as a chatbot response
- ✅ Responsive chat UI
- ✅ Confidence score and fallback for weak matches

## 🛠️ Tech stack

- Python
- Flask
- scikit-learn
- TF-IDF Vectorizer
- Cosine Similarity
- HTML5
- CSS3
- JavaScript

## 🧠 How the AI/NLP matching works

1. The FAQ dataset contains questions and answers.
2. Questions are converted to lowercase and cleaned of URLs, punctuation, numbers and extra whitespace.
3. `TfidfVectorizer` converts FAQ questions into numerical TF-IDF vectors.
4. The user's cleaned question is transformed using the same vectorizer.
5. `cosine_similarity()` compares the user's vector against every FAQ vector.
6. The FAQ with the highest similarity score is selected.
7. A confidence threshold prevents the chatbot from returning a random answer when the match is weak.

### Formula

Cosine similarity compares two vectors based on their angle:

`similarity(A,B) = (A · B) / (||A|| ||B||)`

## 🚀 Run locally

### 1. Create a virtual environment

Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

macOS/Linux:

```bash
python3 -m venv venv
source venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Start the chatbot

```bash
python app.py
```

Open:

```text
http://127.0.0.1:5000
```

## 📁 Project structure

```text
CodeEAlpha_FAQ_Chatbot/
├── app.py
├── requirements.txt
├── Procfile
├── README.md
├── templates/
│   └── index.html
└── static/
    └── style.css
```

## ☁️ Deployment

This Flask project can be deployed on a Python-compatible hosting platform such as Render, Railway or another service supporting Gunicorn.

Start command:

```bash
gunicorn app:app
```

A static GitHub Pages site cannot directly execute this Python backend. Use a Python hosting service if you want a public live demo.

## 🎥 Suggested demo video

Show these steps in 45–60 seconds:

1. Open FAQBot.
2. Ask: `How do I reset my password?`
3. Show the matched answer and confidence score.
4. Ask the same concept in different words.
5. Try a billing/support question.
6. Show the NLP/TF-IDF/cosine similarity explanation.
7. Show the GitHub repository.

## 💼 Suggested LinkedIn caption

> 🤖 Excited to share my FAQ Chatbot developed as part of the CodeAlpha Artificial Intelligence Internship — Task 2.
>
> The chatbot uses NLP preprocessing, TF-IDF vectorization and cosine similarity to identify the most relevant FAQ and return an answer.
>
> Tech Stack: Python, Flask, scikit-learn, NLP, TF-IDF, Cosine Similarity, HTML, CSS and JavaScript.
>
> #CodeAlpha #ArtificialIntelligence #NLP #Python #MachineLearning #Internship #GitHub

## 📌 Customizing the FAQ dataset

Edit the `FAQS` list in `app.py`:

```python
{
    "question": "Your question",
    "answer": "Your answer",
    "category": "Your category"
}
```

Add as many FAQs as required.

---

**CodeAlpha AI Internship — Task 2: Chatbot for FAQs**
