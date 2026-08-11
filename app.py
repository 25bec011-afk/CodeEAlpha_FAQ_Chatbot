from flask import Flask, render_template, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

app = Flask(__name__)

# FAQ knowledge base: replace/add entries for any topic or product.
FAQS = [
    {
        "question": "How do I reset my password?",
        "answer": "Go to the login page, select 'Forgot Password', enter your registered email, and follow the password-reset instructions.",
        "category": "Account"
    },
    {
        "question": "How do I create an account?",
        "answer": "Click 'Create Account', enter your name, email, and a secure password, then complete the verification step.",
        "category": "Account"
    },
    {
        "question": "I forgot my username. What should I do?",
        "answer": "Use the account-recovery option on the login page or contact support with the email address linked to your account.",
        "category": "Account"
    },
    {
        "question": "How can I update my profile?",
        "answer": "Open your profile settings, edit the information you want to change, and select Save Changes.",
        "category": "Account"
    },
    {
        "question": "How do I contact customer support?",
        "answer": "Use the Contact Support option in the application. Include a short description of your issue so the support team can help you faster.",
        "category": "Support"
    },
    {
        "question": "What are the support hours?",
        "answer": "Support hours are Monday to Friday, 9:00 AM to 6:00 PM. You can still submit a support request outside these hours.",
        "category": "Support"
    },
    {
        "question": "How do I report a problem?",
        "answer": "Open Help & Support, choose Report a Problem, describe the issue, and submit the request.",
        "category": "Support"
    },
    {
        "question": "How do I change my email address?",
        "answer": "Go to Profile Settings, select Email Address, enter the new email, and complete the verification sent to it.",
        "category": "Account"
    },
    {
        "question": "How can I cancel my subscription?",
        "answer": "Open Billing Settings, select your active subscription, choose Cancel Subscription, and confirm the cancellation.",
        "category": "Billing"
    },
    {
        "question": "How do I update my payment method?",
        "answer": "Go to Billing Settings and select Payment Methods. Add your new payment method and set it as the default.",
        "category": "Billing"
    },
    {
        "question": "Where can I see my invoices?",
        "answer": "Open Billing Settings and select Invoices. Your available invoices can be viewed or downloaded there.",
        "category": "Billing"
    },
    {
        "question": "Can I get a refund?",
        "answer": "Refund eligibility depends on the purchase and applicable policy. Contact support with your order details to request a review.",
        "category": "Billing"
    },
    {
        "question": "How do I download my data?",
        "answer": "Go to Privacy Settings and select Download My Data. Follow the instructions to request an export.",
        "category": "Privacy"
    },
    {
        "question": "How do I delete my account?",
        "answer": "Open Account Settings, choose Delete Account, read the confirmation information, and confirm the request.",
        "category": "Account"
    },
    {
        "question": "Is my data secure?",
        "answer": "The application is designed to protect account information using standard security practices. Avoid sharing passwords or sensitive credentials.",
        "category": "Security"
    },
    {
        "question": "How do I enable two factor authentication?",
        "answer": "Open Security Settings, select Two-Factor Authentication, and follow the setup instructions for your authenticator or verification method.",
        "category": "Security"
    },
    {
        "question": "Why am I not receiving verification emails?",
        "answer": "Check your spam or junk folder, confirm that your email address is correct, and request another verification email if needed.",
        "category": "Account"
    },
    {
        "question": "What browsers are supported?",
        "answer": "Use a recent version of Chrome, Edge, Firefox, or Safari for the best experience.",
        "category": "General"
    },
    {
        "question": "Does the chatbot understand different ways of asking a question?",
        "answer": "Yes. The chatbot uses TF-IDF vectorization and cosine similarity to compare the meaning-related word patterns of your question with the FAQ questions.",
        "category": "Chatbot"
    },
    {
        "question": "How does this FAQ chatbot work?",
        "answer": "Your question is cleaned during preprocessing, converted into TF-IDF vectors, compared with the FAQ dataset using cosine similarity, and the closest matching answer is returned when the confidence is high enough.",
        "category": "Chatbot"
    }
]

STOP_WORDS = "english"
CONFIDENCE_THRESHOLD = 0.18

def preprocess(text):
    """Basic NLP preprocessing: lowercase, remove URLs/punctuation/numbers and normalize whitespace."""
    text = text.lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

questions = [preprocess(item["question"]) for item in FAQS]

vectorizer = TfidfVectorizer(
    stop_words=STOP_WORDS,
    ngram_range=(1, 2),
    sublinear_tf=True
)
faq_matrix = vectorizer.fit_transform(questions)

def find_best_answer(user_question):
    cleaned = preprocess(user_question)

    if not cleaned:
        return None, 0.0

    # Handle greetings
    greetings = {
        "hi", "hello", "hey", "hii", "hiii",
        "good morning", "good afternoon", "good evening"
    }

    if cleaned in greetings:
        return {
            "question": "Greeting",
            "answer": "Hello! 👋 I'm FAQBot. How can I help you?",
            "category": "General"
        }, 1.0

    user_vector = vectorizer.transform([cleaned])
    similarities = cosine_similarity(user_vector, faq_matrix)[0]

    best_index = similarities.argmax()
    score = float(similarities[best_index])

    if score < CONFIDENCE_THRESHOLD:
        return None, score

    return FAQS[best_index], score

@app.route("/")
def home():
    return render_template("index.html", faqs=FAQS)

@app.route("/api/ask", methods=["POST"])
def ask():
    data = request.get_json(silent=True) or {}
    question = str(data.get("question", "")).strip()

    if not question:
        return jsonify({
            "success": False,
            "message": "Please enter a question."
        }), 400

    match, score = find_best_answer(question)

    if match is None:
        return jsonify({
            "success": True,
            "matched": False,
            "confidence": round(score * 100, 1),
            "answer": "I couldn't find a confident FAQ match. Please rephrase your question or contact support."
        })

    return jsonify({
        "success": True,
        "matched": True,
        "confidence": round(score * 100, 1),
        "question": match["question"],
        "answer": match["answer"],
        "category": match["category"]
    })

@app.route("/api/faqs")
def faqs():
    return jsonify(FAQS)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
