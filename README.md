# 🎬 Movie Recommender System (Collaborative Filtering)

An interactive movie recommendation web app built with **Python** and **Streamlit**, using collaborative filtering on the **MovieLens** dataset.  
Users can rate a few movies and instantly get personalized recommendations based on user similarity or a trained matrix-factorization model.

🔗 **Live Demo:** [movie-recommender-cf.streamlit.app](https://movie-recommender-cf-js9vo4rx9vv6cmhmf5pacw.streamlit.app/#rate-a-few-movies-to-get-started)

---

## 🚀 Features

- **Cold-Start Interface** – users rate a few movies (random or popular titles)  
- **Dynamic Refresh** – swap individual movies or refresh the entire list  
- **Two Recommendation Modes**  
  - *Quick (Cosine Similarity)* – lightweight demo mode  
  - *Trained Model (SVD)* – matrix factorization using the Surprise library  
- **Interactive UI** built in Streamlit  
- Based on the **MovieLens 100K** dataset

---

## 🧠 Tech Stack

- **Python 3.10+**
- **Streamlit**
- **pandas**, **NumPy**
- **scikit-learn**
- **Surprise (SVD algorithm)**

---

## ⚙️ Setup & Run Locally

```bash
git clone https://github.com/amritkundan/movie-recommender-cf.git
cd movie-recommender-cf
pip install -r requirements.txt
streamlit run streamlit_app.py
