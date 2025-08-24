# Analyze Text Sentiment
By writing the text to the Machine learning model will predict the text sentiment and visualize the impact of each word.<br/>
I used **Flask** to build the web application and **Scikit-learn** to create the model, and I used [Sentiment140 1.6M tweets](https://www.kaggle.com/datasets/kazanova/sentiment140) dataset.

[Kaggle notebook](https://www.kaggle.com/code/abdullahsaeedwebdev/sentiment140-tweets-ml-model-acc-80-8) contain the process of createing the model and how to get the negative words and positive words.

### [Live demo](https://analyze-text-sentiment.onrender.com/)

## Project Structure
 - `main.py` Main file, contain the process for preprocessing the data and load the estimators.
 - `templates/index.html` The main page that the user face.
 - `estimators/vectorizer.joblib` Is *sklearn.feature_extraction.text.TfidfVectorizer* transformer, to prepare the data before feed it to the model.
 - `estimators/model.joblib` Is *sklearn.linear_model.LogisticRegression* model. 

## Installation
Clone the project:<br/>
`git clone https://github.com/Abdullah-Saeed-BB/analyze-text-Sentiment.git`

Run the `main.py`, and navigate to localhost:5000, and that's it :)

## Screenshots
<img width="1920" height="1080" alt="‏‏لقطة الشاشة (13)" src="https://github.com/user-attachments/assets/378c26bb-bd32-4da1-b051-b1b53d7b641d" />
<img width="1920" height="1080" alt="‏‏لقطة الشاشة (12)" src="https://github.com/user-attachments/assets/9e842688-c400-4941-890c-fcba899bd03c" />
<img width="1920" height="1080" alt="‏‏لقطة الشاشة (10)" src="https://github.com/user-attachments/assets/e9526d33-52d0-4580-953c-07e532df7c3d" />
