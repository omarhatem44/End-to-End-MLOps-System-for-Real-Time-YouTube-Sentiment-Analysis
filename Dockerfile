FROM python:3.10

WORKDIR /app

COPY . .

RUN pip install -r requirements.txt

RUN python3 - <<EOF
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
EOF

CMD ["python3","flask_api/app.py"]