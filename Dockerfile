FROM python:3.10

COPY requirements.txt requirements.txt
COPY .gitignore .gitignore
COPY /model /model
COPY /app /app

RUN python -m pip install --upgrade pip && pip install -r requirements.txt

EXPOXSE 8000

ENTRYPOINT ["gunicorn", "--bind", "0.0.0.0:8000", "wsgi:app"]