# Iris ML Production API

A beginner-friendly machine learning deployment project built for the
**Getting Started with ML in Production Workshop**.

The project demonstrates the complete loop:

**Train → Export → Load → Serve → Test → Deploy**

## What does the model predict?

This project uses the classic Iris dataset and a scikit-learn
Random Forest classifier.

The model predicts one of three Iris flower species:

- `0` → setosa
- `1` → versicolor
- `2` → virginica

The four input features must be supplied in this order:

1. sepal length
2. sepal width
3. petal length
4. petal width

## Project files

```text
.
├── train.py
├── main.py
├── model.pkl
├── requirements.txt
├── Dockerfile
├── render.yaml
├── .python-version
├── .gitignore
└── README.md
```

## Run locally

### 1. Create a virtual environment

macOS/Linux:

```bash
python3 -m venv venv
source venv/bin/activate
```

Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Train/export the model

```bash
python train.py
```

This creates:

```text
model.pkl
```

### 4. Start FastAPI

```bash
uvicorn main:app --reload
```

Open:

```text
http://127.0.0.1:8000/docs
```

## API endpoints

### GET `/health`

Example response:

```json
{
  "status": "ok",
  "model_loaded": true
}
```

### POST `/predict`

#### Example request body

```json
{
  "features": [5.1, 3.5, 1.4, 0.2]
}
```

#### Example response

```json
{
  "prediction": 0,
  "class_name": "setosa"
}
```

## Deploy on Render

Use a Render **Web Service**.

Build command:

```bash
pip install -r requirements.txt
```

Start command:

```bash
uvicorn main:app --host 0.0.0.0 --port $PORT
```

After deployment, test:

```text
https://YOUR-SERVICE.onrender.com/health
```

and:

```text
https://YOUR-SERVICE.onrender.com/docs
```

## Docker

Docker is included as an optional workshop-aligned deployment method.

Build:

```bash
docker build -t iris-ml-api .
```

Run:

```bash
docker run -p 8000:8000 iris-ml-api
```

Then open:

```text
http://127.0.0.1:8000/docs
```
