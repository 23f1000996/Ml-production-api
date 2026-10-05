# Iris AI — ML Production Website

Interactive Iris flower classifier built with scikit-learn and FastAPI.

Open `/` for the user-facing prediction website. The API also exposes `/health` and `/predict`.

Example `/predict` body:
```json
{"features":[5.1,3.5,1.4,0.2]}
```

Deploy directly to Render using the included Dockerfile.
