
# Make sure Postgres and Qdrant are running
```
docker run -d --name qdrant -p 6334:6334 qdrant/qdrant:latest
export COCOINDEX_DATABASE_URL="postgres://cocoindex:cocoindex@localhost/cocoindex"
```

# Setup QDrant Collection (clip-ViT-L-14 returns 768 dimensions)
```
curl -X PUT
  'http://localhost:6333/collections/image_search' \
  --header 'Content-Type: application/json' \
  --data-raw '{
    "vectors": {
      "embedding": {
        "size": 768,
        "distance": "Cosine"
      }
    }
  }'

```

# Ollama
```
ollama pull gemma3
ollama serve
```

# Put your images in ./img
Place your images in the `img` directory.

# Create virtual environment and install dependencies
```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

```

# Run Backend
```
python main.py cocoindex setup
python main.py cocoindex update
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

# Run Frontend
```
cd frontend
npm install
npm run dev
```

Go to `http://localhost:5174` to search.