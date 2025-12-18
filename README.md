# Filtering the Noise

**ML for Trustworthy Location Reviews**

A standalone machine learning pipeline and API to automatically evaluate location reviews. It detects spam, advertisements, rants, and irrelevant content while scoring review quality.

## Tech Stack
- FastAPI
- Hugging Face Transformers
- Sentence Transformers
- Pandas
- NumPy

## Installation

```bash
git clone <repository-url>
cd filtering-the-noise
pip install -r requirements.txt
```

## Usage

### 1. API Server
Start the API server for real-time analysis:
```bash
uvicorn app:app --reload
```
The API documentation is available at `http://localhost:8000/docs`.

### 2. Batch Processing
Process a CSV file of reviews:
```bash
python run.py --input-file data/your_reviews.csv --output-file results.csv
```

## Testing
Run the included test suite:
```bash
# Test API endpoints (server must be running)
python tests/test_api_endpoints.py

# Test individual components
python tests/test_advertisement_detection.py
python tests/test_spam_detection.py
python tests/test_quality_scoring.py
```

## API Endpoints
- `POST /analyze-review`: Analyze a single review.
- `POST /analyze-reviews-batch`: Analyze multiple reviews.
- `GET /`: Health check.

## Models
- **Embeddings**: Qwen/Qwen3-Embedding-0.6B
- **Spam Detection**: mshenoda/roberta-spam
- **Sentiment**: cardiffnlp/twitter-roberta-base-sentiment-latest
- **Image Classification**: google/mobilenet_v2_1.0_224
