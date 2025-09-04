# Filtering the Noise

*TikTok - ML for Trustworthy Location Reviews*

## Introduction

This project, part of the TikTok "Filtering the Noise" hackathon, challenges participants to use Machine Learning (ML) and Natural Language Processing (NLP) to automatically evaluate Google location reviews. The system aims to:
- Detect spam, advertisements, irrelevant content, and rants from users who likely never visited the location.
- Assess whether reviews are genuinely related to the location.
- Enforce policies by flagging or filtering reviews that are promotional, off-topic, or unverified complaints.

By improving the reliability of review platforms, this project enhances user experience and helps businesses maintain a trustworthy online reputation.

## Tech Stack

### Backend & API
- **FastAPI** - High-performance web framework for APIs
- **Uvicorn** - ASGI server for serving FastAPI applications
- **Pydantic** - Data validation and settings management
- **SQLAlchemy** - Database ORM for CockroachDB integration

### Machine Learning & NLP
- **Transformers (Hugging Face)** - State-of-the-art transformer models
- **Sentence Transformers** - Semantic similarity and embeddings
- **PyTorch** - Deep learning framework
- **Qwen3-Embedding-0.6B** - Primary text embedding model
- **MobileNetV2** - Image classification model

### Database
- **CockroachDB** - Distributed SQL database for review data storage

### Data Processing
- **Pandas** - Data manipulation and analysis
- **NumPy** - Numerical computing

## Data Used

- Complete review data for Alabama from [Google Local Dataset](https://mcauleylab.ucsd.edu/public_datasets/gdrive/googlelocal/)
- Stored in CockroachDB for efficient querying and processing

## API Endpoints

### POST /analyze-review
Analyzes a single review through the ML pipeline.

**Request Body:**
```json
{
  "review_cleaned": "string",
  "business_description": "string (optional)",
  "category": ["string"] (optional),
  "quality_rating": "integer"
}
```

**Response:**
```json
{
  "is_advertisement": "boolean",
  "ad_confidence": "float",
  "is_rant": "boolean", 
  "rant_confidence": "float",
  "is_relevant": "boolean",
  "relevancy_score": "float",
  "quality_rating": "integer",
  "should_filter": "boolean",
  "reason": "string",
  "final_verdict": "string (APPROVE/FILTER)"
}
```

### POST /analyze-reviews-batch
Analyzes multiple reviews in batch through the ML pipeline for efficient bulk processing.

**Request Body:**
```json
{
  "reviews": [
    {
      "review_cleaned": "string",
      "business_description": "string (optional)",
      "category": ["string"] (optional),
      "quality_rating": "integer"
    }
  ]
}
```

**Response:**
```json
{
  "results": [
    {
      "is_advertisement": "boolean",
      "ad_confidence": "float",
      "is_rant": "boolean",
      "rant_confidence": "float", 
      "is_relevant": "boolean",
      "relevancy_score": "float",
      "quality_rating": "integer",
      "should_filter": "boolean",
      "reason": "string",
      "final_verdict": "string (APPROVE/FILTER)"
    }
  ],
  "total_processed": "integer",
  "total_filtered": "integer",
  "processing_summary": {
    "approved": "integer",
    "filtered": "integer",
    "approval_rate": "float",
    "filtering_reasons": "object",
    "average_ad_confidence": "float",
    "average_relevancy_score": "float"
  }
}
```

### GET /
Health check endpoint returning service information.

**Response:**
```json
{
  "message": "[DayOne] Filtering The Noise"
}
```

## System Architecture

The pipeline consists of transformer-based ML layers:

### Core Analysis Layers
1. **Advertisement Detection** - Identifies promotional and spam content using pattern matching and semantic analysis
2. **Relevance Analysis** - Verifies review relevance to business/location using embeddings
3. **Sentiment Analysis** - Detects unconstructive rants and emotional content
4. **LLM Scoring** - Evaluates review usefulness and quality
5. **Image Classification** - Analyzes review images for relevance (MobileNetV2)

### Pipeline Features
- **Transformer-only Models** - Uses state-of-the-art transformer architectures
- **Batch Processing** - Efficient processing of large datasets through database integration
- **Comprehensive Scoring** - Multi-dimensional quality assessment
- **CSV Export** - Complete results with all tags and scores
- **REST API** - FastAPI endpoints for real-time analysis

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd filtering-the-noise

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your database credentials
```

## Usage

### 1. API Server
Start the FastAPI server:

```bash
# Run the API server
python app/main.py

# Or use uvicorn directly
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`

### 2. Batch Processing Pipeline
Process reviews from the database in batches:

```bash
# Process all reviews
python run.py

# Process with custom settings
python run.py --batch-size 50 --limit 1000 --output-file my_results.csv

# Include null/empty reviews in processing
python run.py --include-null-reviews --limit 500
```

## Pipeline Arguments

- `--config-file`: Configuration file path (default: `.env`)
- `--output-file`: Output CSV filename (default: `processed_reviews.csv`)
- `--batch-size`: Processing batch size (default: 100)
- `--limit`: Limit reviews for testing (optional)
- `--include-null-reviews`: Include reviews with null/empty text (default: false)

## Output Format

The CSV output includes:
- **Original Data**: review text, rating, timestamp, business info
- **Advertisement Analysis**: `is_advertisement`, `ad_confidence`
- **Sentiment Analysis**: `sentiment`, `is_rant`, `rant_score`
- **Relevance Analysis**: `is_relevant`, `relevancy_score`
- **Quality Scoring**: `quality_rating`, `helpfulness_score`, `usefulness_score`
- **Final Decision**: `should_filter`, `filter_reasons`, `final_verdict`

## Testing

```bash
# Test individual components
python scripts/test_advertisement_detection.py

# Test image classification
python scripts/test_image_classification.py

# Process sample data
python run.py --limit 100

# Test single review API endpoint
curl -X POST "http://localhost:8000/analyze-review" \
     -H "Content-Type: application/json" \
     -d '{
           "review_cleaned": "This restaurant has amazing food and great service.",
           "business_description": "Italian restaurant",
           "category": ["restaurant", "italian"],
           "quality_rating": 4
         }'

# Test batch processing API endpoint
curl -X POST "http://localhost:8000/analyze-reviews-batch" \
     -H "Content-Type: application/json" \
     -d '{
           "reviews": [
             {
               "review_cleaned": "Great food and excellent service!",
               "business_description": "Italian restaurant",
               "category": ["restaurant"],
               "quality_rating": 5
             },
             {
               "review_cleaned": "Call now for special discounts! Visit our website!",
               "business_description": "Electronics store", 
               "category": ["store"],
               "quality_rating": 1
             }
           ]
         }'
```

## Model Information

- **Text Embeddings**: Qwen/Qwen3-Embedding-0.6B
- **Advertisement Detection**: Pattern matching + semantic similarity
- **Sentiment Analysis**: cardiffnlp/twitter-roberta-base-sentiment-latest
- **Image Classification**: google/mobilenet_v2_1.0_224
- **Spam Detection**: mshenoda/roberta-spam

