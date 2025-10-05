# Preprocessing

The `preprocessing` module is responsible for **cleaning, 
transforming, and encoding** raw performance data 
before it is passed into recommendation models.  

All preprocessing logic currently lives inside `transform_data.py`.

---

## Purpose of `transform_data.py`

This script handles the full preprocessing pipeline, including:

### 1. Text Cleaning
Functions like `clean_text()` remove:
- Special characters
- Bracketed content (`()`, `{}`, `[]`)
- Redundant whitespace
- Trailing symbols or markers

### 2. Age Extraction
The function extract_min_age() converts values like:
- "전체 관람가" → 0
- "만 7세 이상" → 7
- "24개월 이상" → 2

### 3. Preprocessing Pipeline
The function create_preprocessing_pipeline() creates a scikit-learn Pipeline with:

| Feature Type   | Transformer Used   |
|----------------|--------------------|
| Categorical    | `OneHotEncoder`    |
| Numeric        | `MinMaxScaler`     | 
| Text           | `TfidfVectorizer`  |
