# Data Collecting

The `data_collecting` module is responsible for retrieving and generating the data required for training and evaluating the performance recommender system.

It currently contains:

- **`api_fetch.py`** – Fetches performance data from the KOPIS API  
- **`generate_user_data.py`** – Generates user preference data

---
## Purpose
### `api_fetch.py`

This script handles data collection from the **KOPIS API**.

- Authenticates using credentials stored via Dynaconf
- Sends requests to retrieve performance information
- Parses and formats response data
- Saves results as CSV

### `generate_user_data.py`
This script creates randomized user preference data, useful for:

- Generates random user IDs
- Assigns mocked preferences (e.g., genres, age, area, ratings)
- Produces data in a structured format for modeling