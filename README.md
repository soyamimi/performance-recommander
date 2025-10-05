# 🎭 Performance Recommender 🎭 
Performance Recommender is a recommendation system experimentation 
that suggests performances (e.g., shows, plays, musicals, concerts) 
using data collected from the **KOPIS API**.

The project explores and tests multiple recommendation models, 
includes a full data preprocessing pipeline, 
and is organized with clean modular architecture. 

Configuration and secrets are managed with **Dynaconf**, 
and dependencies are handled via **pyproject.toml**.

---

## What This Project Does
This system is designed to:

- Fetch performance-related data from the KOPIS public API
- Preprocess and structure the collected data
- Train and evaluate multiple recommendation models
- Provide clean, maintainable architecture for scalability
- Securely manage configurations and secrets with Dynaconf

The project can be extended for:

- Test different types of recommendation systems
- Integration with backend services or apps
- Building API endpoints on top of the models

---

## Recommendation Models

The recommendation logic is modular, with different approaches under `src/models/`.

| Model Type    | Path                        | Direct Link                                      |
|---------------|-----------------------------|--------------------------------------------------|
| Content-Based | `src/models/content_based/` | [Content-Based Model](src/models/content_based/) |
| Collaborative | `src/models/collaborative/` | [Collaborative Model](src/models/collaborative/) |
| Hybrid        | `src/models/hybrid/`        | [Hybrid Model](src/models/hybrid/)               |

---

## Notebooks

The experimentation notebooks are located in `notebooks/`.