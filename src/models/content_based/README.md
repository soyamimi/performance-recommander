# Modified Content-Based Filtering Experiment

This folder contains an experiment implementing the **Modified Content-Based Filtering (CBF) Method Using K-Nearest Neighbors and Percentile Concept** as described in:  
**Cortez, Dan Michael & Cordero, Nathan & Canlas, Jermaine & Mata, Khatalyn & Regala, Richard & Blanco, Mark Christopher & Alipio, Antolin. (2022). Modified Content-Based Filtering Method Using K-Nearest Neighbors and Percentile Concept. International Journal of Research Publications. 100. 10.47119/IJRP1001001520223119.**  

---

## Table of Contents
- [Introduction](#introduction)  
- [Motivation](#motivation)
- [Method Overview](#method-overview)
- [References](#references)  

---

## Introduction

**Content-Based Filtering (CBF)** is a recommendation system approach that suggests items to users based on:  

- Their preference profile  
- Interactions with items in the system  
- Item descriptions

For example, if a user likes a particular performance, the system recommends performances with **similar genres, title, or descriptions**.  

A common problem in CBF is **overspecialization**:  
> The system repeatedly recommends items that are too similar to what the user has already liked, limiting discovery of new items.

---

## Motivation

The goal of this experiment is to test a **modified CBF method** that:  

1. Uses **K-Nearest Neighbors** to find similar items.  
2. Introduces the **percentile concept** to prevent overspecialization.  

By only recommending items within a specified percentile range of similarity (e.g., 60th–80th percentile), we can:  

- Avoid recommending items that are **too similar**  
- Increase diversity in recommendations  
- Maintain relevance to user preferences  

This experiment helps **validate whether percentile-based selection improves recommendation quality**.

---

## Method Overview

1. **Extract Item Attributes**  
   - Transform item attributes into numerical feature vectors.

2. **Cosine Similarity Matrix**  
   - Compute the **cosine similarity** between all item vectors.

3. **Collect User’s Liked Item**  
   - Identify the item(s) that the user has liked or interacted with.

4. **Create the "Trunk List"**  
   - Arrange all items in **descending order of cosine similarity** with respect to the user’s liked item.

5. **Apply K-Nearest Neighbors (KNN)**  
   - For each of the **top 5 items in the trunk list**, find **3 nearest neighbors** using KNN.

6. **Apply Percentile Concept**  
   - In the trunk list for item X, determine the **60th percentile** and **80th percentile** in the cosine similarity scores.  
   - Recommend items whose similarity score falls between the 60th and 80th percentile. 
   - In this experimentation, we use the range **between the 70th and 90th percentile**.
   - This step ensures **diversity** and prevents **overspecialization**.
---

## References
- Cortez, Dan Michael & Cordero, Nathan & Canlas, Jermaine & Mata, Khatalyn & Regala, Richard & Blanco, Mark Christopher & Alipio, Antolin. (2022). Modified Content-Based Filtering Method Using K-Nearest Neighbors and Percentile Concept. International Journal of Research Publications. 100. 10.47119/IJRP1001001520223119. 
- Thorat, P. B., Goudar, R. M., & Barve, S. (2015). Survey on collaborative filtering, content-based filtering and hybrid recommendation
system. International Journal of Computer Applications, 110(4), 31-36.
- Sharma, L., & Gera, A. (2013). A survey of recommendation system: Research challenges. International Journal of Engineering
Trends and Technology (IJETT), 4(5), 1989-1992
- Badriyah, T., Azvy, S., Yuwono, W., & Syarif, I. (2018, March). Recommendation system for property search using content based
filtering method. In 2018 International conference on information and communications technology (ICOIACT) (pp. 25-29). IEEE.