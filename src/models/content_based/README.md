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

##  Findings: Classic KNN vs. Modified KNN in our dataset
In our dataset, genres are highly imbalanced (e.g., many performances belong to the same category such as pop music). 

When using classic KNN, this leads to overspecialized recommendations:

- Performances with similar genre-heavy features are repeatedly recommended
- For example, a pop music performance generates only other pop music recommendations
- Similarity scores stay very high, but diversity is lost

### Benefits of Modified KNN
- Get recommendations with slightly lower similarity scores, but introduces greater diversity across metadata
- Reduces bias toward dominant genres in the dataset 
- Avoids overspecialization and improves user discovery

### Example
For a user who liked the performance :


|       | Performance data                | 
|-------|---------------------------------|
| name  | 리도어: Tree Opening Day *Snow     | 
| venue | 성동구 연무장길 28-16 (구.타임애프터타임) (1층) | 
| cast  | 이상민, 최승현, 주상욱, 박세웅              | 
| age   | 만 7세 이상                         | 
| area  | 서울특별시                           | 
| genre | 대중음악                            | 
 
the classic KNN would recommend :

| Performance Name                          | Genre    | 
|-------------------------------------------|----------|
| NAM WOO HYUN CONCERT, 식목일5: TREE HIGH SCHOOL | 대중음악     |
| 서울재즈빅밴드, 최성수와 이정식 두 뮤지션의 40년의 음악여정        | 대중음악     | 
| 7080콘서트 [사천]                              | 대중음악     | 
| 플랜비프로젝트 콘써어트 X 탐정케이                       | 대중음악     |
| 청춘고백 [용인]                                 | 대중음악     |

And the modified KNN would suggest :

| Performance Name                          | Genre    | 
|-------------------------------------------|----------|
| 푸에르자 부르타: 아벤 [서울] | 뮤지컬     | 
| 더 나은 휴머니티 [대학로]        | 연극     | 
| PANDA FIRE                             | 대중음악     | 
| 喜怒愛楽                     | 대중음악     | 
| 호야;好夜                                 | 연극     | 


### Conclusion

A hybrid KNN (classic + modified ) strategy can provide a better balance by:

- Preserving high-similarity recommendations that match clear user interests
- Introducing lower-similarity but relevant items to increase diversity and discovery
- Avoiding genre bias while still respecting user taste

---

## References
- Cortez, Dan Michael & Cordero, Nathan & Canlas, Jermaine & Mata, Khatalyn & Regala, Richard & Blanco, Mark Christopher & Alipio, Antolin. (2022). Modified Content-Based Filtering Method Using K-Nearest Neighbors and Percentile Concept. International Journal of Research Publications. 100. 10.47119/IJRP1001001520223119. 
- Thorat, P. B., Goudar, R. M., & Barve, S. (2015). Survey on collaborative filtering, content-based filtering and hybrid recommendation
system. International Journal of Computer Applications, 110(4), 31-36.
- Sharma, L., & Gera, A. (2013). A survey of recommendation system: Research challenges. International Journal of Engineering
Trends and Technology (IJETT), 4(5), 1989-1992
- Badriyah, T., Azvy, S., Yuwono, W., & Syarif, I. (2018, March). Recommendation system for property search using content based
filtering method. In 2018 International conference on information and communications technology (ICOIACT) (pp. 25-29). IEEE.