# RTA_deployment

### Road Traffic Accident Severity Classification

This project predicts the **severity** of road traffic accidents using real-world police records from Addis Ababa (2017–2020). The original dataset contains 12,316 accident instances and 32 features covering road conditions, driver behavior, vehicle information, and environmental factors. The goal is to identify key factors contributing to accident severity and to provide a practical tool that can assist in risk assessment and decision-making.

---

## Project Highlights

- Performed in-depth **Exploratory Data Analysis (EDA)** to understand feature distributions, detect data quality issues, and uncover patterns related to accident severity.  
- Addressed class imbalance using **SMOTE (Synthetic Minority Over-sampling Technique)** and applied comprehensive preprocessing (encoding, scaling, handling missing values) to prepare data for modeling.  
- Trained and compared multiple **machine learning classification algorithms** to identify the best-performing model for severity prediction.  
- Evaluated models using **Accuracy** and **F1-score**, focusing on robust performance across all severity classes rather than just overall accuracy.  
- Built an interactive **Streamlit web application** that allows users to input accident-related parameters and get an instant severity prediction.  
- Deployed the application on **AWS**, making the model accessible as a scalable, production-ready service.

---

## Results

- Best model: **Extra Trees Classifier**  
- Test accuracy: **92%**  
- The final model demonstrated strong generalization on the test set and handled class imbalance effectively, providing reliable severity classification.

---

## Tech Stack

- **Language:** Python  
- **ML & Data:** pandas, scikit-learn, imbalanced-learn (SMOTE), NumPy  
- **Model Serving / UI:** Streamlit  
- **Deployment:** AWS (cloud-hosted Streamlit app)

---

## What This Project Demonstrates

- End-to-end ML workflow: from data collection and cleaning to model training, evaluation, and deployment.  
- Practical application of **classification algorithms** to a real-world safety-critical problem.  
- Experience in **building and deploying** an ML-powered web app that can be integrated into decision-support systems for road safety and urban planning.
