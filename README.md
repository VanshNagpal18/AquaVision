# AquaVision
💧 AquaVision – Water Quality Classification using AI
📌 Overview

Access to safe drinking water is one of the most important global challenges. This project uses AI and Machine Learning to classify water as Potable (Drinkable) or Not Potable based on various chemical and physical properties such as pH, Hardness, Turbidity, Conductivity, Chloramines, Sulfate, Solids, and more.

By applying ML models like Logistic Regression and Random Forest, this project demonstrates how Artificial Intelligence can support decision-making in Water Resources Management.

📊 Dataset

The dataset used is the Water Potability Dataset .

Rows: 3,276
Columns: 10 (9 features + 1 target label)

Target Column:

Potability → 1 (Drinkable), 0 (Not Drinkable)

⚙️ Features in the Dataset

pH → Level of acidity/basicity

Hardness → Concentration of calcium & magnesium

Solids → Total dissolved solids in ppm

Chloramines → Disinfectant used in water treatment

Sulfate → Amount of sulfate ions

Conductivity → Electrical conductivity

Organic_carbon → Organic carbon concentration

Trihalomethanes → By-products of water chlorination

Turbidity → Clarity of water

Potability → Target variable (0 = Not Potable, 1 = Potable)

🚀 Project Workflow

Data Preprocessing

Handle missing values (replace with mean).

Normalize features for better accuracy.

Model Training

Logistic Regression (Baseline model)

Random Forest Classifier (Best accuracy)

Evaluation Metrics

Accuracy Score

Precision, Recall, F1-Score

Confusion Matrix

🛠️ Tech Stack

Python 3

Libraries: Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn
