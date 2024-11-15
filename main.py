import numpy as np
import pandas as pd
import joblib
import nltk
from nltk.tokenize import word_tokenize
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Step 1: Load Simulated Data
def simulate_data():
    np.random.seed(42)
    data = pd.DataFrame({
        'soil_moisture': np.random.uniform(0.1, 1.0, 1000),
        'ice_sheet_change': np.random.uniform(-5, 5, 1000),
        'forest_cover': np.random.uniform(20, 80, 1000),
        'climate_event': np.random.choice(['drought', 'flood', 'stable'], 1000)
    })
    data['event_label'] = data['climate_event'].map({'drought': 0, 'flood': 1, 'stable': 2})
    return data

# Step 2: Train ML Model
def train_ml_model(data):
    X = data[['soil_moisture', 'ice_sheet_change', 'forest_cover']]
    y = data['event_label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    clf = RandomForestClassifier(random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print("ML Model Performance:\n", classification_report(y_test, y_pred))
    
    joblib.dump(clf, 'models/climate_model.pkl')
    return clf

# Step 3: Process Climate Reports with NLP
def process_climate_reports():
    reports = [
        "Severe droughts are likely due to decreasing soil moisture in arid regions.",
        "Flooding observed in areas with high deforestation rates.",
        "Stable conditions observed in regions with sufficient forest cover."
    ]
    nltk.download('punkt')
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(reports)
    
    report_labels = [0, 1, 2]  # Corresponding to drought, flood, and stable
    nlp_model = LogisticRegression(random_state=42)
    nlp_model.fit(tfidf_matrix, report_labels)
    
    joblib.dump(vectorizer, 'models/vectorizer.pkl')
    joblib.dump(nlp_model, 'models/nlp_model.pkl')
    return reports, vectorizer, nlp_model

# Step 4: Data Prioritization Based on Danger Level
def prioritize_data(input_data):
    # Danger score logic:
    danger_score = 0
    if input_data['soil_moisture'] < 0.3:
        danger_score += 50  # High risk if soil moisture is low (drought)
    if abs(input_data['ice_sheet_change']) > 2.0:
        danger_score += 40  # High risk if ice sheet change is significant
    if input_data['forest_cover'] < 30:
        danger_score += 30  # High risk if forest cover is low (flood risk)
    
    # If danger score is above a threshold, trigger an immediate alert
    if danger_score >= 70:
        print("ALERT: Critical Climate Event Detected! Immediate Attention Required!")
        return True  # Indicates the data needs immediate reporting
    
    return False  # No immediate alert needed

# Step 5: Integrate ML and NLP for Insights
def generate_insights(input_data):
    clf = joblib.load('models/climate_model.pkl')
    nlp_model = joblib.load('models/nlp_model.pkl')
    vectorizer = joblib.load('models/vectorizer.pkl')
    
    reports = [
        "Severe droughts are likely due to decreasing soil moisture in arid regions.",
        "Flooding observed in areas with high deforestation rates.",
        "Stable conditions observed in regions with sufficient forest cover."
    ]
    
    input_df = pd.DataFrame([input_data])
    event_pred = clf.predict(input_df)
    
    relevant_reports = []
    for report in reports:
        report_vectorized = vectorizer.transform([report])
        report_pred = nlp_model.predict(report_vectorized)
        if report_pred[0] == event_pred[0]:
            relevant_reports.append(report)
    
    event_mapping = {0: 'Drought', 1: 'Flood', 2: 'Stable'}
    insight = {
        "Predicted Event": event_mapping[event_pred[0]],
        "Supporting Insights": relevant_reports
    }
    return insight

# Main Execution
if __name__ == "__main__":
    data = simulate_data()
    train_ml_model(data)
    reports, vectorizer, nlp_model = process_climate_reports()
    
    example_input = {
        'soil_moisture': 0.2,
        'ice_sheet_change': -2.5,
        'forest_cover': 25.0
    }
    
    # Step 6: Prioritize Data and Generate Insights
    is_critical = prioritize_data(example_input)
    
    if not is_critical:
        insight = generate_insights(example_input)
        print("Generated Insight:\n", insight)
