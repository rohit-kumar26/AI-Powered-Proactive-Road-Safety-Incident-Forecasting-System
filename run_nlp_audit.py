import pandas as pd
import sys
sys.path.append('src')
from nlp_utils import check_risk_keywords

# Load data
df = pd.read_csv('data/raw/synthetic_accident_data_v1.csv')
print(f'Loaded {len(df)} records.')

# Apply NLP Audit
print('Running NLP Audit on descriptions...')
df['NLP_Risk_Flag'] = df['Incident_Description'].apply(check_risk_keywords)

# Correction Logic
df['Corrected_Severity'] = df.apply(
    lambda row: 1 if (row['Reported_Severity'] == 0 and row['NLP_Risk_Flag'] == 1) 
    else row['Reported_Severity'], 
    axis=1
)

# Results
misclassified_count = df[df['Reported_Severity'] != df['Corrected_Severity']].shape[0]
print(f'NLP Audit Complete!')
print(f'Found and Fixed {misclassified_count} mislabeled accidents.')
print(f'Reported Severity Distribution: {df["Reported_Severity"].value_counts().to_dict()}')
print(f'Corrected Severity Distribution: {df["Corrected_Severity"].value_counts().to_dict()}')

# Save
import os
os.makedirs('data/processed', exist_ok=True)
final_df = df.drop(columns=['Actual_Severity', 'Incident_Description'])
final_df.to_csv('data/processed/training_data_cleaned.csv', index=False)
print('Cleaned data saved for modeling.')
