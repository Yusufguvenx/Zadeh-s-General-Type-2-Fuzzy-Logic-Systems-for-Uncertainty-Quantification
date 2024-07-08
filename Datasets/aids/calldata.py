from ucimlrepo import fetch_ucirepo 
import pandas as pd
  
# fetch dataset 
aids_clinical_trials_group_study_175 = fetch_ucirepo(id=890) 
  
# data (as pandas dataframes) 
X = aids_clinical_trials_group_study_175.data.features 
y = aids_clinical_trials_group_study_175.data.targets 
  
# metadata 
print(aids_clinical_trials_group_study_175.metadata) 
  
# variable information 
print(aids_clinical_trials_group_study_175.variables) 


df = pd.DataFrame(X)

file_path = "x.csv"

df.to_csv(file_path, index=False)


df = pd.DataFrame(y)

file_path = "y.csv"

df.to_csv(file_path, index=False)