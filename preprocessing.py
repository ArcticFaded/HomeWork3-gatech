from sklearn.preprocessing import MinMaxScaler
from functools import reduce
import pandas as pd

def preprocess_bank_data(bank):
    bank = bank[bank.education != 'unknown']
    bank = bank[bank.job != 'unknown']

    job_categories = pd.get_dummies(bank.job)
    marital_status = pd.get_dummies(bank.marital)
    education_categories = pd.get_dummies(bank.education)
    outcome_categories = pd.get_dummies(bank.poutcome)
    contact_categories = pd.get_dummies(bank.contact)
    month_categories = pd.get_dummies(bank.month)

    def toCategorical(x):
        return 1 if x == 'yes' else 0

    processed_csv = [job_categories, 
                     marital_status, 
                     education_categories,
                     outcome_categories,
                     contact_categories,
                     month_categories,
                     pd.DataFrame(bank.housing.transform(lambda x : toCategorical(x))),
                     pd.DataFrame(bank.default.transform(lambda x : toCategorical(x))),
                     pd.DataFrame(bank.loan.transform(lambda x: toCategorical(x))),
                     bank[['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']],
                     pd.DataFrame(bank.y.transform(lambda x: toCategorical(x)))
                    ]
    df_final = reduce(lambda left,right: pd.merge(left,right, left_index=True, right_index=True), processed_csv)

    norm_columns = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']
    scalers = {name: MinMaxScaler() for name in norm_columns}

    for names in norm_columns:
        df_final[names] = scalers[names].fit_transform(df_final[[names]])
        
    
    balanced_df = df_final[df_final.y == 1]
    negative_examples = df_final[df_final.y == 0].sample(len(balanced_df))

    df_final = pd.concat([balanced_df, negative_examples])
    
    return df_final, scalers

def preprocess_heart_data(heart):
    chest_pains = pd.get_dummies(heart.cp, prefix='cp')
    slopes = pd.get_dummies(heart.slope, prefix='slope')
    major_vessels = pd.get_dummies(heart.ca, prefix='ca')
    thals = pd.get_dummies(heart.thal, prefix='thals')

    processed_csv = [
        chest_pains,
        slopes,
        major_vessels,
        thals,
        heart[['age', 'sex', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak']],
        pd.DataFrame(heart.target)
    ]

    df_final = reduce(lambda left, right: pd.merge(left, right, left_index=True, right_index=True), processed_csv)

    # scaling
    norm_columns = ['age', 'trestbps', 'chol', 'restecg', 'thalach', 'oldpeak']

    scalers = {name: MinMaxScaler() for name in norm_columns}

    for names in norm_columns:
        df_final[names] = scalers[names].fit_transform(df_final[[names]])
        
    return df_final, scalers