import numpy as np
import pickle

pipeline = pickle.load(open('./model/model.pkl', 'rb'))

example = {
    'credit_usage': 12.0,
    'cc_age': 25.0,
    'credit_history_critical/other existing credit': 0,
    'credit_history_delayed previously': 0,
    'credit_history_existing paid': 1,
    'credit_history_no credits/all paid': 0,
    'over_draft_less_than0': 0,
    'over_draft_greater_than200': 0,
    'over_draft_no checking': 0,
    'Average_Credit_Balance_500less_than=Xless_than1000': 0,
    'Average_Credit_Balance_less_than100': 1,
    'Average_Credit_Balance_greater_than1000': 0,
    'Average_Credit_Balance_no known savings': 0
}

def make_prediction(features):
    X = np.array([features['credit_usage'],
                  features['cc_age'],
                  int(features['credit_history_critical/other existing credit']=='EC'),
                  int(features['credit_history_delayed previously']=='DP'),
                  int(features['credit_history_existing paid']=='EP'),
                  int(features['credit_history_no credits/all paid']=='NP'),
                  int(features['over_draft_less_than0']=='LZ'),
                  int(features['over_draft_greater_than200']=='G2'),
                  int(features['over_draft_no checking']=='NA'),
                  int(features['Average_Credit_Balance_500less_than=Xless_than1000']=='G5'),
                  int(features['Average_Credit_Balance_less_than100']=='L1'),
                  int(features['Average_Credit_Balance_greater_than1000']=='G10'),
                  int(features['Average_Credit_Balance_no known savings']=='NS')
                 ]).reshape(1,-1)
    prob_survived = pipeline.predict_proba(X)[0, 1]

    result = {
        'prediction': int(prob_survived > 0.5),
        'prob_survived': prob_survived
    }
    return result

if __name__ == '__main__':
    print(make_prediction(example))
