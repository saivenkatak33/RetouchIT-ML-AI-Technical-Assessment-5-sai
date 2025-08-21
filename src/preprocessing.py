
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def build_preprocessor():
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), ['amount', 'oldbalanceOrg', 'newbalanceOrig',
                                       'oldbalanceDest', 'newbalanceDest']),
            ('cat', OneHotEncoder(handle_unknown='ignore'), ['type'])
        ]
    )
    return preprocessor
