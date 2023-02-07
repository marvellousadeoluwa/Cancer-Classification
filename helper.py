__version__ = '0.1.1' #release 1_patch 1
model_path = 'breast_cancer_model-0.1.0.pkl'

final_features = ['mean concavity',
 'mean concave points',
 'worst radius',
 'worst perimeter',
 'worst area',
 'worst concave points']

#['mean area', 'mean concavity', 'mean concave points','worst radius','worst perimeter','worst area','worst concavity', 'worst concave points']


cancer = {0: 'Benign', 1: 'Malignant'}

def predict_cancer(model, features):
    result = model.predict([features])

    return cancer[result[0]]