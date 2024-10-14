import lime
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
import joblib
from tensorflow.keras.models import load_model  # For DNN and CNN models
from lime import lime_image
from lime.lime_tabular import LimeTabularExplainer


class Helper():

    def __init__(self):
        # Loading preprocessors
        self.preprocessor = joblib.load('pkl/preprocessor_pipeline.pkl')

        # Loading Models
        self.dnn_model = load_model('models/dnn_model.h5')  # Keras DNN model

        self.ada_model = joblib.load(
            'pkl/adaboost_model.pkl')  # AdaBoost model
        self.xgb_model = joblib.load(
            'pkl/xgboost_model.pkl')  # XGBoost model

        # Ensemble model combining Ada, XGBoost, RandomForest
        self.ensemble_model = joblib.load(
            'pkl/ensemble_model.pkl')

        # CNN model for image classification
        self.cnn_model = load_model('models/cnn_model.h5')
        # dnn_image_model = load_model('dnn_image_model.h5')  # DNN model for image classification

        # Training Data
        # Load X_train from a pickle file
        self.X_train = joblib.load('pkl/X_train.pkl')

        # Labels To predict
        # Labels for image classification (replace with your actual labels)
        self.class_labels = ['drink', 'food', 'inside', 'menu', 'outside']

    def preprocess_image(self, img):
        img = img.resize((128, 128))  # Resizing the image to 256x256
        img_array = img_to_array(img)
        # Expanding dimensions to match model input
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0  # Normalize to [0, 1]
        return img_array


class ModelPrediction(Helper):

    def Interpretability_for_parking(self):
        # LIME Tabular explainer for the first tab
        # Preprocess X_train
        X_train_encoded = self.preprocessor.transform(self.X_train)

        # Extract feature names from the preprocessor after transformation
        feature_names = self.preprocessor.get_feature_names_out()

        # Identify categorical features (if they were one-hot encoded in preprocessor)
        categorical_features = [i for i, col in enumerate(
            feature_names) if 'onehot' in col]

        return LimeTabularExplainer(
            X_train_encoded,
            mode='classification',
            feature_names=feature_names,
            class_names=['No Parking', 'Parking'],
            categorical_features=categorical_features,
            discretize_continuous=True
        )

    # Prediction function for CSV-based models
    def predict_parking(self, model, input_data):
        prediction = model.predict(input_data)
        return prediction

    # Output interpretation function for DNN parking prediction
    def interpret_dnn_output(self, predictions):
        result = []
        for i, pred in enumerate(predictions):
            if pred >= 0.5:
                result.append(
                    ("Business has parking and parking is validated.", "success"))
            else:
                result.append(
                    ("Business does not have parking or parking is not validated.", "error"))
        return result

# LIME prediction function for interpretability
    def predict_fn(self, x):
        proba_class_1 = self.dnn_model.predict(x)
        proba_class_0 = 1 - proba_class_1
        return np.hstack((proba_class_0, proba_class_1))

    def get_predictions(self, images):
        return self.cnn_model.predict(images)
