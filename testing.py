import streamlit as st
import pandas as pd
import joblib
from tensorflow.keras.models import load_model  # For DNN and CNN models
from PIL import Image
import numpy as np
import lime.lime_tabular
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image as keras_image
from tensorflow.keras.preprocessing.image import img_to_array
from lime import lime_image
from skimage.segmentation import mark_boundaries

# Load models and preprocessors
preprocessor = joblib.load('pkl/preprocessor_pipeline.pkl')
dnn_model = load_model('models/dnn_model.h5')  # Keras DNN model
ada_model = joblib.load('pkl/adaboost_model.pkl')  # AdaBoost model
xgb_model = joblib.load('pkl/xgboost_model.pkl')  # XGBoost model
# Ensemble model combining Ada, XGBoost, RandomForest
ensemble_model = joblib.load('pkl/ensemble_model.pkl')

# CNN model for image classification
cnn_model = load_model('models/cnn_model.h5')
# dnn_image_model = load_model('dnn_image_model.h5')  # DNN model for image classification

# Labels for image classification (replace with your actual labels)
class_labels = ['drink', 'food', 'inside', 'menu', 'outside']

# Load X_train from a pickle file
X_train = joblib.load('pkl/X_train.pkl')  # Ensure correct path

# Preprocess X_train
X_train_encoded = preprocessor.transform(X_train)

# Extract feature names from the preprocessor after transformation
feature_names = preprocessor.get_feature_names_out()

# Identify categorical features (if they were one-hot encoded in preprocessor)
categorical_features = [i for i, col in enumerate(
    feature_names) if 'onehot' in col]

# LIME Tabular explainer for the first tab
explainer = lime.lime_tabular.LimeTabularExplainer(
    X_train_encoded,
    mode='classification',
    feature_names=feature_names,
    class_names=['No Parking', 'Parking'],
    categorical_features=categorical_features,
    discretize_continuous=True
)

# Prediction function for CSV-based models


def predict_parking(model, input_data):
    prediction = model.predict(input_data)
    return prediction

# Output interpretation function for DNN parking prediction


def interpret_dnn_output(predictions):
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


def predict_fn(x):
    proba_class_1 = dnn_model.predict(x)
    proba_class_0 = 1 - proba_class_1
    return np.hstack((proba_class_0, proba_class_1))


def get_predictions(images):
    return cnn_model.predict(images)


def preprocess_image(img):
    img = img.resize((128, 128))  # Resizing the image to 256x256
    img_array = img_to_array(img)
    # Expanding dimensions to match model input
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize to [0, 1]
    return img_array


# Streamlit app with two tabs
st.title("🚗 Business Parking and Image Labeling with Interpretability")

# Create tabs
tab1, tab2 = st.tabs(["CSV Prediction (Parking)", "Image Prediction (Label)"])

# ---- Tab 1: CSV Prediction for Business Parking ----
with tab1:
    st.subheader("Predict Parking Availability and Validation from CSV Data")

    # File uploader for CSV data
    st.write("#### Upload your input file in CSV format:")
    uploaded_file = st.file_uploader("Choose a file", type="csv")

    if uploaded_file is not None:
        # Read and display CSV file
        input_data = pd.read_csv(uploaded_file)
        st.write("### 📋 Preview of the uploaded data:")
        st.write(input_data.head())

        # Preprocess the data
        try:
            input_data_encoded = preprocessor.transform(input_data)
        except Exception as e:
            st.error(f"❌ Error in preprocessing the input data: {e}")
            st.stop()

        # Ensure input shape matches the model
        if input_data_encoded.shape[1] != dnn_model.input_shape[1]:
            st.error(f"❌ Input shape mismatch. Expected {
                     dnn_model.input_shape[1]} features, but got {input_data_encoded.shape[1]}.")
            st.stop()

        # Model selection
        model_choice = st.selectbox(
            "Select the Model", ["DNN", "Adaboost", "XGBoost", "Ensemble"])

        # Prediction
        if st.button("🔍 Predict"):
            if model_choice == "DNN":
                predictions = predict_parking(dnn_model, input_data_encoded)
            elif model_choice == "Adaboost":
                predictions = predict_parking(ada_model, input_data_encoded)
            elif model_choice == "XGBoost":
                predictions = predict_parking(xgb_model, input_data_encoded)
            elif model_choice == "Ensemble":
                predictions = predict_parking(
                    ensemble_model, input_data_encoded)

            # Binary conversion for DNN/XGBoost/Ensemble
            if model_choice in ["DNN", "XGBoost", "Ensemble"]:
                predictions = (predictions > 0.5).astype(int)

            # Beautify predictions
            st.write("### 🔍 Predictions and Interpretations:")
            interpretation = interpret_dnn_output(predictions)
            for i, (text, status) in enumerate(interpretation):
                if status == "success":
                    st.success(f"Sample {i + 1}: {text}")
                else:
                    st.error(f"Sample {i + 1}: {text}")

            # Download results
            result_df = pd.DataFrame({"Prediction": predictions.flatten(
            ), "Interpretation": [text for text, _ in interpretation]})
            csv = result_df.to_csv(index=False)
            st.download_button(label="📥 Download Prediction Results",
                               data=csv, file_name="predictions.csv", mime="text/csv")

            # LIME interpretability for CSV predictions (for all samples)
            st.write("### 🧠 Lime Interpretability for All Samples")

            # Loop through each sample and show the LIME explanation graph
            # Limit to first 10 samples for display purposes
            for i in range(min(10, len(input_data_encoded))):
                st.write(f"#### Sample {i + 1} Explanation:")
                exp = explainer.explain_instance(
                    input_data_encoded[i], predict_fn, num_features=10)

                # Display the LIME explanation as a graph
                fig = exp.as_pyplot_figure()
                st.pyplot(fig)

# ---- Tab 2: Image Prediction for Labeling ----
with tab2:
    st.subheader("Predict Label of an Image using CNN or DNN")

    # File uploader for image data
    st.write("#### Upload an image file:")
    uploaded_image = st.file_uploader(
        "Choose an image", type=["png", "jpg", "jpeg"])

    if uploaded_image is not None:
        # Load and display the image
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Preprocess the image (ensure the image is resized to the input shape expected by the CNN model)
        preprocessed_image = preprocess_image(image)

        # Predict label
        predictions = get_predictions(preprocessed_image)
        predicted_class = np.argmax(predictions[0])
        predicted_label = class_labels[predicted_class]

        # Display predicted label
        st.write(f"### 🎯 Predicted Label: **{predicted_label}**")

        # Generate LIME explanation
        explainer = lime_image.LimeImageExplainer()

        # Ensure the image is correctly passed to LIME (convert to numpy array and normalize)
        explanation = explainer.explain_instance(
            # Resizing to match the input size and normalizing
            np.array(image.resize((256, 256))) / 255.0,
            get_predictions,
            top_labels=5,
            hide_color=0,
            num_samples=1000
        )

        # Get the image and mask for the predicted class
        temp, mask = explanation.get_image_and_mask(
            predicted_class,
            positive_only=True,
            num_features=5,
            hide_rest=True
        )

        # Visualize the results
        st.write("### 🧠 LIME Explanation")
        fig, ax = plt.subplots(1, 3, figsize=(15, 5))

        # Original image
        ax[0].imshow(image)
        ax[0].set_title('Original Image')
        ax[0].axis('off')

        # LIME explanation
        ax[1].imshow(mark_boundaries(temp / 2 + 0.5, mask))
        ax[1].set_title(f'LIME Explanation\nPredicted Class: {
                        predicted_label}')
        ax[1].axis('off')

        # Heatmap
        ax[2].imshow(mask, cmap='hot', interpolation='nearest')
        ax[2].set_title('Heatmap')
        ax[2].axis('off')

        # Display the LIME explanations as a plot
        st.pyplot(fig)

        # Show the top 5 features contributing to the prediction
        st.write("### 🔍 Top 5 Features Contributing to the Prediction:")
        ind = explanation.top_labels[0]
        dict_heatmap = dict(explanation.local_exp[ind])
        sorted_features = sorted(dict_heatmap.items(),
                                 key=lambda x: x[1], reverse=True)

        for feature, importance in sorted_features[:5]:
            st.write(f"Feature {feature}: Importance {importance:.4f}")
