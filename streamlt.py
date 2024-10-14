from matplotlib import pyplot as plt
import numpy as np
import streamlit as st
import pandas as pd
from PIL import Image
from lime import lime_image
from utils import Helper, ModelPrediction
from skimage.segmentation import mark_boundaries


class StreamLit_App(Helper):

    def init__(self):
        self.explainer = ModelPrediction().Interpretability_for_parking
        self.model_choice = None
        self.input_data_encoded = None

    def predict(self, model, data):
        return ModelPrediction().predict_parking(model, data)

    def PredictChoice(self, model_choice, input_data_encoded):
        if model_choice == "DNN":
            predictions = self.predict(self.dnn_model, input_data_encoded)
        elif model_choice == "Adaboost":
            predictions = self.predict(self.ada_model, input_data_encoded)
        elif model_choice == "XGBoost":
            predictions = self.predict(self.xgb_model, input_data_encoded)
        elif model_choice == "Ensemble":
            predictions = self.predict(
                self.ensemble_model_model, input_data_encoded)

        # Binary conversion for DNN/XGBoost/Ensemble
        if model_choice in ["DNN", "XGBoost", "Ensemble"]:
            predictions = (predictions > 0.5).astype(int)

        return predictions

    def result(self, predictions, interpretation):
        # Download results
        return pd.DataFrame({"Prediction": predictions.flatten(
        ), "Interpretation": [text for text, _ in interpretation]}).to_csv(index=False)

    def lime_interpretability(self):
        # Loop through each sample and show the LIME explanation graph
        # Limit to first 10 samples for display purposes
        for i in range(min(10, len(self.input_data_encoded))):
            st.write(f"#### Sample {i + 1} Explanation:")
            exp = self.explainer.explain_instance(
                self.input_data_encoded[i], ModelPrediction().predict_fn, num_features=10)

            # Display the LIME explanation as a graph
            fig = exp.as_pyplot_figure()
            st.pyplot(fig)

    def predict_button(self):
        predictions = self.PredictChoice(
            self.model_choice, self.input_data_encoded)

        # Beautify predictions
        st.write("### 🔍 Predictions and Interpretations:")

        interpretation = ModelPrediction().interpret_dnn_output(predictions)
        for i, (text, status) in enumerate(interpretation):
            if status == "success":
                st.success(f"Sample {i + 1}: {text}")
            else:
                st.error(f"Sample {i + 1}: {text}")

        st.download_button(label="📥 Download Prediction Results",
                           data=self.result(predictions, interpretation), file_name="predictions.csv", mime="text/csv")

        # LIME interpretability for CSV predictions (for all samples)
        st.write("### 🧠 Lime Interpretability for All Samples")

        self.lime_interpretability(self.input_data_encoded)

    def tabOne(self, tab1):
        '''
        CSV Prediction for Business Parking
        '''
        with tab1:
            st.subheader(
                "Predict Parking Availability and Validation from CSV Data")

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
                    self.input_data_encoded = self.preprocessor.transform(
                        input_data)
                except Exception as e:
                    st.error(f"❌ Error in preprocessing the input data: {e}")
                    st.stop()

                # Ensure input shape matches the model
                if self.input_data_encoded.shape[1] != self.dnn_model.input_shape[1]:
                    st.error(f"❌ Input shape mismatch. Expected {
                        self.dnn_model.input_shape[1]} features, but got {self.input_data_encoded.shape[1]}.")
                    st.stop()

                # Model selection
                self.model_choice = st.selectbox(
                    "Select the Model", ["DNN", "Adaboost", "XGBoost", "Ensemble"])

                # Prediction
                if st.button("🔍 Predict"):
                    self.predict_button()

    def tabTwo(self, tab2):
        '''
        Tab 2: Image Prediction for Labeling
        '''
        with tab2:
            st.subheader("Predict Label of an Image using CNN or DNN")

            # File uploader for image data
            st.write("#### Upload an image file:")
            uploaded_image = st.file_uploader(
                "Choose an image", type=["png", "jpg", "jpeg"])

            if uploaded_image is not None:
                # Load and display the image
                image = Image.open(uploaded_image)
                st.image(image, caption="Uploaded Image",
                         use_column_width=True)

                # Preprocess the image (ensure the image is resized to the input shape expected by the CNN model)
                preprocessed_image = ModelPrediction().preprocess_image(image)

                # Predict label
                predictions = ModelPrediction().get_predictions(preprocessed_image)
                predicted_class = np.argmax(predictions[0])
                predicted_label = self.class_labels[predicted_class]

                # Display predicted label
                st.write(f"### 🎯 Predicted Label: **{predicted_label}**")

                # Generate LIME explanation
                explainer = lime_image.LimeImageExplainer()

                # Ensure the image is correctly passed to LIME (convert to numpy array and normalize)
                explanation = explainer.explain_instance(
                    # Resizing to match the input size and normalizing
                    np.array(image.resize((256, 256))) / 255.0,
                    ModelPrediction().get_predictions,
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
