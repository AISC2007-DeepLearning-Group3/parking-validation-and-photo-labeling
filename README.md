# Parking Validation and Photo Labelling 

This project stems from an assignment provided to us, focusing on applying advanced machine learning and deep learning techniques. Specifically, we employ Convolutional Neural Networks (CNN) and Deep Neural Networks (DNN) to predict image categories, such as food, indoor settings, outdoor settings, drinks, and menus. Additionally, we leverage machine learning models like AdaBoost and XGBoost to validate the availability of parking spaces at a restaurant, further enhancing the prediction accuracy and reliability of the system.
## Main App: https://dlcasestudy1group3.streamlit.app/
## DNN App: https://dlassignmentgroup3-dnn.streamlit.app/



https://github.com/user-attachments/assets/81bb9dff-a03b-45c1-b4a9-45cd5b1c7dfa


## Features

- **Overview:** This project deploys CNN, DNN, AdaBoost, and XGBoost models to predict parking availability and classify images (e.g., food, drink) based on provided datasets.
- **Dataset:** The datasets include business features for parking prediction and image data with labels like food, drink, indoor/outdoor, etc. Data's are preprocessed and code is saved as .pkl for easy use in the modelling bit.
- **Model Architecture:** We implemented DNN for parking prediction, AdaBoost/XGBoost for parking validation, and CNN/DNN for image label classification. Tuned models are saved as .h5 to be used for fast prediction
- **Deployment Details:** The saved models are utilized by our Streamlit application, which is hosted on the Streamlit server.
- **Results Interpretation:** The GUI displays both the model predictions and an interpretation of the outputs for better understanding.

## Getting Started

### Prerequisites

- Basic Understanding of Machine Learning and Deep Learning
- Python Programming
- Basic Knowledge of Streamlit


### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/AISC2007-DeepLearning-Group3/parking-validation-and-photo-labeling
   ```
2. Navigate to the project directory:
   ```bash
   cd parking-validation-and-photo-labeling
   ```
3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Application

Run the following command in your terminal:
```bash
streamlit run app.py
```

## Usage
Once your application is operational, you can upload an image to predict its label or submit a sample CSV file to validate parking. A demo video is provided above.

## Contributing
Contributions are welcome! Please fork the repository and submit a pull request with your improvements.

## License
This project is licensed under the MIT License - see the [LICENSE.md](LICENSE) file for details.

## Acknowledgements
- The Streamlit team for providing an easy way to create web applications for Python scripts.


