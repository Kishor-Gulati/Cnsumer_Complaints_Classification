# Consumer_Complaints_Classification

## Consumer Complaint Classification using RNN

This repository contains a Jupyter Notebook implementing a text classification task using Recurrent Neural Networks (RNNs). The task involves classifying consumer complaints into categories such as "Bank account or service," "Credit card or prepaid card," and "Student loan" based on the narrative provided by consumers.

### Dataset
The dataset used for this project is sourced from consumer complaints data, which includes complaints about various financial products and services. It consists of narratives provided by consumers along with their associated product categories.

### Key Steps
- Data preprocessing: Cleaning text data, tokenization, padding sequences.
- Model development: Building an RNN model using Keras with an Embedding layer, SimpleRNN layer, and Dense layer.
- Training and evaluation: Training the model on the training set and evaluating its performance on the test set. Metrics such as accuracy, precision, recall, and F1-score are used for evaluation.
- Prediction: Making predictions on new, unseen data to classify consumer complaints into categories.

### Results
The trained model achieves an accuracy of approximately 79.6% on the test set, demonstrating its effectiveness in classifying consumer complaints.

### Files
- `Consumer_Complaint_Classification_RNN.ipynb`: Jupyter Notebook containing the project code.
- `consumer_complaint_subset.csv`: Subset of the original dataset used for training and testing the model.
- `model_rnn.h5`: Trained RNN model saved in HDF5 format.

### Requirements
- Python 3.x
- TensorFlow
- Keras
- NumPy
- Pandas
- Matplotlib
- NLTK

### Usage
1. Clone the repository: `git clone <repository_url>`
2. Open and run the Jupyter Notebook `Consumer_Complaint_Classification_RNN.ipynb`.
3. Follow the step-by-step instructions to preprocess the data, build, train, and evaluate the RNN model.
4. Make predictions on new data using the trained model.

Feel free to explore and modify the notebook according to your needs. If you have any questions or feedback, please don't hesitate to reach out!
