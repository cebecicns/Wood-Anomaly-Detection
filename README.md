# Wood-Anomaly-Detection
RUNNING THE PROJECT

Follow the steps below to run the project in a Google Colab environment:
1.	Clone the Repository
!git clone https://github.com/cebecicns/Wood-Anomaly-Detection.git
%cd Wood-Anomaly-Detection

2. Mount Google Drive
To access the dataset, mount your Google Drive:
from google.colab import drive
drive.mount('/content/drive')
3.  Create a Symbolic Link for the Dataset
Link the dataset stored in your Google Drive to the project directory:
!mkdir -p data
!ln -s "/content/drive/MyDrive/wood_dataset" "data/wood_dataset"
4.  Run the Project
Execute the main script to train and test the model:
!python main.py
