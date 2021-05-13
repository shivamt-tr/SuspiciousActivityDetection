# SuspiciousActivityDetection
## Final Year Project
The objective of our system is to detect suspicious or abnormal activities in videos to avoid future mishappening or to give alert whenever any type of mishappening occurs. These anomalous activity recognition systems classify safe and suspicious activities in the videos using deep learning.

The analytic video is a very challenging area of research in computer vision. Ensuring a high level of security in a public space monitored by a surveillance camera is a difficult task in recent years. Identification of suspicious entities from real time and offline video streaming can help monitor the activities with minimal human effort.

The aim of our project is to build a video classifier which can classify the given video clip into one of the 2 categories i.e. Suspicious & Safe and automates the task of monitoring real time videos and avoiding the manual monitoring by humans.

The video classification system is based on convolution and recurrent neural networks and it classifies the videos based using extracted features from frames.

## Pipeling of the Project
Our work proposes a simple pipeline to classify the video into categories of type of action performed.
* The system uses features extracted by a pre-trained 2-Dimensional Convolution Neural Network that is applied to each frame of the video. We will be using a pre-trained model called inception developed by Google. Inception-v3 is trained on the ImageNet Large Visual Recognition Challenge dataset. This is a standard task in computer vision, where models try to classify entire images into 1,000 classes like “zebra,” “dalmatian,” and “dishwasher.” The features extracted are high-level features of the images and thus have reduced complexity and is suitable for classification.
* These features are processed into a feature map in the order of frames and fed to LSTM (long short term memory).
* The LSTM's purpose of this network is to make sense of the sequence of the actions portrayed. This network has an LSTM cell in the first layer, followed by two hidden layers (one with 1,024 neurons and relu activation and the other with 50 neurons with a sigmoid activation), and the output layer is a three-neuron layer with softmax activation, which gives us the final classification.
