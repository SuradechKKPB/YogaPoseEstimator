# YogaPoseEstimator
DAAN570 Final course project
Problem statement : Repetitive stress injury is extremely prevalent for anyone who works at the same spot for a long length of time, especially with the development of COVID-19 and the increase in work from home trend. We've identified certain flaws in traditional RSI prevention software on the market, and we've noted that yoga's popularity is growing by the day. The reason for this is the numerous physical, mental, and spiritual advantages that yoga may provide. Many people are following this trend and practice yoga without the help of a professional. However, doing yoga incorrectly or without adequate instruction can lead to serious health problems such as strokes and nerve damage. As a result, adhering to appropriate yoga poses is a vital consideration. In this work, we present a method for identifying the user's postures and providing visual guidance to the user. In order to be more engaging with the user, this procedure is done in real-time and utilizes the traditional webcam on the laptop/desktop to run the application.

Keywords : Yoga, posture, classification, movenet, keypoint

Data Collection: We took some images from open source yoga posture dataset from three following sites and applied basic data cleaning manually (e.g. remove corrupted images, remove misclassified yoga posture images).

1. Open source dataset from https://www.kaggle.com/general/192938
2. 3D synthetic dataset from https://laurencemoroney.com/2021/08/23/yogapose-dataset.html
3. Yoga-82 dataset from https://sites.google.com/view/yoga-82/home

There are total 5 notebooks in this series listed as following:
1. EDA and image augmentation note books >> https://www.kaggle.com/suradechk/01-eda-and-image-augmentation-v2
2. Setting up a baseline model using CNN >> https://www.kaggle.com/suradechk/02-baseline-model-using-cnn-v2
3. Keypoint generation using movenet >> https://www.kaggle.com/suradechk/03-keypoint-movenet-v2
4. Classification keypoint output using classical ML >> https://www.kaggle.com/suradechk/04-classification-using-keypoints-output-v2
5. Classification keypoint output using ANN >> https://www.kaggle.com/suradechk/05-classification-using-ann-v2

After completing model building steps with the above 5 notebooks, we have created our own unique RSI prevention software prototype using key loggers and webcam with the Jupyter notebook.  The ipynb file is uploaded to this github repository.  There can be version conflict with different python libraries so I have uploaded requirement.txt file in the link to set up virtual environment also.
