# Detection of Arrhythmia in Real-time using ECG Signal Analysis and Convolutional Neural Networks

## Summary:

In this paper, we focus on presenting a method through which Cardiac Arrhythmia can be detected and classified in real time by a wearable ECG device. The MIT-BIH Database is used to train and evaluate our CNN model, resulting in an accuracy of 99.625%. We are using an Arduino Nano, hooked to an AD8232 breakout board to measure and save heartbeat data. The IEEE paper for this project is attached to the repository. Below are snippets of the paper presented at the CPEE 2020 International Conference, held in Poland, where we received the best paper award.


![image](https://github.com/sashank3/Arrhythmia/assets/41186713/7d3faea0-f8b9-4ac7-8216-163c3f4cfa1b)
![image](https://github.com/sashank3/Arrhythmia/assets/41186713/bf174917-9454-4dbb-a23b-5106b98be2b0)



## Working:

To download the datasets required for the code to run, please visit the following google drive links:

MIT-BIH Test Data (98.1 MB) - https://drive.google.com/file/d/1Nh8YM2L62F_fO9zwrNNfq0V7JwOqscZT/view?usp=drive_link

MIT-BIH Train Data (392.4 MB) - https://drive.google.com/file/d/1HBjHTxCoRISQ_JMDYyT7Tw3yOAKVHoVV/view?usp=drive_link

PTBDB Normal Data (18.1 MB) - https://drive.google.com/file/d/1bKAIplMgpyNHqfsxqIPegwdCW1WHqeYk/view?usp=drive_link

PTBDB Abnormal Data (47.1 MB) - https://drive.google.com/file/d/1BqNAqd00MRhO3Zf5_jz1JY2UsuhfCHHI/view?usp=drive_link



All required code is available in the working_directory. To start, please run the main_trainer python script inside the 'code' folder.

