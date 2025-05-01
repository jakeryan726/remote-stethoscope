# Remote Stethoscope Diagnosis Assistant

## Disclaimer
#### This product is intended for research and educational purposes. This product is in no way a replacement for professional medical diagnosis, treatment, or advice. The model is designed to assist qualified medical professionals in making educated medical decisions, not for individual patients to self-diagnose. Note that this model may produce incorrect or incomplete results, and I make no guarantees regarding its reliability in practice. I accept no liability for any harm or consequences resulting from the use or misuse of this tool. Commercial use of this model is not permitted without explicit written permission.

## Introduction
#### This project develops a machine learning model to analyze and predict audio recorded with a remote stethoscope. By combining real data with synthetic data generated using a variational autoencoder, the prediction model is able to achieve a higher level of performance. The system is designed to detect nine different conditions: healthy, asthma, aortic stenosis, mitral regurgitation, mitral stenosis, mitral valve prolapse, COPD, heart failure, and pneumonia.

## Setup and Use
#### To create predictions using the model, make sure you have installed the required dependencies here:
#### - Python
#### - torch
#### - torchaudio
#### - scipy
#### - numpy

#### The model is run using the predict.py script in the src folder. This script takes one argument passed in with the flag --d, the path to the folder containing the WAV files on which to make predictions. To run the script for the folder test_audio_files in the project root directory, use the command:
#### ```python .\predict.py --d ../test_audio_files```

#### The prediction for each file in the test_audio_files folder will then be printed to the terminal. Note that this will fail if there are non-WAV files included in the test_audio_files folder.