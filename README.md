# Speech Accent Detector
## Motivation
We wanted to create a project that could predict the country a person was raised in based on their speech accents. The idea behind this project is that most people develop distinctive linguistic patterns from the countries they grow up speaking in. Specifically, we want to identify noticeable accent patterns on English speech, and in the long term we hope that our project can be used to improve communication between people across various cultures and countries.

## Data and Preprocessing
### Dataset
[Official Kaggle Speech Accent Archive](https://www.kaggle.com/rtatman/speech-accent-archive)
Categories:
1. Arabic
2. English
3. French
4. Spanish
5. Russian

### Mel Frequency Cepstral Coefficient (MFCC) 
Primary preprocessing technique involves converting the wav to MFCC features. Wav file sampled at rate of 8000, creating an array of 12 MFCC values.

## Model Architectures
1. Random Forest
2. 1D Conv Net

