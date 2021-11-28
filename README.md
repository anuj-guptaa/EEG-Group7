# EEG Group 7
## Contributers:
Anuj Gupta - 122561

Nutapol Thungpao - 122148

Praewphan Tocharoenkul - 122497

Suphawich Sungkhavorn - 122564
## Preliminary Dataset Chosen:
Left/Right Hand Fist Movement - https://www.physionet.org/content/eegmmidb/1.0.0/
## Progress - November 21, 2021
Individual models will be attempted by each group member:
- Anuj: Bi-Directional LSTM
- Nutapol: CNN with Spectogram
- Praewphan: LSTM with attention
- Suphawich: Conv1D/Conv2D
## Progress - November 14, 2021
- Training method has been attempted on different (but similar dataset) by Nutapol https://github.com/nutapol97/Python-for-DS-AI_Nutapol_T./blob/main/physionet.ipynb?fbclid=IwAR3uxY8hj-Bpq09_LuVWFcwDkHt-QXDc_QOVN-hIlDJ_0cXXl0NnaOHSRXc
## NEXT STEPS:
Complete training and achieve good accuracy.
## Progress - November 7, 2021
- Dataset has been converted into frequency-domain with all channels seperated
- Power line at 50 Hz has been removed
- Bandpass filter (Unsure which frequency range to use yet)
## Progress - October 31, 2021
- Dataset has been chosen
- Group members have read at least 2 papers (Prasant Kumar Pattnaik, Jay Sarraf, Brain Computer Interface issues on hand movement + 2 more papers each)
- Chosen dataset has been explored by all members and coverted from .mat to .csv format by Nutapol
## NEXT STEPS:
Perform preprocessing - mainly with respect to artefact removal and fourier transform into time domain. (Eye was closed)
## Literature Review:
Research done on same dataset - https://www.frontiersin.org/articles/10.3389/fnhum.2020.00338/full

### Additional background reading done:
- Anuj Gupta 
  1. Paper - Performance Analysis of Left/Right Hand Movement Classification from EEG Signal by Intelligent Algorithms
  2. Paper - https://core.ac.uk/download/pdf/53189287.pdf

- Nutapol Thungpao
  1. Paper - https://www.sciencedirect.com/science/article/abs/pii/S1746809420303116
  2. Book - Data-Driven Science and Engineering (Steven L. Brunton, J. Nathan Kutz)
  
- Praewphan Tocharoenkul
  1. Paper - https://ieeexplore.ieee.org/document/5952111?fbclid=IwAR0wHq2DYeYiS0zyvJPbO3yEjzBai7LGSrXiyEr8IGuSjRXdnkUE9C9qbC0
  2. Paper - https://www.frontiersin.org/articles/10.3389/fncom.2017.00103/full?fbclid=IwAR0QL6QNLx65-geSEqc9FB3TT9hTndAVxWwdxfNKNWizypS6TfSuStjzL5Y
  
- Suphawich Sungkhavorn
  1. Paper - https://www.sciencedirect.com/science/article/pii/S1319157816300714?fbclid=IwAR1p35mBrWXlexk__kit2liY04cOIPAlOAYqE3eOIVXPWx4MCV2-L0mGoPA
  2. Paper - https://www.researchgate.net/publication/313643430_Motor_Imagery_Classification_Based_on_Deep_Convolutional_Neural_Network_and_Its_Application_in_Exoskeleton_Controlled_by_EEG?fbclid=IwAR3ZinlGjfp496J2rNCRab4poDC1uSatbt_WUxecn-v_y3J_6Je1p7r1rNg
