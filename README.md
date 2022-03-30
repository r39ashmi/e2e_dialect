
# Deep neural architectures for dialect classification with single frequency filtering and zero−time windowing feature representations

### Pre-requisites:
Install Matlab for feature extraction and Python==3.8 for classification </br>
Install required packages using: pip install -r requirements.txt

### Corpus: UT-Podcast
UT-Podcast is a speech corpus collected from podcasts, it has three dialects of English (US, UK, AU). Please download it from [here](https://crss.utdallas.edu/corpora/UT-Podcast/). For more details [refer](https://dl.acm.org/doi/abs/10.1016/j.specom.2015.12.004)

### Corpus: VoxCeleb
The train, validation, and test split of VoxCeleb corpus is provided in voxceleb_corpus folder. VoxCeleb1 corpus can be dowloaded from [here](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox1.html)


### Feature Extraction
For extraction of features (STFT, SFF, and ZTW based features), MATLAB is used. Code for feature extraction will soon be updated at feature_extraction/

### Neural Network Architectures for Dialect Classification
This project implements three neural architectures:
1. The code for **Convolution Neural Network** architecture can be found in main_cnn.py
2. The code for **Convolution Neural Network with embedded spectra filter as convolution layer** architecture can be found in cnn_spectral_layer.py
3. The code for **Temporal Convolution Neural Network** architecture can be found in main_tcnn.py
4. The code for **Time delay Neural Network** architecture can be found in main_tdnn.py

**NOTE**: Please find the pre-trained models at:
https://drive.google.com/drive/folders/1O4ZK1c8I5Vkglyka2fniUTpolyokTAsL?usp=sharing

### Classification metric
Unweighted Average Recall (UAR) is used as classification metric. Evaluation results will be updated soon.


### Citation

@article{dialect_class,  </br>
title = {Deep neural architectures for dialect classification with single frequency filtering and zero-time windowing feature representations},  </br>
author={Kethireddy, Rashmi and Kadiri, Sudarsana Reddy and Gangashetty, Suryakanth V},  </br>
journal = JASA,  </br>
volume = {151},  </br>
number = {2},  </br>
pages = {1077-1092},  </br>
year = {2022} </br>
}

