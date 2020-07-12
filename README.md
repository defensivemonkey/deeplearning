# DeepLearning

This repository is for all deep learning related materials.

### DEEPLEARNING.AI  [https://www.coursera.org/specializations/deep-learning?]    
WARNING: jupyter notebooks contains answers to the programming assignments in the course. Please do not copy if you are taking the course, only use it for study purposes.      

COURSE 1: Neural Networks and Deep Learning    
- Logistic Regression with a Neural Network mindset       
- Planar data classification with a hidden layer   
- Building your deep neural network: Step by Step    
- Deep Neural Network Application    

COURSE 2: Improving Deep Neural Networks: Hyperparameter tuning, Regularization and Optimization   
- Initialization     
- Regularization  
- Gradient Checking  
- Optimization  
- Tensorflow   

COURSE 3: Structuring Machine Learning Projects     

COURSE 4: Convolutional Neural Networks   
- Convolutional Model: step by step   
- Convolutional Model: application   
- Keras Tutorial  
- Residual Networks   
- Car Detection with YOLO
- Face Recognition   
    - Triplet Loss
    - Face Verification
    - Face Recognition
- Art Generation with Neural Style Transfer (NST)   
    - Content Cost Function   
      $$J_{content}(C,G) =  \frac{1}{4 \times n_H \times n_W \times n_C}\sum _{ \text{all entries}} (a^{(C)} - a^{(G)})^2\tag{1} $$
    - Style Cost Function  
      Gram Matrix: ${\displaystyle G_{ij} = v_{i}^T v_{j} = np.dot(v_{i}, v_{j})  }$    
      style cost one layer: $$J_{style}^{[l]}(S,G) = \frac{1}{4 \times {n_C}^2 \times (n_H \times n_W)^2} \sum _{i=1}^{n_C}\sum_{j=1}^{n_C}(G^{(S)}_{ij} - G^{(G)}_{ij})^2\tag{2} $$      
      style cost multiple layers: $$J_{style}(S,G) = \sum_{l} \lambda^{[l]} J^{[l]}_{style}(S,G)$$   
    - Total Cost Function  
      $$J(G) = \alpha J_{content}(C,G) + \beta J_{style}(S,G)$$   


COURSE 5: Sequence Models     
- Building a recurrent neural network - step by step  
- Dinosaur Island - Character-Level Language Modeling  
- Jazz improvisation with LSTM  
- Operations on word vectors - Debiasing  
- Emojify  
- Neural Machine Translation with Attention  
- Trigger word detection   


### FAST.AI  [https://www.fast.ai/]  
