# SpamFilter
### 1. 요약
  - 본 프로젝트는 이메일 내 불필요한 스팸메일을 구분하기 위해 진행되었다. 스팸 여부가 표시되어 있는 메일 데이터를 사용하였으며, fasttext와 CNN을 사용하여 binary classification을 진행하였다. 
  - 가장 성능이 좋은 CNN 모델은 2-gram~6-gram을 사용한 train 모델(11050156_100_0.0512.hdf5)로 test accuracy(leader board data 사용)은 94.979%의 성능을 보였다.  

    |test_size|val_size|test_size|epochs|batch_size|early stopping patience|
    |---|---|---|---|---|---|
    |0.8|0.2|0.1|100|128|30|  

    *[표_11050156_100_0.0512.hdf5 모델의 hyperparameters]*

### 2. 프로젝트 개요 및 primitives
- Spam filter classification은 binary classification에 해당하며, 스팸 메일이라면 1, 스팸 메일이 아니라면 0으로 분류한다. 
- Spam Filter Classifier가 갖춰야 하는 기능은 아래와 같다.
  1) 원본 데이터의 ‘message’를 word vector형태로 변환 : message를 CNN의 feature로 사용하기 위해서는 벡터로 표현되어야 하므로, 변환이 필요하다. 데이터 확인 시, 구어체의 단어가 많이 분포하고 있어, OOV(Out of Vocabulary)의 가능성이 높다. 대부분의 구어체는 핵심 형태소와 오타가 결합된 형태로, 구어체와 문어체의 일부분이 일치하는 특성이 있다.(e.g. 구어체 hellow와 문어체 hello) 따라서, 단어의 n-gram(n개의 character로 이루어진 group)을 기반으로 학습한 fasttext를 사용하였다.  
  2) 각 sample 별 binary classification : document의 경우, 단순한 단어의 분포로 분류하는 데에는 어려움이 있는데, 이는 같은 단어일지라도 다양한 class에서 사용될 수 있기 때문이다. 따라서, 문맥을 반영할 수 있는 모델이 적합하므로,  Convolutional Neural Networks for Sentence Classification (EMNLP 2014) 논문을 참고하여 CNN 모델을 구현하였다. 

### 3. 프로그램 구성 및 세부사항
  - spamFilter.py는 1) main과 2) class SpamFilter()와 두가지 부분으로 이루어져 있다. 
  - 3-1. main
    - 사용자가 입력한 모드에 따라, Train, Test, Infer 기능을 수행할 수 있다.  

      |Mode Number|Mode|Description|
      |---|---|---|
      |0|Train|새로운 classification model을 학습함|
      |1|Test|모델 훈련시 분리한 test data를 사용하여 accuracy 측정|
      |2|Infer|label이 없는 데이터와 pre-trained model을 사용하여 class 예측|  

        *[표_프로그램 실행 mode]*
  - 3-2. class SpamFilter()  
    ![image](https://user-images.githubusercontent.com/62787552/109521364-937b4080-7af0-11eb-9b87-fc590952a3c7.png)  
    *[그림_SpamFilter 구조]*

