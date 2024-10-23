### Deep Learing, 딥 러닝  

- 인공 신경망(Artificial Neural Network)의 층을 연속적으로 깊게 쌓아올려 데이터를 학습하는 방식을 의미한다.
- 인간이 학습하고 기억하는 매커니즘을 모방한 기계학습이다.
- 인간은 학습 시, 뇌에 있는 뉴런이 자극을 받아들여서 일정 자극 이상이 되면, 화학물질을 통해 다른 뉴런과 연결되며 해당 부분이 발달한다.
- 자극이 약하거나 기준치를 넘지 못하면, 뉴런은 연결되지 않는다.
- 입력한 데이터가 활성 함수에서 임계점을 넘게 되면 출력된다.
- 초기 인공 신경망(Perceptron)에서 깊게 층을 쌓아 학습하는 딥 러닝으로 발전한다.
- 딥 러닝은 Input nodes layer, Hidden nodes layer, Output nodes layer, 이렇게 세 가지 층이 존재한다.

![](https://velog.velcdn.com/images/heejookang/post/e9d31be7-c53d-4e77-b60a-bbe16bd12676/image.png)


### SLP (Single Layer Perceptron), 단층 퍼셉트론, 단일 퍼셉트론

- 가장 단순한 형태의 신경망으로서, Hidden Layer가 없고 Single Layer로 구성되어 있다.
- 퍼셉트론의 구조는 입력 feature와 가중치, activation function, 출력 값으로 구성되어 있다.
- 신경 세포에서 신호를 전달하는 축삭돌기의 역할을 퍼셉트론에서는 가중치가 대신하고,  
  입력 값과 가중치 값은 모두 인공 뉴런(활성 함수)으로 도착한다.
- 가중치의 값이 클수록 해당 입력 값이 중요하다는 뜻이고,  
  인공 뉴런(활성 함수)에 도착한 각 입력 값과 가중치 값을 곱한 뒤 전체 합한 값을 구한다.
- 인공 뉴런(활성 함수)은 보통 시그모이드 함수와 같은 계단 함수를 사용하여,  
  합한 값을 확률로 변환하고 이 때, 임계치를 기준으로 0 또는 1을 출력한다.

![](https://velog.velcdn.com/images/heejookang/post/8a36e434-a71e-4981-bd21-b75b9ef99a2f/image.png)


- 로지스틱 회귀 모델이 인공 신경망에서는 하나의 인공 뉴런으로 볼 수 있다.
- 결과적으로 퍼셉트론은 회귀 모델과 마찬가지로 실제 값과 예측 값의 차이가 최소가 되는 가중치 값을 찾는 과정이 퍼셉트론이 학습하는 과정이다.
- 최초 가중치 값을 설정한 뒤 입력 feature 값으로 예측 값을 계산하고, 실제 값과의 차이를 구한 뒤 이를 줄일 수 있도록 가중치 값을 변경한다.
- 퍼셉트론의 활성화 정도를 편향(bias)으로 조절할 수 있으며, 편향을 통해 어느정도의 자극을 미리 주고 시작할 수 있다.
- 뉴런이 활성화되기 위해 필요한 자극이 1000이라고 가정하면, 입력 값을 500만 받아도 편향을 2로 주어 1000을 만들 수 있다.

  
![](https://velog.velcdn.com/images/heejookang/post/54f19018-2378-4d79-81d8-38de1d6fb13a/image.png)
![](https://velog.velcdn.com/images/heejookang/post/933b8360-7b8f-4d43-9b50-11ff16ec21dd/image.png)


  
- 퍼셉트론의 출력 값과 실제 값의 차이를 줄여나가는 방향성으로 계속해서 가중치 값을 변경하며, 이 때 경사하강법을 사용한다.  
  
![](https://velog.velcdn.com/images/heejookang/post/ec8560f7-0fa2-4bfb-931c-fa1913342f24/image.gif)


#### SGD (Stochastic Gradient Descent), 확률적 경사 하강법
- 경사 하강법 방식은 전체 학습 데이터를 기반으로 계산한다. 하지만 입력 데이터가 크고 레이어가 많을 수록 많은 자원이 소모된다.
- 일반적으로 메모리 부족으로 인해 연산이 불가능하기 때문에, 이를 극복하기 위해 SGD 방식이 도입되었다.
- 전체 학습 데이터 중, 단 한 건만 임의로 선택하여 경사 하강법을 실시하는 방식을 의미한다.
- 많은 건 수 중에 한 건만 실시하기 때문에, 빠르게 최적점을 찾을 수 있지만 노이즈가 심하다.
- 무작위로 추출된 샘플 데이터에 대해 경사 하강법을 실시하기 때문에 진폭이 크고 불안정해 보일 수 있다.
- 일반적으로 사용되지 않고, SGD를 얘기할 때에는 보통 미니 배치 경사 하강법을 의미한다.

![](https://velog.velcdn.com/images/heejookang/post/0c0ddfbc-dfe3-4525-8a35-a83137eede43/image.png)


#### Mini-Batch Gradient Descent, 미니 배치 경사 하강법  
- 전체 학습 데이터 중, 특정 크기(Batch 크기)만큼 임의로 선택해서 경사 하강법을 실시한다. 이 또한, 확률적 경사 하강법

![](https://velog.velcdn.com/images/heejookang/post/7af21027-4d98-4700-8a94-224e17409e27/image.png)


- 전체 학습 데이터가 1000건이라고 하고, batch size를 100건이라 가정하면,  
  전체 데이터를 batch size만큼 나눠서 가져온 뒤 섞고, 경사하강법을 계산한다.  
  이 경우, 10번 반복해야 1000개의 데이터가 모두 학습되고 이를 epoch라고 한다. 즉, 10 epoch * 100 batch이다.

![](https://velog.velcdn.com/images/heejookang/post/1b4d59d0-8b32-4aeb-a921-24d97ed0c568/image.png)


### Multi Layer Perceptron, 다층 퍼셉트론, 다중 퍼셉트론
- 보다 복잡한 문제의 해결을 위해서 입력층과 출력층 사이에 은닉층이 포함되어 있다.
- 퍼셉트론을 여러층 쌓은 인공 신경망으로서, 각 층에서는 활성함수를 통해 입력을 처리한다.
- 층이 깊어질 수록 정확한 분류가 가능해지지만, 너무 깊어지면 Overfitting이 발생한다.

![](https://velog.velcdn.com/images/heejookang/post/8bfaadf0-c082-4fdf-81f7-94a064ccdf02/image.png)

![](https://velog.velcdn.com/images/heejookang/post/64f30f22-048c-44af-aa9f-e49178886461/image.png)

### ANN (Artificial Neural Network), 인공 신경망

- 은닉층이 1개일 경우 이를 인공 신경망이라고 한다.

![](https://velog.velcdn.com/images/heejookang/post/da0f2dd0-45b1-4154-9637-a8c62a06f5b6/image.png)


### DNN (Deep Neural Network), 심층 신경망
- 은닉층이 2개 이상일 경우 이를 심층 신경망 이라고 한다.

![](https://velog.velcdn.com/images/heejookang/post/d0c0a6d1-7a22-43b3-a91f-0a4302c62640/image.png)

### Back-Propagation, 역전파

- 심층 신경망에서 최종 출력(예측)을 하기 위한 식이 생기지만 식이 너무 복잡해지기 때문에 편미분을 진행하기에 한계가 있다.
- 즉, 편미분을 통해 가중치 값을 구하고, 경사 하강법을 통해 가중치 값을 업데이트 하며, 손실 함수의 최소값을 찾아야 하는데,
  순 방향으로는 복잡한 미분식을 계산할 수가 없다. 따라서 미분의 연쇄 법칙(Chain Rule)을 상ㅇ하여 역방향으로 편미분을 진행한다.

#### 합성 함수의 미분
![](https://velog.velcdn.com/images/heejookang/post/a3526b73-52bb-4499-b6f1-7c20ad89afef/image.png)

---
![](https://velog.velcdn.com/images/heejookang/post/c6e9c612-bbf6-45c4-bda6-2403b8699c5e/image.png)


![](https://velog.velcdn.com/images/heejookang/post/28462670-89eb-4f2c-9e8e-5fef66a1ea25/image.png)

![](https://velog.velcdn.com/images/heejookang/post/70f88b0c-4502-4201-a56e-6c7c2613b8dc/image.png)

![](https://velog.velcdn.com/images/heejookang/post/7706cbe2-0465-440f-9a92-3f703b3e8de8/image.png)


### Activation Function, 활성화 함수

- 인공 신경망에서 입력 값에 가중치를 곱한 뒤 합한 결과를 적용하는 함수이다.
--- 

1. 시그모이드 함수
    - 은닉층이 아닌 최종 활성화 함수 즉, 출력층에서 사용된다.
    - 은닉층에서 사용 시, 입력 값이 양의 방향으로 큰 값일 경우 출력 값의 변화가 없으며, 음의 방향도 마찬가지이다.
    - 평균이 0이 아니기 때문에 정규 분포 형태가 아니고, 이는 방향에 따라 기울기가 달라져서 탐색 경로가 비효율적(지그재그)이 된다.

![](https://velog.velcdn.com/images/heejookang/post/62296bbd-4987-4fc2-b077-2c2880d1aa0c/image.png)


2. 소프트맥스 함수
    - 은닉층이 아닌 최종 활성화 함수(출력층)에서 사용된다.
    - 시그모이드와 유사하ㅔ 0 ~ 1 사이의 값을 출력 하지만, 이진 분류가 아닌 다중 분류를 통해 모든 확률값이 1이 되도록 해준다.
    - 여러개의 타겟 데이터를 분류하는 다중 분류의 최종 활성화 함수(출력층)로 사용된다.  

![](https://velog.velcdn.com/images/heejookang/post/241d4749-111e-4473-853a-d93647e953e6/image.png)


 
3. 탄젠트 함수
    - 은닉층이 아닌 최종 활성화 함수(출력층)에서 사용된다.
    - 은닉층에서 사용 시, 시그모이드와 달리 -1 ~ 1 사이의 값을 출력해서 평균이 0이 될 수 있지만,
      여전히 입력 값의 양의 방향으로 큰값일 경우 출력값의 변화가 미비 하고 음의 방향도 마찬가지다.

![](https://velog.velcdn.com/images/heejookang/post/6cb039b4-ec2e-4e83-8b1b-812bbae98196/image.png)


4. 렐루 함수
   - 대표적인 은닉층의 활성 함수이다.
   - 입력 값이 0보다 작으면 출력은 0, 0보다 크면 입력 값을 출력하게 된다.
  
![](https://velog.velcdn.com/images/heejookang/post/de162791-92a3-4239-94a0-2ef9ec936ab0/image.png)

### Cross Entropy
- 실제 데이터의 확률 분포와, 학습된 모델이 계산한 확률 분포의 차이를 구하는 데 사용된다.
- 분류 문제에서 원-핫 인코딩을 통해 사용할 수 있는 오차 계산법이다.

![](https://velog.velcdn.com/images/heejookang/post/e691c90b-1d6b-4f96-97ef-0adaae17321f/image.png)


### Optimizer, 최적화
- 최적의 경사 하강법을 적용하기 위해 필요하며, 최소값을 찾아가는 방법들을 의미한다.
- loss를 줄이는 방향으로 최소 loss를 보다 빠르고 안정적으로 수렴할 수 있어야 한다.

![](https://velog.velcdn.com/images/heejookang/post/acccaa39-9740-4ed8-8f91-63511562c7f4/image.png)



#### Momentum
- 가중치를 계속 업데이트할 때마다 이전의 값을 일정 수준 반영시키면서 새로운 가중치로 업데이트한다.
- 지역 최소값에서 벗어나지 못하는 문제를 해결할 수 있으며, 진행했던 방향만큼 추가적으로 더하여, 관성처럼 빠져나올 수 있게 해준다.

![](https://velog.velcdn.com/images/heejookang/post/e9f38f97-06ad-4294-8fe0-902612c34c15/image.png)


#### AdaGrad (Adaptive Gradient)
- 가중치 별로 서로 다른 학습률을 동적으로 적용한다.
- 적게 변화된 가중치는 보다 큰 학습률을 적용하고, 많이 변화된 가중치는 보다 작은 학습률을 적용시킨다.
- 처음에는 큰 보폭으로 이동하다가 최소값에 가까워질 수록 작은 보폭으로 이동하게 된다.
- 과거의 모든 기울기를 사용하기 때문에 학습률이 급격히 감소하여, 분모가 커짐으로써 학습률이 0에 가까워지는 문제가 있다.

![](https://velog.velcdn.com/images/heejookang/post/7a3b8a6b-76b8-412b-bb4e-9406f8bb130e/image.png)
![](https://velog.velcdn.com/images/heejookang/post/64d90943-2260-427e-bcec-ba003524d25c/image.png)


#### RMSProp (Root Mean Sqaure Propagation)
- AdaGrad의 단점을 보완한 기법으로서, 학습률이 지나치게 작아지는 것을 막기 위해 지수 가중 평균법(exponentially weighted average)을 통해 구한다.
- 지수 가중 평균법이란, 데이터의 이동 평균을 구할 때 오래된 데이터가 미치는 영향을 지수적으로 감쇠하도록 하는 방법이다.
- 이전의 기울기들을 똑같이 더해가는 것이 아니라 훨씬 이전의 기울기는 조금 반영하고 최근의 기울기를 많이 반영한다.
- feature마다 적절한 학습률을 적용하여 효율적인 학습을 진행할 수 있고, AdaGrad보다 학습을 오래 할 수 있다.

#### Adam (Adaptive Moment Estimation)
- Momentum과 RMSProp 두 가지 방식을 결합한 형태로서, 진행하던 속도에 관성을 주고, 지수 가중 평균법을 적용한 알고리즘이다.
- 최적화 방법 중에서 가장 많이 사용되는 알고리즘이며, 수식은 아래와 같다.

![](https://velog.velcdn.com/images/heejookang/post/d3a8d9ff-d8cc-4d27-b044-d42218ccd37e/image.png)
![](https://velog.velcdn.com/images/heejookang/post/732c3558-beb4-4c1f-9119-31828be156cd/image.png)

### Tensorflow, 텐서플로우
- 구글이 개발한 오픈소스 소프트웨어 라이브러리이며, 머신러닝과 딥러닝을 쉽게 사용할 수 있도록 다양한 기능을 제공한다.
- 주로 이미지 인식이나 반복 신경망 구성, 기계 번역, 필기 숫자 반별 등을 위한 각종 신경망 학습에 사용된다.
- 딥러닝 모델을 만들 때, 기초부터 세세하게 작업해야 하기 때문에 진입장벽이 높다.

![](https://velog.velcdn.com/images/heejookang/post/203f86ce-3d76-4d35-8343-ccd9121e8da2/image.png)


### Keras, 케라스
- 일반 사용 사례에 "최적화, 간단, 일관, 단순화"된 인터페이스를 제공한다.
- 손쉽게 딥러닝 모델을 개발하고 활용할 수 있도록 직관적인 API를 제공한다.
- 텐서플로우 2버전 이상부터 케라스가 포함되었기 때문에 텐서플로우를 통해 케라스를 사용한다.
- 기존 Keras 패키지보다는 이제 Tensorflow에 내장된 Keras 사용이 더 권장된다.

![](https://velog.velcdn.com/images/heejookang/post/378af155-f8bb-4a91-99c6-0622476c0683/image.png)

### Grayscale, RGB
- 흑백 이미지와 컬러 이미지는 각 2차원과 3차원으로 표현될 수 있다.
- 흑백 이미지는 0 ~ 255를 갖는 2차원 배열(높이 X 너비)이고,  
  컬러 이미지는 0 ~ 255를 갖는 R, G, B 2차원 배열 3개를 갖는 3차원 배열(높이 X 너비 X 채널)이다.

![](https://velog.velcdn.com/images/heejookang/post/73d0ee24-4891-48af-879b-807aecf406cd/image.png)

![](https://velog.velcdn.com/images/heejookang/post/2812662e-299e-49ea-9c56-7eb804bd7d40/image.png)

### Grayscale Image Matrix
- 검은색에 가까운 색은 0에 가깝고 흰색에 가까우면 255에 가깝다.
- 모든 픽셀이 feature이다.

![](https://velog.velcdn.com/images/heejookang/post/06eb9280-2e8a-4f94-ac27-f6173dadbce8/image.png)

### Sequential API, Functional API

#### Sequential API
- 간단한 모델을 구현하기에 적합하고 단순하게 층을 쌓는 방식으로 쉽고 사용하기가 간단한다.
- 단일 입력 및 출력만 있으므로 레이어를 공유하거나 여러 입력 또는 출력을 가질 수 있는 모델을 생성할 수 없다.

#### Functional API
- Functional API는 Sequential API로는 구현하기 어려운 복잡한 모델들을 구현할 수 있다.
- 여러 개의 입력 및 출력을 가진 모델을 구현하거나 층 간의 연결 및 연산을 수행하는 모델 구현 시 사용한다.

#### 성능 평가
![](https://velog.velcdn.com/images/heejookang/post/cadd4911-0928-40bf-8033-9cd73aa09ffb/image.png)


### Callback API
- 모델이 학습 중에 충돌이 발생하거나 네트워크가 끊기면, 모든 훈련 시간이 낭비될 수 있고,  
  과적합을 방지하기 위해 훈련을 중간에 중지해야 할 수도 있다.
- 모델이 학습을 시작하면 학습이 완료될 때까지 아무런 제어를 하지 못하게 되고,  
  신경망 훈련을 완료하는 데에는 몇 시간 또는 며칠이 걸릴 수 있기 때문에 모델을 모니터링하고 제어할 수 있는 기능이 필요하다.
- 훈련 시(fit()) Callback API를 등록시키면 반복 내에서 특정 이벤트 발생마다 등록된 callback이 호출되어 수행된다.

**ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=False, save_weight_only=False, mode='auto')**
- 특정 조건에 따라서 모델 또는 가중치를 파일로 저장한다.
- filepath: "weights.{epoch:03d}-{val_loss:.4f}-{acc:.4f}.weights.hdf5"와 같이 모델의 체크포인트를 저장한다.
- monitor: 모니터링할 성능 지표를 작성한다.
- save_best_only: 가장 좋은 성능을 나타내는 모델을 저장할 지에 대한 여부
- save_weights_only: weights만 저장할 지에 대한 여부
- mode: {auto, min, max} 중 한 가지를 작성한다. monitor의 성능 지표에 따라 좋은 경우를 선택한다.  
  *monitor의 성능 지표가 감소해야 좋은 경우 min, 증가해야 좋은 경우 max, monitor의 이름으로부터 자동으로 유추하고 싶다면 auto


**ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=0, mode='auto', min_lr=0)**
- 특정 반복동안 성능이 개선되지 않을 때, 학습률을 동적으로 감소시킨다.
- monitor: 모니터링할 성능 지표를 작성한다.
- factor: 학습률을 감소시킬 비율, 새로운 학습률 = 기존 학습률 * factor
- patience: 학습률을 줄이기 전에 monitor할 반복 횟수
- mode: {auto, min, max} 중 한 가지를 작성한다. monitor의 성능 지표에 따라 좋은 경우를 선택한다.  
  *monitor의 성능 지표가 감소해야 좋은 경우 min, 증가해야 좋은 경우 max, monitor의 이름으로부터 자동으로 유추하고 싶다면 auto


**EarlyStopping(monitor='val_loss', patience=0, verbose=0, mode='auto')**
- 특정 반복동안 성능이 개선되지 않을 때, 학습을 조기에 중단한다.
- monitor: 모니터링할 성능 지표를 작성한다.
- patience: Early Stopping을 적용하기 전에 monitor할 반복 횟수.
- mode: {auto, min, max} 중 한 가지를 작성한다. monitor의 성능 지표에 따라 좋은 경우를 선택한다.  
  *monitor의 성능 지표가 감소해야 좋은 경우 min, 증가해야 좋은 경우 max, monitor의 이름으로부터 자동으로 유추하고 싶다면 auto

  ### CNN (Convolutional Neural Network), 합성곱 신경망
- 실제 이미지 데이터는 분류 대상이 이미지에서 고정된 위치에 있지 않은 경우가 대부분이다.
- 실제 이미지 데이터를 분류하기 위해서는, 이미지의 각 feature들을 그대로 학습하는 것이 아닌, CNN으로 패턴을 인식한 뒤 학습해야 한다.

![](https://velog.velcdn.com/images/heejookang/post/60fd1625-3845-4191-b0ad-a88c07a80a76/image.png)![](https://velog.velcdn.com/images/heejookang/post/25294bdf-d20a-4cbb-83e5-dce99d0380f6/image.png)



- 이미지의 크기가 커질 수록 굉장히 많은 Weight가 필요하기 때문에 분류기에 바로 넣지 않고, 이를 사전에 추출 및 축소해야 한다.
- CNN은 인간의 시신경 구조를 모방한 기술로서, 이미지의 패턴을 찾을 때 사용한다.
- Feature Extraction을 통해 각 단계를 거치면서, 함축된 이미지 조각으로 분리되고 각 이미지 조각을 통해 이미지의 패턴을 인식한다.

![](https://velog.velcdn.com/images/heejookang/post/056be217-2427-4d89-b335-591fb7a11719/image.png)


- CNN은 분류하기에 적합한 최적의 featrue를 추출하고, 최적의 feature를 추출하기 위한 최적의 Weight와 Filter를 계산한다.

![](https://velog.velcdn.com/images/heejookang/post/dd3b8235-d968-4ee4-808c-ad9d763a97d9/image.png)

#### Filter
- 보통 정방 행렬로 구성되어 있고, 원본 이미지에 슬라이딩 윈도우 알고리즘을 사용하여 순차적으로 새로운 픽셀값을 만들면서 적용한다.
- 사용자가 목적에 맞는 특정 필터를 만들거나 기존에 설계된 다양한 필터를 선택하여 이미지에 적용한다.  
  하지만, CNN은 최적의 필드값을 학습하여 스스로 최적화한다.

![](https://velog.velcdn.com/images/heejookang/post/3f9a7ed6-0a5d-4dbf-afdb-4a4665e15e24/image.gif)

![](https://velog.velcdn.com/images/heejookang/post/c8e1759b-00be-4a53-9014-d6379169f6b9/image.png)



- 필터 하나 당, 이미지의 채널 수 만큼 Kernel이 존재하고, 각 채널에 할당된 필터의 커널을 적용하여 출력 이미지를 생성한다.
- 출력 feature map의 개수는 필터의 개수와 동일하다.

![](https://velog.velcdn.com/images/heejookang/post/0ce9e414-1c82-4d8b-a260-246834f603cc/image.gif)

#### Kernel
- filter 안에 1 ~ n 개의 커널이 존재한다. 커널의 개수는 반드시 이미지의 채널 수와 동일해야 한다.
- Kernel Size는 가로 X 세로를 의미하며, 가로와 세로는 서로 다를 수 있지만 보통은 일치시킨다.
- Kernel Size가 크면 클 수록 입력 이미지에서 더 많은 feature 정보를 가져올 수 있지만,  
  큰 사이즈의 Kernel로 Convolution Backbone을 할 경우 훨씬 더 많은 연산량과 파라미터가 필요하다.

![](https://velog.velcdn.com/images/heejookang/post/91ca72d6-89ad-4068-b5a4-76b8e90f71ec/image.gif)

#### Stride
- 입력 이미지에 Convolution Filter를 적용할 때 Sliding Window가 이동하는 간격을 의미한다.
- 기본 stride는 1이지만, 2를 적용하면 입력 feature map 대비 출력 feature map의 크기가 절반정도 줄어든다.
- stride를 키우면 feature 정보를 손실할 가능성이 높아지지만,  
  오히려 불필요한 특성을 제거하는 효과를 가져올 수 있고 Convolution 연산 속도를 향상 시킨다.

![](https://velog.velcdn.com/images/heejookang/post/48dd14b8-4b85-445a-af68-53aad636823a/image.gif)

![](https://velog.velcdn.com/images/heejookang/post/16a97fae-edfb-4ea3-84f7-7e613f0edb28/image.gif)

#### Padding
- Filter를 적용하여 Convolution 수행 시 출력 Feature map 이 입력 feature map 대비 계속해서 작아지는 것을 막기위해 사용한다.
- Filter 적용 전, 입력 feature map의 상하좌우 끝에 각각 열과 행을 추가한 뒤, 0으로 채워서 크기를 증가시킨다.
- 출력 이미지의 크기를 입력 이미지의 크기와 동일하게 유지하기 위해서 직접 계산할 필요 없이 'same'이라는 값을 통해 입력 이미지의 크기와 동일하게 맞출 수 있다.

![](https://velog.velcdn.com/images/heejookang/post/5a209bbb-ae94-49e7-8376-0afc79be1f23/image.gif)

#### Pooling
- Convolution이 적용된 feature map의 일정 영역별로 하나의 값을 추출하여 feature map의 사이즈를 줄인다.
- 보통은 Convolution -> Relu activation -> Pooling 순서로 적용한다.
- 비슷한 feature들이 서로 다른 이미지에서 위치가 달라지면서 다르게 해석되는 현상을 중화시킬 수 있고,  
  feature map의 크기가 줄어들기 때문에, 연산 성능이 향상된다.
- Max Pooling과 Average Pooling이 있으며, Max Pooling은 중요도가 가장 높은 feature를 추출하고, Average Pooling은 전체를 버무려서 추출한다.

![](https://velog.velcdn.com/images/heejookang/post/7b9e4217-7596-4675-a086-7a198d4b90b6/image.gif)

#### 🚩정리
- stride를 증가시키는 것과 pooling을 적용하는 것은 출력 feature map의 크기를 줄이는데 사용하는 것이다.
- Convolution 연산을 진행하면서, feature map의 크기를 줄이면, 위치 변화에 따른 feature의 영향도도 줄어들기 떄문에 과적합을 방지할 수 있는 장점이 있다.
- Pooling의 경우 특정 위치의 feature 값이 손실되는 이슈등으로 인하여 최근 Advanced CNN에서는 많이 사용되지 않는다.
- Classifier에서는 Fully Connected Layer의 지나친 연결로 인해 많은 파라미터가 생성되므로 오히려 과적합이 발생할 수 있다.

![](https://velog.velcdn.com/images/heejookang/post/c1d7ed20-a824-4305-8f2b-36977b24676e/image.png)

- Dropout을 사용해서 Layer간 연결을 줄일 수 있으며 과적합을 방지할 수 있다.

![](https://velog.velcdn.com/images/heejookang/post/5c405537-5a46-4b8d-b49a-507f313fb990/image.png)


CNN 모델의 성능 개선과 과적합 개선 기법 

### CNN Performance
- CNN 모델을 제작할 때, 다양한 기법을 통해 성능 개선 및 과적합 개선이 가능하다.

#### Weight Initialization, 가중치 초기화
- 처음 가중치를 어떻게 줄 것인지를 정하는 방법이며, 처음 가중치를 어떻게 설정하느냐에 따라 모델의 성능이 크게 달라질 수 있다.
  
> 1. 사비에르 글로로트 초기화
> - 고정된 표준편차를 사용하지 않고, 이전 층의 노드 수에 맞게 현재 층의 가중치를 초기화한다.
> - 층마다 노드 개수를 다르게 설정하더라도 이에 맞게 가중치가 초기화되기 때문에 고정된 표준편차를 사용하는 것보다 이상치에 민감하지 않다.
> - 활성화 함수가 ReLU일 때, 층이 지날 수록 활성화 값이 고르지 못하게 되는 문제가 생겨서, 출력층에서만 사용한다.

![](https://velog.velcdn.com/images/heejookang/post/31d147f0-bcf7-49a5-92e3-167a65d69c17/image.png)
![](https://velog.velcdn.com/images/heejookang/post/74465db1-01d3-4ae7-b257-e02f7842b532/image.png)


> 2. 카이밍 히 초기화
> - 고정된 표준편차를 사용하지 않고, 이전 층의 노드 수에 맞게 현재 층의 가중치를 초기화한다.
> - 층마다 노드 개수를 다르게 설정하더라도 이에 맞게 가중치가 초기화되기 때문에 고정된 표준편차를 사용하는 것보다 이상치에 민감하지 않다.
> - 활성화 함수가 ReLU일 때, 추천하는 초기화 방법으로서, 층이 깊어지더라도 모든 활성값이 고르게 분포된다.
![](https://velog.velcdn.com/images/heejookang/post/317b16f5-e04d-4b18-a092-ad5f41510b75/image.png)


#### Batch Normalization, 배치 정규화
- 입력 데이터 간에 값의 차이가 발생하면, 가중치의 비중도 달라지기 때문에 층을 통과할 수록 편차가 심해진다.  
   이를 내부 공변량 이동(Internal Convariant Shift)이라고 한다.
- 가중치의 값의 비중이 달라지면, 특정 가중치에 중점을 두면서 경사 하강법이 진행되기 때문에,  
  모든 입력값을 표준 정규화하여 최적의 parameter를 보다 빠르게 학습할 수 있도록 해야한다.
- 가중치를 초기화할 때 민감도를 감소시키고, 학습 속도 증가시키며, 모델을 일반화하기 위해서 사용한다.

![](https://velog.velcdn.com/images/heejookang/post/971006a2-00c7-4691-982c-1bb488b3e467/image.png)

![](https://velog.velcdn.com/images/heejookang/post/8da7e5ff-bac1-42c6-8665-f8f4091d22cd/image.png)

- BN은 activation function 앞에 적용하면, weight 값은 평균이 0, 분산이 1인 상태로 정규분포가 된다.
- ReLU가 activation으로 적용되면 음수에 해당하는(절반 정도) 부분이 0이 된다.  
  이러한 문제를 해결하기 위해서 γ(감마)와 β(베타)를 사용해서 음수부분이 모두 0이 되는 것을 막아준다.

![](https://velog.velcdn.com/images/heejookang/post/5209a7f5-2d2a-43dd-88a6-6ddd574be2b4/image.png)

![](https://velog.velcdn.com/images/heejookang/post/6e689c24-1950-4525-ade3-d106b42c184a/image.png)

#### Batch Size
- batch size를 작게 하면, 적절한 noise가 생겨서 overfitting을 방지하게 되고, 모델의 성능을 향상시키는 계기가 될 수 있지만, 너무 작아서는 안된다.
- batch size를 너무 작게 하는 경우에는 batch당 sample 수가 작아져서 훈련 데이터를 학습하는 데에 부족할 수 있다.
- 따라서 굉장히 크게 주는 것 보다는 작게 주는 것이 좋으며, 이를 또 너무 작게 주어서는 안된다.  
  **논문에 따르면 8보다 크고 32보다 작게 주는 것이 효과적이라고 한다.**

![](https://velog.velcdn.com/images/heejookang/post/d2d33153-045f-4844-8e0b-e47496b27d95/image.png)
![](https://velog.velcdn.com/images/heejookang/post/744c231c-0383-4711-8399-94a59a7f2a5c/image.png)


#### Global Average Pooling
- 이전의 Pooling들은 면적을 줄이기 위해 사용했지만, Global Average Pooling은 면적을 없애고 채널 수 만큼 값을 나오게 한다.
- feature map의 가로 X 세로의 특정 영역을 Sub smapling하지 않고, 채널 별로 평균 값을 추출한다.
- feature map의 채널 수가 많을 경우(보통 512개 이상) 이를 적용하고, 채널 수가 적다면 Flatten을 적용한다.
- Flatten 후에 Classification Dense Layer로 이어지면서 많은 파라미터들로 인한 overfitting 유발 가능성 증대 및 학습 시간 증가로 이어지기 때문에  
  맨 마지막 feature map의 채널 수가 크다면 Global Average Pooling을 적용하는 것이 더 나을 수 있다.

![](https://velog.velcdn.com/images/heejookang/post/20ad5fa4-84f1-4113-b89f-27003ae88f5d/image.png)


#### Weight Regularization (가중치 규제), Weight Decay (가중치 감소)
- loss function은 loss 값이 작아지는 방향으로 가중치를 update한다.
- 하지만, loss를 줄이는 데에만 신경쓰게 되면, 특정 가중치가 너무 커지면서 오히려 나쁜 결과를 가져올 수 있다.
- 기존 가중치에 특정 연산을 수행하여 loss function의 출력 값과 더해주면 loss function의 결과를 어느정도 제어할 수 있게 된다.
- 보통 파라미터가 많은 Dense Layer에서 많이 사용되고 가중치 규제보다는 loss function에 규제를 걸어 가중치를 감소시키는 원리이다.
- kernel_regularizer 파라미터에서 l1, l2을 선택할 수 있다.

![](https://velog.velcdn.com/images/heejookang/post/4f62c232-8301-4f9a-826e-044c5f7bd44e/image.png)

## Pretrained Model

### VGGNet (옥스포드 대학의 연구팀)

- 2014년 ILSVRC에서 GoogleNet이 1위, VGG는 2위를 차지했다.
- GoogleNet의 오류율은 6.7%, VGG의 오류율은 7.3%이고, 0.6%차이밖에 나지 않았다.
- 간결하고 단순한 아키텍처임에도 불구하고 1위인 GoogleNet과 큰 차이 없는 성능을 보여서 주목을 받게 되었다.
- 네트워크 깊이에 따른 모델 성능의 영향에 대한 연구에 집중하여 만들어진 네트워크이다.
- 신경망을 깊게 만들 수록 성능이 좋아짐을 확인하였지만, 커널 사이즈가 클 수록 이미지 사이즈가 급격하게 축소되기 때문에, 더 깊은 층을 만들기 어렵고 파라미터 개수와 연산량도 더 많이 필요하다는 것을 알았다.
- 따라서 kernel 크기를 3X3으로 단일화했으며, Padding, Strides 값을 조정하여 단순한 네트워크로 구성되었다.
- 2개의 3X3 커널은 5X5 커널과 동일한 크기의 feature map을 생성하기 때문에 3X3 커널로 연산하면, 층을 더 만들 수 있게 된다.

![](https://velog.velcdn.com/images/heejookang/post/a70fe253-c4f7-431c-8bb0-0aebb8cb1643/image.png)

### Inception Network (GoogleNet)
- 여러 사이즈의 커널들을 한꺼번에 결합하는 방식을 사용하며, 이를 묶어서 inception module이라고 한다.
- 여러 개의 inception module을 연속적으로 이어서 구성하고 여러 사이즈의 필터들이 서로 다른 공간 기반으로 feature들을 추출한다.
- inception module을 결합하면서 보다 풍부한 Feature Extractor Layer를 구성하게 된다.
- 하지만 여러 사이즈의 커널을 결합하게 되면, Convolution 연산을 수행할 때 파라미터 수가 증가되고 과적합으로 이어진다.
- 이를 극복하고자 연산을 수행하기 전에 1X1 Convolution을 적용해서 파라미터 수를 획기적으로 감소시킨다.
- 1X1 Convolution을 적용하면 입력 데이터의 특징을 함축적으로 표현하면서 파라미터 수를 줄이는 차원 축소 역할을 수행하게 된다.

![](https://velog.velcdn.com/images/heejookang/post/be8a3643-3bbc-4741-8e20-2039a2fff416/image.png)
![](https://velog.velcdn.com/images/heejookang/post/f93c8c58-2baa-4d82-b696-132148ceda47/image.png)

#### 1X1 Convolution
- 행과 열의 크기 변환 없이 Channel의 수를 조절할 수 있고, weight 및 비선형성을 추가하는 역할을 한다.
- 행과 열의 사이즈를 줄이고 싶다면, Pooling을 사용하면 되고, 채널 수만 줄이고 싶다면 1X1 Convolution을 사용하면 된다.

![](https://velog.velcdn.com/images/heejookang/post/86200993-cf41-4ca8-82cc-2ee7b0b83663/image.png)

### ResNet (마이크로소프트)
- VGG 이후 더 깊은 Network에 대한 연구가 증가했지만, Network 깊이가 깊어질 수록 오히려 accuracy가 떨어지는 문제가 있었다.
- 층이 깊어질 수록 계속해서 기울기가 0에 가까워지는 Grdient vanishing이 발생하기 때문이다.

![](https://velog.velcdn.com/images/heejookang/post/9d7de67a-a758-47a1-9d8a-32a69990ff80/image.png)

- 이를 해결하고자 층을 만들되, Input 데이터와 결과가 동일하게 나올 수 있도록 하는 층을 연구하기 시작했다.
- 함수로 나타내면 H(x) = x이다.
- 하지만 활성화 함수를 통과한 값을 기존 Input 데이터와 동일하게 만드는 것은 굉장히 복잡했기 때문에 H(x) = F(x) + x 즉, F(x)를 0으로 만드는 F(x)에 포커스를 하게 된다.
- input은 x이고, Model인 F(x)라는 일련의 과정을 거치면서 자신인 x가 더해져서 output으로 F(x) + x가 나오는 구조가 된다.

![](https://velog.velcdn.com/images/heejookang/post/7dc71352-c738-4cc7-809d-bc3b0b42eca6/image.png)

![](https://velog.velcdn.com/images/heejookang/post/d61bdc5c-b824-474a-b30a-f3db92ef0e07/image.png)


### Transfer Learning, 전이 학습
- 이미지 분류 문제를 해결하는 데에 사용했던 모델을 다른 데이터세트 혹은 다른 문제에 적용시켜 해결하는 것을 의미한다.
- 즉, 사전에 학습된 모델을 다른 작업에 이용하는 것을 의미한다.
- Pretrained Model의 Convolutional Base 구조(Conv2D + Pooling)를 그대로 두고 분류기(FC)를 붙여서 학습시킨다.

![](https://velog.velcdn.com/images/heejookang/post/10c38309-c368-44ba-a6ce-5f83767c5556/image.png)
![](https://velog.velcdn.com/images/heejookang/post/c37b16dd-f826-4efc-afab-05af138298dd/image.png)

- 사전 학습된 모델의 용도를 변경하기 위한 층별 미세조정(fine tuning)은 데이터 세트의 크기와 윳성을 기반으로 고민하여 조정한다.
- 2018년 FAIR(Facebook AI Research)논문에서 실험을 통해 '전이학습이 학습 속도 면에서 효과가 있다.'라는 것을 밝혀냈다.

![](https://velog.velcdn.com/images/heejookang/post/93dcea31-1b41-44d4-b63d-7437f18e49fe/image.png)

