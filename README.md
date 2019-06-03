***Deep Learning 2019, PNU-CSE***
# **Yeast Protein-Protein Interaction Graph Edge Prediction using Matrix Factorization and Graph Convolution Network**

Graph Convolution Network(GCN)는 그래프를 입력으로 받는 Deep learning 모델이다. Yeast PPI 데이터셋을 사용하여 기존에 수행된 연구에서는 단순히 edge의 연결 정보만으로 추가 edge를 추론하였다. 우리는 예측 성능을 좀 더 높이기 위해 몇 가지 방법을 추가하였다.

## 1. Sub Graph Sampling

![](https://github.com/leekh7411/leekh7411.github.io/blob/master/assets/subgraphs.png?raw=true)
하나의 거대한 Graph를 일정한 노드 수를 갖는 Sub Graph들로 분해 하여 edge prediction을 수행한다. 우리는 ForestFire 알고리즘을 사용하여 아래 그림과 같이 거대한 Graph를 여러개의 Sub-Graph로 샘플링 하였다 

## 2. Matrix Factorization
<img src="https://d3ansictanv2wj.cloudfront.net/mf_matrix-c1c20c1013a11279a8defe10e3e05a4b.png" width="40%">

추천 시스템에서 많이 사용되는 방법으로 어떤 (M,N) 크기의 행렬에서 존재하는 일부 데이터만 사용하여 나머지 0 값들을 채워 넣는 알고리즘이다. 해당 과제에서 수행하고자 하는 edge prediction은 label edge들에서 학습용을 제외한 나머지를 모두 제거하고 이를 예측하는 작업을 학습하게 된다. 따라서 Matrix Factorization으로 Sub-Graph의 Adjacency Matrix 성분을 분해하여 Graph Convolution 연산에 추가함으로서 prediction 성능 향상을 목표로 한다

## 3. Graph Convolution Network
Interaction관계를 하나의 그래프로 볼 수 있기 때문에 기본적으로 edge prediction을 위한 데이터셋은 그래프 데이터셋이다. 따라서 최근 Graph dataset에 적합한 Deep learning 모델로 사용되는 Graph Convolution 을 기반으로 모델을 설계한다

## Results
실험중에 있다.


## References
- Graph Sampling Package - https://github.com/Ashish7129/Graph_Sampling
- Yeast PPI - [SNAP(Stanford Network Analysis Project) - GCN example with Yeast PPI](http://snap.stanford.edu/deepnetbio-ismb/ipynb/Graph+Convolutional+Prediction+of+Protein+Interactions+in+Yeast.html)