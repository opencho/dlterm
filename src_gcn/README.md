***SubGraph Sampling for GCN***
# **Sub Graph Sampling**
Graph Convolution Network(GCN)는 그래프를 입력으로 받는 Deep learning 모델이다. 여기서 사용하려고 하는 그래프는 Node와 Edge의 수가 너무 많아 한번에 입력으로 처리하기 힘들다. 그리고 batch training의 장점을 살리기 위해서는 Graph dataset에서 일정 Node 수에 따른 Sampling 작업이 필요하다.
이를 위해 우리는 (https://github.com/Ashish7129/Graph_Sampling)의 Graph Sampling 알고리즘 패키지를 사용하였다

![](https://github.com/leekh7411/leekh7411.github.io/blob/master/assets/subgraphs.png?raw=true)

위 그림은 ForestFire 알고리즘으로 Yeast Protein Protein Interaction Network를 Sub Graph로 Sampling을 수행 한 결과이다. PPI의 특성상 주변 Node 와의 상관 관계를 중요하게 봐야하므로 해당 알고리즘을 선택하였다. 동시에 ForestFire 알고리즘은 수행 속도가 빠르기 때문에 거대한 그래프에 대해서 Sampling 작업을 빠르게 수행 할 수 있다

## About the Raw Data
`data/` 폴더에 있는 Yeast Network를 위한 edge list 데이터를 사용하여 Python의 networkX 패키지로 Graph 데이터를 간단히 불러올 수 있다.  

### Yeast Node Sequence Download
`y2seq_download.py` 에서는 앞서 소개된 Graph의 Node의 feature를 추출하기 위해 각 Node가 의미하는 Protein ID를 바탕으로 필요한 Protein Sequence를 다운로드 받는다.  개별 sequence를 하나씩 다운로드 하기 때문에 약 4시간 정도 소요된다

### Yeast Sub Graph Sampling & Feature Preprocessing
`y2sg_preprocessing.py` 에서는 앞서 다운로드 한 Node sequence들을 일정한 길이의 feature vector로 전처리 한다. 추가로 ForestFire 알고리즘을 사용한 Sub Graph Sampling과 Adjacency matrix 추출 작업들을 함께 수행한다. Feature preprocessing 작업이 꽤 오래 걸리며 저장되는 데이터셋의 사이즈가 꽤 크기 때문에 저장공간을 고려하고 실행해야 한다. 200개 Node에서 30000개 Sampling을 수행한 경우 약 19G 정도 된다

## References
- Graph Sampling Package - https://github.com/Ashish7129/Graph_Sampling
- Yeast PPI - [SNAP(Stanford Network Analysis Project) - GCN example with Yeast PPI](http://snap.stanford.edu/deepnetbio-ismb/ipynb/Graph+Convolutional+Prediction+of+Protein+Interactions+in+Yeast.html)