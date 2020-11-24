# Distances

Distance类计算input embeddings之间的成对距离/相似度。

根据 [distance documentation](https://kevinmusgrave.github.io/pytorch-metric-learning/distances/) 里面的介绍, 可以使用这里的distance来替换metric learning中的loss使用的distance衡量方法.

## 例子:
1. TripletMarginLoss 使用 squared L2 distance
```
from pytorch_metric_learning.distances import LpDistance
loss_func = TripletMarginLoss(margin=0.2, distance=LpDistance(power=2))
```
2. TripletMarginLoss 使用 unnormalized L1 distance
```
loss_func = TripletMarginLoss(margin=0.2, distance=LpDistance(normalize_embeddings=False, p=1))
```
3. TripletMarginLoss 使用 signal-to-noise ratio
```
from pytorch_metric_learning.distances import SNRDistance
loss_func = TripletMarginLoss(margin=0.2, distance=SNRDistance())
```
4. TripletMarginLoss 使用 cosine similarity (会进行一些自适应操作)
```
from pytorch_metric_learning.distances import CosineSimilarity
loss_func = TripletMarginLoss(margin=0.2, distance=CosineSimilarity())
```

- 分别有
    - Lp distance  正则距离 (L1, L2,...,Lp距离)
    - SNR distance 通过信噪比来衡量的距离.
    - dot product similarity 点乘相似度
    - cosine similarity cos相似度 (和dot的区别在于先进行了正则化)

## 实现:

- `BaseDistance` \
    所有的distance都继承于 `BaseDistance` 类, 都会调用它的__init__函数.
    ```
    distances.BaseDistance(collect_stats = True,
                             normalize_embeddings=True, 
                             p=2, 
                             power=1, 
                             is_inverted=False)
    ```
    参数: 
    - `collect_stats`:  (boolean) \
        如果是True的话，将收集实验中可能有用的各种统计数据。 \
        如果为False，将跳过这些计算。
    - `normalize_embeddings`: (boolean) \
        如果是True的话，在计算 距离/相似度 之前，嵌入的特征向量长度将被使用Lp规范归一化为1. \
        (在计算 cos类的loss时候可能会用到)
    - `p`: (int) \
        距离计算的范数.
    - `power`: (int) \
        如果不是 1, 则 距离/相似度矩阵 中的每个元素都将提高到这个幂。
    - `is_inverted`: (boolean) \
        应该被子类设置: \
        如果为False，那么小的值代表彼此相近的嵌入.
        如果为True，则大的值代表彼此相似的嵌入.
    
    在实现的时候`必须要实现`的部分:
    1. `compute_mat`(self, query_emb, ref_emb)
    ```
        # Must return a matrix where mat[j,k] represents 
        # the distance/similarity between query_emb[j] and ref_emb[k]
        def compute_mat(self, query_emb, ref_emb):
            raise NotImplementedError
    ```

    2. `pairwise_distance`(self, query_emb, ref_emb)
    ```
        # Must return a tensor where output[j] represents 
        # the distance/similarity between query_emb[j] and ref_emb[j]
        def pairwise_distance(self, query_emb, ref_emb):
            raise NotImplementedError
    ```

- `CosineSimilarity` \
    使用cos相似度来对距离进行衡量. \
    返回的`mat[i,j]`是`query_emb[i]`和`ref_emb[j]`之间的余弦相似度。
    ```
    distances.CosineSimilarity(**kwargs)
    ```
    参数: 
    - 除了`is_inverted`以外和上面的一样, 下面的也是.

- `DotProductSimilarity` \
    点乘相似度 \
    返回的`mat[i,j]`等于`torch.sum(query_emb[i] * ref_emb[j])`
    ```
    distances.DotProductSimilarity(**kwargs)
    ```

- `LpDistance` \
    点乘相似度 \
    返回的`mat[i,j]`等于`torch.sum(query_emb[i] * ref_emb[j])`
    ```
    distances.DotProductSimilarity(**kwargs)
    ```
    返回的`mat[i,j]`是`query_emb[i]`和`ref_emb[j]`之间的Lp距离。在默认参数下，这是欧氏距离。

- `SNRDistance` \
    [通过信噪比来衡量的距离](http://openaccess.thecvf.com/content_CVPR_2019/papers/Yuan_Signal-To-Noise_Ratio_A_Robust_Distance_Metric_for_Deep_Metric_Learning_CVPR_2019_paper.pdf)
    ```
    distances.DotProductSimilarity(**kwargs)
    ```
    返回的`mat[i,j]`等于:
    ```
    torch.var(query_emb[i] - ref_emb[j]) / torch.var(query_emb[i])
    ```






    
