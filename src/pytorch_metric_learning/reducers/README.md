# Reducer
Reducer规定了如何从许多loss到单一loss值的方法。

## 流程解释
ContrastiveLoss会计算出一个批次中每个正负pair的loss。一个Reducer将把所有这些每个pair的`loss`还原成一个`scalar`。下面展示了Reducer在这个库的处理和计算流程中的位置:

```
数据集中的数据 --> Sampler --> Miner --> Loss --> Reducer --> 最终的 loss 值
```

## 例子 & 使用方法:
```
from pytorch_metric_learning import losses, reducers
reducer = reducers.SomeReducer()
loss_func = losses.SomeLoss(reducer=reducer)
loss = loss_func(embeddings, labels) # in your training for-loop
```

在`Reducer`内部实现中，`losses`会创建一个包含损失和其他信息的`diction`。 \
`Reducer`接受这个字典, 然后进行降维, 并返回一个`Scalar`, 在这个值上可以直接调用`.backward()`方法进行训练. \
大多数的`Reducer`都可以被传递到任何`losses`中。

- 分别有
    - DoNothingReducer 默认啥也不干的Reducer
    - MeanReducer 直接求平均的Reducer
    - AvgNonZeroReducer 将所有的非负数求平均降维
    - ThresholdReducer 只使用指定阀值内的值进行求平均
    - ClassWeightedReducer 根据类进行加权求和
    - DivisorReducer 将所有的loss除以一个值
    - MultipleReducers 分别对不同的loss进行指定


## 文档:
- `BaseReducer` \
    所有的`Reducer`都是这个类的子类.
    ```
    reducers.BaseReducer(collect_stats=True)
    ```
    参数:
    - collect_stats: (boolean) \
        如果是True的话，将收集实验中可能有用的各种统计数据。 \
        如果为False，将跳过这些计算。


- `DoNothingReducer` \
    这将返回其输入。换句话说，就是啥也不干。输出将是传入的`loss dict`。
    ```
    reducers.DoNothingReducer(**kwargs)
    ```

- `MeanReducer` \
    这将返回`loss`的平均值。
    ```
    reducers.MeanReducer(**kwargs)
    ```

- `AvgNonZeroReducer` \
    只使用所有的非负数求平均降维, 等同于`ThresholdReducer(low=0)`
    ```
    reducers.AvgNonZeroReducer(**kwargs)
    ```

- `ThresholdReducer` \
    计算指定范围内的loss值的平均loss.
    ```
    reducers.ThresholdReducer(low=None, high=None **kwargs)
    ```
    参数:
    - `low`:  (float) \
        小于此值的`loss`将被忽略。
    - `high`: (float) \
        大于此值的`loss`将被忽略。

    两个参数至少要指定一个.




