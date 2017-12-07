# Capsule_gluon
## capsule生成函数
 PrimaryCap的生成非常类似group convolution  
 len_vector: PrimaryCap生成的向量长度  
 cap_channels: 向量通道数  
## CapsuleLayer
CapsuleLayer运算并不复杂，最重要的是要对向量进行对齐  
len_vectors_input: 输入向量的长度  
len_vectors_output: 输出向量长度  
num_input: capsule的输入个数，这个参数可由上一层输出得到，例：(6\*6\*32) = 1152  
num_output: capsule的输出个数  
num_routing: 路由迭代次数  
### Dynamic routing
动态路由并不会根据反向传播更新，而是根据向量间的相似性来更新。  
由于输出向量是由输入向量加权求得，如果输出向量与输入向量中的某些向量相似(方向一致)，那么他们的向量积越大，反之越小。  


```python
def createnet(batch_size=2, ctx=mx.cpu()):
    CapNet = nn.Sequential()
    with CapNet.name_scope():
        CapNet.add(nn.Conv2D(channels=64, kernel_size=9, strides=1, padding=(0,0), activation='relu'))
        CapNet.add(PrimaryCap(k_size=9, cap_channels=32, len_vectors=8, strides=2))
        CapNet.add(CapsuleLayer(len_vectors_input=8,len_vectors_output=16,batch_size=batch_size, num_input=1152, num_output=10, num_routing=3))
        CapNet.add(length())
    CapNet.initialize(ctx=ctx)
    return CapNet
```

## 损失计算
由于capsulenet输出的是向量，因此不能简单的使用softmaxcrossentry来计算损失，这里用的是Margin loss.


```python
def loss(y_pred, y_true):
    L = y_true * nd.square(nd.maximum(0., 0.9 - y_pred)) + 0.5 * (1 - y_true) * nd.square(nd.maximum(0., y_pred - 0.1))
    return nd.mean(nd.sum(L, 1))
```

## 主函数


```python
if __name__ == "__main__":
    ctx = mx.cpu(0)
    Train = True
    batch_size = 2
    num_epochs = 20
    train_data, test_data = utils.load_data_mnist(batch_size, resize=28)
    net = createnet(batch_size, ctx)
#     print(net)
    if Train:
        print('================Train==================')
        
        trainer = gluon.Trainer(net.collect_params(), 'adam', {'learning_rate': 0.01})
        utils.traincaps(train_data, test_data, net, loss, trainer, ctx, num_epochs, print_batches=100)
        net.save_params('capsulenet_%d.params'%(num_epochs))
    else:
        net.load_params('capsulenet_%d.params'%(num_epochs), ctx=ctx)
```


```python

```
