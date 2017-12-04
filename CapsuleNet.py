import mxnet as mx
from mxnet import gluon, nd, autograd, image, init
from mxnet.gluon import nn
import time

def load_data_mnist(batch_size, resize=None):
    def transform_mnist(data, label):
        if resize:
            data = image.imresize(data, resize, resize)
        return nd.transpose(data.astype('float32'), (2,0,1))/255, label.astype('float32')
    mnist_train = gluon.data.vision.MNIST(root='./data', train=True, transform=transform_mnist)
    mnist_test = gluon.data.vision.MNIST(root='./data', train=False, transform=transform_mnist)
    train_data = gluon.data.DataLoader(mnist_train, batch_size, shuffle=True)
    test_data = gluon.data.DataLoader(mnist_test, batch_size, shuffle=False)
    return train_data, test_data
	
class PrimaryCap(nn.Block):
    def __init__(self, k_size, cap_channels, len_vectors, strides, **kwargs):
        super(PrimaryCap, self).__init__(**kwargs)
        self.k = k_size
        self.c = cap_channels
        self.l = len_vectors
        self.s = strides
        self.net = nn.Sequential()
        with self.name_scope():
            for _ in range(self.c):
                self.net.add(nn.Conv2D(channels=self.l, kernel_size=self.k, strides=self.s))
    def forward(self, x):
        out = []
        for i, net in enumerate(self.net):
            out.append(nd.reshape(net(x),(0,0,-1,1)))
        return Squash(nd.expand_dims(nd.concat(*out, dim=2),axis=4),axis=1)
		
def Squash(vector, axis):
    norm = nd.sum(nd.square(vector), axis, keepdims=True)
    v_j = norm/(1+norm)/nd.sqrt(norm, keepdims=True)*vector
    return v_j

class CapsuleLayer(nn.Block):
    def __init__(self, len_vectors_input, len_vectors_output, batch_size, num_input, num_output, num_routing, **kwargs):
        super(CapsuleLayer, self).__init__(**kwargs)
        self.bs = batch_size
        self.lvi = len_vectors_input
        self.lvo = len_vectors_output
        self.ni = num_input
        self.no = num_output
        self.nr = num_routing
        with self.name_scope():
            self.W = self.params.get('weight', shape=(1, self.lvi, self.ni, self.lvo, self.no), init=init.Normal(0.5))
    def forward(self, x):
        #x.shape: (batchsize, 8, 1152, 1, 1)
        #routing_weight.shape: (1, 1, 1152, 1, 10)
        routing_weight = nd.softmax(nd.zeros(shape=(1, 1, self.ni, 1, self.no), ctx=x.context),axis=1)
        
        #u.shape: (batchsize, 1, 1152, 16, 10)
        u = nd.sum(x*self.W.data(), axis=1, keepdims=True)
        
        #s.shape: (batchsize, 1, 1, 16, 10)
        s = nd.sum(u*routing_weight, axis=2, keepdims=True)
        
        #v.shape: (batchsize, 1, 1, 16, 10)
        v = Squash(s, axis=3)
        
        for i in range(self.nr):
            
            routing_weight = routing_weight + nd.sum(u*v, axis=3, keepdims=True)
            c = nd.softmax(routing_weight, axis=2)
            s = nd.sum(u*c, axis=2, keepdims=True)
            v = Squash(s, axis=3)
        
        return nd.reshape(v,shape=(-1, self.lvo, self.no))
	
class length(nn.Block):
    def __init__(self, axis=1, **kwargs):
        super(length, self).__init__(**kwargs)
        self.axis = axis
    def forward(self, x):
        out = nd.sqrt(nd.sum(nd.square(x), self.axis))
        return out

def createnet(batch_size=2, ctx=mx.cpu()):
    CapNet = nn.Sequential()
    with CapNet.name_scope():
        CapNet.add(nn.Conv2D(channels=256, kernel_size=9, strides=1, padding=(0,0), activation='relu'))
        CapNet.add(PrimaryCap(k_size=9, cap_channels=32, len_vectors=8, strides=2))
        CapNet.add(CapsuleLayer(len_vectors_input=8,len_vectors_output=16,batch_size=batch_size, num_input=1152, num_output=10, num_routing=3))
        CapNet.add(length())
    CapNet.initialize(ctx=ctx)
    return CapNet

def loss(y_pred, y_true):
    L = y_true * nd.square(nd.maximum(0., 0.9 - y_pred)) + 0.5 * (1 - y_true) * nd.square(nd.maximum(0., y_pred - 0.1))
    return nd.mean(nd.sum(L, 1))

def accuracy(output, label):
    return nd.mean(nd.argmax(output, axis=1) == label).asscalar()

def _get_batch(batch, ctx):
    if isinstance(batch, mx.io.DataBatch):
        data = batch.data[0]
        label = batch.label[0]
    else:
        data, label = batch
    return data.as_in_context(ctx), label.as_in_context(ctx)

def evaluate_accuracy(data_iterator, net, ctx=mx.cpu()):
    acc = 0.
    if isinstance(data_iterator, mx.io.MXDataIter):
        data_iterator.reset()
    for i, batch in enumerate(data_iterator):
        data, label = _get_batch(batch,ctx)
        output = net(data)
        acc += accuracy(output, label)
    return acc/(i+1)

def train(train_data, test_data, net, loss, trainer, ctx, num_epochs, print_batches=None):
    for epoch in range(num_epochs):
        train_loss = 0
        train_acc = 0
        n = 0
        Loss = []
        for i, batch in enumerate(train_data):
            tic = time.time()
            data, label = batch
            one_hot_label = nd.one_hot(label, 10)
            with autograd.record():
                output = net(data.as_in_context(ctx))
                L = loss(output, one_hot_label.as_in_context(ctx))
            L.backward()
            trainer.step(data.shape[0], ignore_stale_grad=True)
            train_loss += nd.mean(L).asscalar()
            print('Batch: %d | Time: %.3f. | mean L: %f'%(i, time.time()-tic, nd.mean(L).asscalar()))
            Loss.append(nd.mean(L).asscalar())
            train_acc += accuracy(output, label.as_in_context(ctx))
#             net.save_params('./capsulenet_%d.params'%(i))
            n = i + 1
            if print_batches and n%print_batches == 0:
                print('Batch %d | Loss: %f | Train acc: %f'%(n, train_loss/n, train_acc/n))
        test_acc = evaluate_accuracy(test_data, net, ctx)
        print('Epoch %d | Loss: %f | Train acc: %f | Test acc: %f'%(epoch, train_loss/n, train_acc/n, test_acc))

if __name__ == "__main__":
    ctx = mx.gpu(0)
    batch_size = 20
    num_epochs = 20
    train_data, test_data = load_data_mnist(batch_size, resize=28)
    net = createnet(batch_size, ctx)
#     print(net)
    if True:
        print('================Train==================')
        
        trainer = gluon.Trainer(net.collect_params(), 'adam', {'learning_rate': 0.01})
        train(train_data, test_data, net, loss, trainer, ctx, num_epochs, print_batches=200)
        net.save_params('./capsulenet_%d.params'%(num_epochs))
