import mxnet as mx
import logging
logging.getLogger().setLevel(logging.INFO)

X=mx.nd.random_normal(shape=(100,10),ctx=mx.cpu())
Y=mx.nd.ones((100),ctx=mx.cpu())
dataset = mx.gluon.data.dataset.ArrayDataset(X, Y)
gluon_data_loader = mx.gluon.data.DataLoader(dataset, batch_size=10)

class SimpleIter(object):
    def __init__(self, gluon_data_loader):
        self.gluon_data_loader=gluon_data_loader
        self.gluon_data_loader_iter=iter(self.gluon_data_loader)
        
        data,label=next(self.gluon_data_loader_iter)
        data_desc = mx.io.DataDesc(name='data', shape=data.shape, dtype=data.dtype)
        label_desc = mx.io.DataDesc(name='softmax_label', shape=label.shape, dtype=label.dtype)
        
        self.gluon_data_loader_iter=iter(self.gluon_data_loader)
        
        self._provide_data = [data_desc]
        self._provide_label = [label_desc]

    def __iter__(self):
        return self

    def reset(self):
        self.gluon_data_loader_iter=iter(self.gluon_data_loader)

    def __next__(self):
        return self.next()

    @property
    def provide_data(self):
        return self._provide_data

    @property
    def provide_label(self):
        return self._provide_label

    def next(self):
        data,label=next(self.gluon_data_loader_iter)
        batch = mx.io.DataBatch(data=[data], label=[label], provide_data=self._provide_data,
                                provide_label=self._provide_label)
        return batch


net = mx.sym.Variable('data')
net = mx.sym.FullyConnected(net, name='fc1', num_hidden=64)
net = mx.sym.Activation(net, name='relu1', act_type="relu")
net = mx.sym.FullyConnected(net, name='fc2', num_hidden=2)
net = mx.sym.SoftmaxOutput(net, name='softmax')

mod = mx.mod.Module(symbol=net,
                    context=mx.cpu(),
                    data_names=['data'],
                    label_names=['softmax_label'])

train_iter=SimpleIter(gluon_data_loader)


# fit the module
mod.fit(train_iter,
        eval_data=train_iter, # set eval = train_iter ~
        optimizer='sgd',
        optimizer_params={'learning_rate':0.1},
        eval_metric='acc',
        num_epoch=8)