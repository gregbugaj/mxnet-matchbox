from __future__ import print_function
import mxnet as mx
from mxnet import nd, autograd
from mxnet import gluon
from mxnet import symbol as sym
import sys
import os

def load_checkpoint(dir, prefix, epoch):
    """@ref https://beta.mxnet.io/_modules/mxnet/model.html#load_checkpoint"""
    symbol = sym.load(os.path.join(dir, '%s-symbol.json' % prefix))
    save_dict = nd.load(os.path.join(dir, '%s-%04d.params' % (prefix, epoch)))
    arg_params = {}
    aux_params = {}
    for k, v in save_dict.items():
        print(k)
        tp, name = k.split(':', 1)
        if tp == 'arg':
            arg_params[name] = v
        if tp == 'aux':
            aux_params[name] = v
    return (symbol, arg_params, aux_params)


def main():
    # dir_name = 'lenet'
    dir_name = 'py-mlp'

    if not os.path.exists(dir_name):
        raise Exception("Directory not valid")
    (symbol, arg_params, aux_params) = load_checkpoint(dir_name, "lenet", 1)

    print(aux_params)
    print(arg_params)

    #prefix = 'lenet'
    #epoch = 0
    #sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, epoch)


if __name__ == "__main__":
    main()