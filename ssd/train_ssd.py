
import mxnet as mx
import time
import random
import os
import platform, subprocess, sys, os
import argparse
import cv2



def train():
    print("Training SSD")

def parse_args():
    """Parse arguments."""
    parser = argparse.ArgumentParser(
        prog='train',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Train Classifier or Single-shot detection network')
    parser.add_argument('--train', default=['classifier', 'detector', 'ssd'], type=str, 
                        help='Train the network [classifier|detector]')
    parser.add_argument('--data-dir', dest='data_dir', 
                        help='data directory to use', default=os.path.join(os.getcwd(), 'data'), type=str)
    parser.add_argument('--batch-size', type=int, default=64,
                        help='batch size for training and testing (default: 100)')
    parser.add_argument('--epochs', type=int, default=10,
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.1,
                        help='learning rate (default: 0.1)')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--gpu',  default=False,
                        help='Train on GPU with CUDA')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()

    if args.train == 'ssd':
        epochs = args.epochs
        batch_size = args.batch_size
        lr = args.lr
        momentum = args.momentum
        data_dir = os.path.abspath(args.data_dir)
        print(args)
        print(data_dir)
        print('----- Hyperparameters -----')
        print('epochs     : %s' %(epochs))
        print('batch_size : %s' %(batch_size))
        print('lr         : %s' %(lr))
        print('momentum   : %s' %(momentum))
        
        train(data_dir)