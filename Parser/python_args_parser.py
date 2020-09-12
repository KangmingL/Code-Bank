"""
Python arguments parser
"""
import argparse

parser = argparse.ArgumentParser(description='code')
parser.add_argument('--data_path', type=str, default=None, help='path to dataset')
parser.add_argument('--seed', type=int, default=1, help='random seed')
parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
parser.add_argument('--momentum', type=float, default=0.01, help='momentum')
parser.add_argument('--epochs', type=int, default=10, help='epochs')
args = parser.parse_args()

# Usage:
# ex: args.seed