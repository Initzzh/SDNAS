import argparse

def add(args):
     r = args.x + args.y
     print('x + y = ', r)

def sub(args):
     r = args.x - args
     print('x - y = ', r)

parser = argparse.ArgumentParser(prog='PROG')
subparsers = parser.add_subparsers(help='sub-command help')
#添加子命令 add
parser_a = subparsers.add_parser('add', help='add help')
parser_a.add_argument('-x', type=int, help='x value')
parser_a.add_argument('-y', type=int, help='y value')
#设置默认函数
parser_a.set_defaults(func=add)
#添加子命令 sub
parser_s = subparsers.add_parser('sub', help='sub help')
parser_s.add_argument('-x', type=int, help='x value')
parser_s.add_argument('-y', type=int, help='y value')
#设置默认函数
parser_s.set_defaults(func=sub)
args = parser.parse_args()
#执行函数功能
args.func(args)