import os
import argparse
import numpy as np
from model.ripple_net import RippleNet
from model.my_model import my_RippleNet

np.random.seed(555)

base_path = os.getcwd()
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='movie', help='which dataset to use')
parser.add_argument('--dim', type=int, default=18, help='dimension of entity and relation embeddings')
parser.add_argument('--n_hop', type=int, default=3, help='maximum hops')
parser.add_argument('--kge_weight', type=float, default=0.00001, help='weight of the KGE term')
parser.add_argument('--l2_weight', type=float, default=1e-6, help='weight of the l2 regularization term')
parser.add_argument('--lr', type=float, default=0.002 , help='learning rate')
parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
parser.add_argument('--n_epoch', type=int, default=30, help='the number of epochs')
parser.add_argument('--n_memory', type=int, default=16, help='size of ripple set for each hop')
parser.add_argument('--patience', type=int, default=10, help='early stop patience')
parser.add_argument('--item_update_mode', type=str, default='plus_transform',help='how to update item at the end of each hop')

# parser.add_argument('--item_update_mode', type=str, default='plus',help='how to update item at the end of each hop')
parser.add_argument('--using_all_hops', type=bool, default=True,
                    help='whether using outputs of all hops or just the last hop when making prediction')
parser.add_argument('--base_path', type=str, default=base_path, help='base work dir')

args = parser.parse_args()

"""
原始的训练模型
"""
# ripple_net = RippleNet(args)
# ripple_net.train()
# ripple_net.evaluate()
# ripple_net.predict()

# 我的训练数据
ripple_net=my_RippleNet(args)
ripple_net.train()
ripple_net.evaluate()

# result,pred,y_true=ripple_net.predict()
# pred=np.array(pred)
# result=np.array(result)
# y_true=np.array(y_true)
#
# true_idx=(y_true==1)
#
# prob_true=result[true_idx]
# pred_true=pred[true_idx]
# label_true=y_true[true_idx]
#
# final_array=np.array([prob_true,pred_true,label_true])
# print(final_array)
#
#
# true_idx=(result==1)
# prob_true=result[true_idx]
# pred_true=pred[true_idx]
# label_true=y_true[true_idx]
#
# final_array=np.array([prob_true,pred_true,label_true])
# print(final_array)

print(parser.parse_args())