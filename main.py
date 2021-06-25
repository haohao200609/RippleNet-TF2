import os
import argparse
import numpy as np
from model.ripple_net import RippleNet
from model.my_model import my_RippleNet
from itertools import product


np.random.seed(555)



"""
best para

batch_size=1024, dataset='movie', dim=18, item_update_mode='plus_transform_drop_out', kge_weight=0.001, l2_weight=1e-05, lr=0.01, n_epoch=30, n_hop=9, n_memory=16, patience=10, using_all_hops=True
"""

para_dict={
    'dim':[16],
    'n_hop':[2,5],
    'kge_weight':[0.0001],
    'l2_weight':[1e-6],
    'lr':[0.01,0.002],
    'n_epoch':[20,40],
    'n_memory':[16]

}
def generate_conf(confs):
    """
    generate parameters from dict
    :param confs:
    confs = {
        'param1':[1,2, ..., n1],
        ...
        'param9':[1,2, ..., n9]
    }
    :yield: conf = {
        'param1': x1,
        ...
        'param9': x9
    }
    """
    for conf in product(*confs.values()):
        yield {k:v for k,v in zip(confs.keys(),conf)}

def get_para_parse(cur_para):
    base_path = os.getcwd()
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='movie', help='which dataset to use')
    parser.add_argument('--dim', type=int, default=cur_para['dim'], help='dimension of entity and relation embeddings')
    parser.add_argument('--n_hop', type=int, default=cur_para['n_hop'], help='maximum hops')
    parser.add_argument('--kge_weight', type=float, default=cur_para['kge_weight'], help='weight of the KGE term')
    parser.add_argument('--l2_weight', type=float, default=cur_para['l2_weight'], help='weight of the l2 regularization term')
    parser.add_argument('--lr', type=float, default=cur_para['lr'], help='learning rate')
    parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
    parser.add_argument('--n_epoch', type=int, default=cur_para['n_epoch'], help='the number of epochs')
    parser.add_argument('--n_memory', type=int, default=cur_para['n_memory'], help='size of ripple set for each hop')
    parser.add_argument('--patience', type=int, default=10, help='early stop patience')
    # parser.add_argument('--item_update_mode', type=str, default='plus_transform',help='how to update item at the end of each hop')

    parser.add_argument('--item_update_mode', type=str, default='plus_transform_drop_out',
                        help='how to update item at the end of each hop')
    # parser.add_argument('--item_update_mode', type=str, default='plus',help='how to update item at the end of each hop')
    parser.add_argument('--using_all_hops', type=bool, default=True,
                        help='whether using outputs of all hops or just the last hop when making prediction')
    parser.add_argument('--base_path', type=str, default=base_path, help='base work dir')

    args = parser.parse_args()

    return args

# base_path = os.getcwd()
# parser = argparse.ArgumentParser()
# parser.add_argument('--dataset', type=str, default='movie', help='which dataset to use')
# parser.add_argument('--dim', type=int, default=18, help='dimension of entity and relation embeddings')
# parser.add_argument('--n_hop', type=int, default=9, help='maximum hops')
# parser.add_argument('--kge_weight', type=float, default=0.001, help='weight of the KGE term')
# parser.add_argument('--l2_weight', type=float, default=1e-5, help='weight of the l2 regularization term')
# parser.add_argument('--lr', type=float, default=0.01 , help='learning rate')
# parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
# parser.add_argument('--n_epoch', type=int, default=30, help='the number of epochs')
# parser.add_argument('--n_memory', type=int, default=16, help='size of ripple set for each hop')
# parser.add_argument('--patience', type=int, default=10, help='early stop patience')
# # parser.add_argument('--item_update_mode', type=str, default='plus_transform',help='how to update item at the end of each hop')
#
# parser.add_argument('--item_update_mode', type=str, default='plus_transform_drop_out',help='how to update item at the end of each hop')
# # parser.add_argument('--item_update_mode', type=str, default='plus',help='how to update item at the end of each hop')
# parser.add_argument('--using_all_hops', type=bool, default=True,
#                     help='whether using outputs of all hops or just the last hop when making prediction')
# parser.add_argument('--base_path', type=str, default=base_path, help='base work dir')
#
# args = parser.parse_args()

"""
原始的训练模型
"""
# ripple_net = RippleNet(args)
# ripple_net.train()
# ripple_net.evaluate()
# ripple_net.predict()

# 我的训练数据
best_score={'auc':0.1}

for cur_para in generate_conf(para_dict):
    args=get_para_parse(cur_para)
    ripple_net=my_RippleNet(args)
    ripple_net.train()
    result,auc=ripple_net.evaluate()
    print('tuning result')
    print(str(cur_para))
    print(result)
    print(" ")

    if auc >best_score['auc']:
        best_score['auc']=auc
        best_score['para']=cur_para
        best_score['result']=result

print('')
print('the best para is:')
print(str(best_score['para']))
print(str(best_score['result']))


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

# print(parser.parse_args())