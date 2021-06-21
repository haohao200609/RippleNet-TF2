#coding:utf-8
import os
import numpy as np
import collections


class LoadData:
    def __init__(self, args):
        self.args = args
        self.data_path = os.path.join(args.base_path, 'data/')

    def load_data(self):
        train_data, test_data, user_history_dict = self.load_rating()
        n_entity, n_relation, kg = self.load_kg()
        ripple_set = self.get_ripple_set(kg, user_history_dict)
        return train_data, test_data, n_entity, n_relation, ripple_set

    def load_choujiang_data(self):
        n_entity, n_relation, kg, entity_map_dict, relation_map_dict=self.load_choujiang_kg()
        train_data, test_data=self.load_choujiang_file()
        return train_data, test_data,n_entity, n_relation, kg, entity_map_dict, relation_map_dict

    """
    返回的全部都是有交互的数据
    """
    def load_rating(self):
        print('reading rating file ...')

        # reading rating file
        rating_file = self.data_path + self.args.dataset + '/ratings_final'
        if os.path.exists(rating_file + '.npy'):
            rating_np = np.load(rating_file + '.npy')
        else:
            rating_np = np.loadtxt(rating_file + '.txt', dtype=np.int32)
            np.save(rating_file + '.npy', rating_np)

        print('splitting dataset ...')

        # train:test = 6:2
        test_ratio = 0.2
        n_ratings = rating_np.shape[0]

        test_indices = np.random.choice(n_ratings, size=int(n_ratings * test_ratio), replace=False)
        train_indices = set(range(n_ratings)) - set(test_indices)

        test_indices_old = test_indices[:]
        train_indices_old = list(train_indices)[:]

        # traverse training data, only keeping the users with positive ratings
        user_history_dict = dict()
        for i in train_indices:
            user = rating_np[i][0]
            item = rating_np[i][1]
            rating = rating_np[i][2]
            if rating == 1:
                if user not in user_history_dict:
                    user_history_dict[user] = []
                user_history_dict[user].append(item)
        """
        user_history_dict只是保存所有有过交互的user的信息，下面train_indices是为了过滤一些，没有再train_indice只在test里面有过的user
        
        这里统计的是，玩家训练集里面所有的历史购买的数据，但是玩家history_dict在训练数据里面的时候，是没有区分任何先后顺序的，也就是我再训练的时候，我周一看了A，周二看了B，周三看了C，我的特征里面，就是认为看了A和C的玩家，都会看B。用这样的一个特征取进行预测。我新按照这样的一版，去试一试，看看结果吧
        
        这样的处理有好处，就是模型能学到，看了A和C的人会去看B，如果有时序，模型只能学到看了A的人会去看B。但是movie len是要给玩家推送我没看过的电影，药品购买，则是玩家历史买了物品之后，我还继续给他退
        """
        train_indices = [i for i in train_indices if rating_np[i][0] in user_history_dict]
        test_indices = [i for i in test_indices if rating_np[i][0] in user_history_dict]

        train_data = rating_np[train_indices]
        test_data = rating_np[test_indices]

        return train_data, test_data, user_history_dict

    def parse_hist_buy_info(self,hist_buy_list,entity_map_dict):
        hist_buy_info=eval(hist_buy_list)
        idx_hist_list=[]
        for item in hist_buy_info:
            idx=entity_map_dict[item]
            idx_hist_list.append(idx)
        return idx_hist_list

    def load_choujiang_file(self,entity_map_dict,relation_map_dict):
        print('reading choujiang file...')
        choujiang_file=self.data_path  + '/tf_kg_train.txt'
        total_data=[]
        for i, line in enumerate(open(choujiang_file)):
            dt, hostnum, uid, item_id, label,hist_buy_list, effect_beast_mp_left, effect_hp_left, effect_beast_hp_left, cur_lingqi, effect_mp_left,beast_mp_seg,hp_seg,best_hp_seg,lingqi_seg,mp_seg=line.strip().split('\t')
            hostnum=int(hostnum)
            uid=int(uid)
            label=int(label)
            item_index=entity_map_dict(item_id)
            hist_buy_idx_list=self.parse_hist_buy_info(hist_buy_list,entity_map_dict)
            beast_mp_seg='pet_blue'+'_remain'+'~'+str(beast_mp_seg)
            hp_seg='human_red'+'_remain'+'~'+str(hp_seg)
            best_hp_seg='pet_red'+'_remain'+'~'+str(best_hp_seg)
            lingqi_seg = 'huimengyin' + '_remain' + '~' + str(lingqi_seg)
            mp_seg = 'human_blue' + '_remain' + '~' + str(mp_seg)


            remain_list=[beast_mp_seg,hp_seg,best_hp_seg,lingqi_seg,mp_seg]
            remain_idx_list=list(map(entity_map_dict.get,remain_list))
            result=[hostnum,uid,item_index,label,hist_buy_idx_list,remain_idx_list]
            total_data.append(result)
        # 转换为np array才可以使用idx的映射
        total_data=np.array(total_data)
        test_ratio = 0.2
        n_ratings = len(total_data)
        test_indices = np.random.choice(n_ratings, size=int(n_ratings * test_ratio), replace=False)
        train_indices = set(range(n_ratings)) - set(test_indices)
        train_data = total_data[train_indices]
        test_data = total_data[test_indices]
        return train_data,test_data


    def load_choujiang_kg(self):
        print('reading KG file ...')
        file_name = self.data_path + '/kg_build.txt'

        entity_map_dict={}
        relation_map_dict={}
        entity_idx=0
        relaton_idx=0
        kg = collections.defaultdict(list)
        for i, line in enumerate(open(file_name)):
            start_node,relation_type,end_node=line.strip().split('\t')
            if start_node not in entity_map_dict.keys():
                entity_map_dict[start_node]=entity_idx
                entity_idx+=1
            if relation_type not in relation_map_dict.keys():
                relation_map_dict[relation_type]=relaton_idx
                relaton_idx+=1
            if end_node not in entity_map_dict.keys():
                entity_map_dict[end_node]=entity_idx
                entity_idx+=1
            head=entity_map_dict[start_node]
            relation=relation_map_dict[relation_type]
            tail=entity_map_dict[end_node]
            kg[head].append((tail, relation))
        n_entity=len(entity_map_dict.keys())
        n_relation=len(relation_map_dict.keys())
        return n_entity, n_relation, kg,entity_map_dict,relation_map_dict

    def load_kg(self):
        print('reading KG file ...')

        # reading kg file
        kg_file = self.data_path + self.args.dataset + '/kg_final'
        if os.path.exists(kg_file + '.npy'):
            kg_np = np.load(kg_file + '.npy')
        else:
            kg_np = np.loadtxt(kg_file + '.txt', dtype=np.int32)
            np.save(kg_file + '.npy', kg_np)

        n_entity = len(set(kg_np[:, 0]) | set(kg_np[:, 2]))
        n_relation = len(set(kg_np[:, 1]))

        print('constructing knowledge graph ...')
        kg = collections.defaultdict(list)
        for head, relation, tail in kg_np:
            kg[head].append((tail, relation))

        return n_entity, n_relation, kg

    def get_ripple_set(self, kg, user_history_dict):
        print('constructing ripple set ...')

        # user -> [(hop_0_heads, hop_0_relations, hop_0_tails), (hop_1_heads, hop_1_relations, hop_1_tails), ...]
        ripple_set = collections.defaultdict(list)

        for user in user_history_dict:
            for h in range(self.args.n_hop):
                memories_h = []
                memories_r = []
                memories_t = []

                if h == 0:
                    tails_of_last_hop = user_history_dict[user]
                else:
                    tails_of_last_hop = ripple_set[user][-1][2]

                for entity in tails_of_last_hop:
                    for tail_and_relation in kg[entity]:
                        memories_h.append(entity)
                        memories_r.append(tail_and_relation[1])
                        memories_t.append(tail_and_relation[0])

                """
                if the current ripple set of the given user is empty, we simply copy the ripple set of the last hop here
                this won't happen for h = 0, because only the items that appear in the KG have been selected
                this only happens on 154 users in Book-Crossing dataset (since both book dataset and the KG are sparse)
                """
                if len(memories_h) == 0:
                    ripple_set[user].append(ripple_set[user][-1])
                else:
                    # sample a fixed-size 1-hop memory for each user
                    replace = len(memories_h) < self.args.n_memory
                    indices = np.random.choice(len(memories_h), size=self.args.n_memory, replace=replace)
                    memories_h = [memories_h[i] for i in indices]
                    memories_r = [memories_r[i] for i in indices]
                    memories_t = [memories_t[i] for i in indices]
                    ripple_set[user].append((memories_h, memories_r, memories_t))
        # ripple_set: [user][(第一跳的内容),(第二跳的内容)]，第一跳内容(head_list,relation_list,tail_list)
        return ripple_set
