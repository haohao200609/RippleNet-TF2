from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler, TensorBoard
from tools.load_data import LoadData
import numpy as np
import datetime
import math
from tensorflow.keras.layers import Embedding, Input, Dense, Softmax, Activation, Lambda
from model.layers import Squeeze, ExpandDims, Embedding2D
from tensorflow.keras.losses import binary_crossentropy
from tools.metrics import auc, f1, precision, recall
from tensorflow.keras.metrics import binary_accuracy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from model.model import BuildModel
from tensorflow import keras
import tensorflow as tf
from collections import defaultdict

class my_RippleNet_remain_kg:
    def __init__(self, args):
        self.args = args
        self._parse_args()
        self.data_info = LoadData(args)
        # self.train_data, self.test_data, self.n_entity, self.n_relation, self.ripple_set = self.data_info.load_data()

        self.train_data, self.test_data, self.n_entity, self.n_relation, self.kg, self.entity_map_dict, self.relation_map_dict,self.id_entity_map,self.id_relation_map=self.data_info.load_choujiang_data()

        self.model = self.build_model()

    def _parse_args(self):
        self.batch_size = self.args.batch_size
        self.epochs = self.args.n_epoch
        self.patience = self.args.patience
        self.dim = self.args.dim
        self.n_hop = self.args.n_hop
        self.kge_weight = self.args.kge_weight
        self.l2_weight = self.args.l2_weight
        self.lr = self.args.lr
        self.n_memory = self.args.n_memory
        self.item_update_mode = self.args.item_update_mode
        self.using_all_hops = self.args.using_all_hops
        self.save_path = self.args.base_path + "/data/" + self.args.dataset
        self.save_path += "/ripple_net_{}_model.h5".format(self.args.dataset)
        current_time = datetime.datetime.now().strftime(('%Y%m%d-%H%M%S'))
        self.log_path = self.args.base_path + "/logs/{}_{}".format(self.args.dataset, current_time)

    def step_decay(self, epoch):
        # learning rate step decay
        initial_l_rate = self.lr
        drop = 0.5
        epochs_drop = 10.0
        l_rate = initial_l_rate * math.pow(drop, math.floor((1 + epoch) / epochs_drop))
        print("learning_rate", l_rate)
        return l_rate

    def build_model(self):
        # Input Tensor
        item_inputs = Input(shape=(), name="items", dtype=tf.int32)

        remain_inputs= Input(shape=(5,), name="remain", dtype=tf.int32)
        remain_relation_inputs=Input(shape=(5,), name="remain_relation_inputs", dtype=tf.int32)

        # label_inputs = Input(shape=(), name="labels", dtype=tf.float32)
        h_inputs = []
        r_inputs = []
        t_inputs = []

        for hop in range(self.n_hop):
            h_inputs.append(Input(shape=(self.n_memory,), name="h_inputs_{}".format(hop), dtype=tf.int32))
            r_inputs.append(Input(shape=(self.n_memory,), name="r_inputs_{}".format(hop), dtype=tf.int32))
            t_inputs.append(Input(shape=(self.n_memory,), name="t_inputs_{}".format(hop), dtype=tf.int32))


        # Matmul layer
        matmul = Lambda(lambda x: tf.matmul(x[0], x[1]))

        # Embedding layer
        l2 = keras.regularizers.l2(self.l2_weight)
        entity_embedding = Embedding(self.n_entity,
                                     self.dim,
                                     embeddings_initializer='glorot_uniform',
                                     embeddings_regularizer=l2,
                                     name="entity_embedding")
        relation_embedding = Embedding2D(self.n_relation,
                                         self.dim,
                                         self.dim,
                                         embeddings_initializer='glorot_uniform',
                                         embeddings_regularizer=l2,
                                         name="relation_embedding")

        # item and ripple embedding
        # [batch size, dim]
        item_embeddings = entity_embedding(item_inputs)
        h_embeddings = []
        r_embeddings = []
        t_embeddings = []
        for hop in range(self.n_hop):
            # [batch size, n_memory, dim]
            h_embeddings.append(entity_embedding(h_inputs[hop]))

            # [batch size, n_memory, dim, dim]
            r_embeddings.append(relation_embedding(r_inputs[hop]))

            # [batch size, n_memory, dim]
            t_embeddings.append(entity_embedding(t_inputs[hop]))

        # # [batch size, 5, dim]
        # remain_embeddings=entity_embedding(remain_inputs)
        # # [batch size, 5, dim, dim]
        # remain_relation_embeddings=relation_embedding(remain_relation_inputs)
        # # [batch_size, 5, dim, 1]
        # reshape_remain=ExpandDims(3)(remain_embeddings)
        #
        # # [batch_size, 5, dim, 1]
        # rh_remain=matmul([remain_relation_inputs,reshape_remain])
        # rh_remain=tf.keras.layers.Flatten(rh_remain)


        """
        weight sum 权重相乘来转换remain embedding
        """
        # [batch size, 5, dim]
        remain_embeddings = entity_embedding(remain_inputs)
        # [batch_size,dim,1]
        item_v=ExpandDims(2)(item_embeddings)
        # [batch_size,n_remain,1]
        remain_prob=Squeeze(2)(matmul([remain_embeddings, item_v]))
        remain_prob_normalized=Softmax()(remain_prob)
        # # [batch_size, n_remain, 1]
        remain_prob_normalized=ExpandDims(2)(remain_prob_normalized)
        # [batch_size, dim]
        remain_combine_embedding=keras.backend.sum(remain_embeddings * remain_prob_normalized, axis=1)


        # """
        # concatenate 直接网络dense转换，remain embedding
        # """
        # # [batch size, 5, dim]
        # remain_embeddings = entity_embedding(remain_inputs)
        # rh_remain = tf.keras.layers.Flatten()(remain_embeddings)
        # print('debug rh_remain.shape')
        # print(rh_remain.shape)
        # remain_combine_embedding=Dense(self.dim,input_dim=40, use_bias=False, kernel_initializer='glorot_uniform', kernel_regularizer=l2,activation='relu')(rh_remain)
        # remain_combine_embedding=tf.keras.layers.Dropout(0.2)(remain_combine_embedding)


        # update item embedding
        o_list = []
        for hop in range(self.n_hop):
            # [batch_size, n_memory, dim, 1]
            reshape_h = ExpandDims(3)(h_embeddings[hop])

            # [batch_size, n_memory, dim]
            Rh = matmul([r_embeddings[hop], reshape_h])
            Rh = Squeeze(3)(Rh)

            # [batch_size, dim, 1]
            v = ExpandDims(2)(item_embeddings)

            # [batch_size, n_memory]
            probs = Squeeze(2)(matmul([Rh, v]))

            # [batch_size, n_memory]
            probs_normalized = Softmax()(probs)

            # [batch_size, n_memory, 1]
            probs_normalized = ExpandDims(2)(probs_normalized)

            # [batch_size, dim]
            o = keras.backend.sum(t_embeddings[hop] * probs_normalized, axis=1)

            item_embeddings = self.update_item_embedding(item_embeddings, o, l2)
            o_list.append(o)

        y = o_list[-1]
        if self.using_all_hops:
            for i in range(self.n_hop - 1):
                y += o_list[i]

        # # 加上剩余物品的向量映射
        # y+=remain_combine_embedding

        # sum

        # Output
        scores = Squeeze()(keras.backend.sum(item_embeddings * y, axis=1))
        scores_normalized = Activation('sigmoid', name='score')(scores)

        # Model
        # model = Model(inputs=[item_inputs, label_inputs] + h_inputs + r_inputs + t_inputs, outputs=scores_normalized)

        # model = Model(inputs=[item_inputs] + h_inputs + r_inputs + t_inputs, outputs=scores_normalized)

        # model = Model(inputs=[item_inputs] + h_inputs + r_inputs + t_inputs+[remain_inputs], outputs=scores_normalized)

        model = Model(inputs=[item_inputs, h_inputs , r_inputs , t_inputs , remain_inputs],
                      outputs=scores_normalized)


        # Loss
        # base_loss = binary_crossentropy(label_inputs, scores_normalized)  # base loss

        kge_loss = 0  # kg loss
        for hop in range(self.n_hop):
            h_expanded = ExpandDims(2)(h_embeddings[hop])
            t_expanded = ExpandDims(3)(t_embeddings[hop])
            # @矩阵相乘
            hRt = Squeeze()(h_expanded @ r_embeddings[hop] @ t_expanded)
            # kge_loss += keras.backend.mean(Activation('sigmoid')(hRt))
            kge_loss += keras.backend.mean(Activation('relu')(hRt))

        l2_loss = 0  # l2 loss
        for hop in range(self.n_hop):
            l2_loss += keras.backend.sum(keras.backend.square(h_embeddings[hop]))
            l2_loss += keras.backend.sum(keras.backend.square(r_embeddings[hop]))
            l2_loss += keras.backend.sum(keras.backend.square(t_embeddings[hop]))

        # model.add_loss(base_loss)
        model.add_loss(self.l2_weight * l2_loss)
        model.add_loss(self.kge_weight * -kge_loss)

        # model.compile(optimizer=Adam(self.lr), metrics=[binary_accuracy, auc, f1, precision, recall])
        """
        使用这个loss里面的，是就不需要传入label的情况了
        """
        # model.compile(loss='binary_crossentropy', optimizer=Adam(self.lr),
        #               metrics=[binary_accuracy, auc, f1, precision, recall])

        model.compile(loss='binary_crossentropy', optimizer=Adam(self.lr),
                      metrics=[ tf.keras.metrics.AUC(), tf.keras.metrics.Precision(thresholds=0.3), tf.keras.metrics.Recall(thresholds=0.3)])

        return model

    # 获得玩家当前这一条的ripple set
    def get_ripple_item(self,kg,hist_buy_idx_list,hop):
        # memories_h = [memories_h[i] for i in indices]
        # memories_r = [memories_r[i] for i in indices]
        # memories_t = [memories_t[i] for i in indices]
        # ripple_set[user].append((memories_h, memories_r, memories_t))

        if len(hist_buy_idx_list)==1:
            if self.id_entity_map[hist_buy_idx_list[0]]=='-999':

                a=1
        ripple_set = {}
        for h in range(self.args.n_hop):
            memories_h = []
            memories_r = []
            memories_t = []
            if h == 0:
                tails_of_last_hop = hist_buy_idx_list
            else:
                last_hop=h-1
                tails_of_last_hop = ripple_set[last_hop][2]

            for entity in tails_of_last_hop:
                for tail_and_relation in kg[entity]:
                    memories_h.append(entity)
                    memories_r.append(tail_and_relation[1])
                    memories_t.append(tail_and_relation[0])
            replace = len(memories_h) < self.args.n_memory
            indices = np.random.choice(len(memories_h), size=self.args.n_memory, replace=replace)
            memories_h = [memories_h[i] for i in indices]
            memories_r = [memories_r[i] for i in indices]
            memories_t = [memories_t[i] for i in indices]
            ripple_set[h]=(memories_h, memories_r, memories_t)
        return ripple_set[hop]

    # def data_parse(self, data):
    #     # build X, y from data
    #     np.random.shuffle(data)
    #     items = data[:, 1]
    #     labels = data[:, 2]
    #     memories_h = list(range(self.n_hop))
    #     memories_r = list(range(self.n_hop))
    #     memories_t = list(range(self.n_hop))
    #     for hop in range(self.n_hop):
    #         memories_h[hop] = np.array([self.ripple_set[user][hop][0] for user in data[:, 0]])
    #         memories_r[hop] = np.array([self.ripple_set[user][hop][1] for user in data[:, 0]])
    #         memories_t[hop] = np.array([self.ripple_set[user][hop][2] for user in data[:, 0]])
    #     x=[items, labels, memories_h , memories_r , memories_t]
    #     # return [items, labels] + memories_h + memories_r + memories_t, labels
    #
    #     return [items] + memories_h + memories_r + memories_t, labels


    # 解析输入数据，并且把输入数据，满足网络结构
    def choujiang_data_parse(self, data):
        np.random.shuffle(data)
        # 这个是最后输出结果的内容，但是需要自己先用一个dict来转换
        memories_h = list(range(self.n_hop))
        memories_r = list(range(self.n_hop))
        memories_t = list(range(self.n_hop))

        # 因为这里内外循环的问题，所以需要用一个dict做中间转换
        memories_h_dict=defaultdict(list)
        memories_r_dict = defaultdict(list)
        memories_t_dict = defaultdict(list)

        item_list=[]
        label_list=[]
        remain_list=[]

        for data_line in data:
            hostnum,uid,item_index,label,hist_buy_idx_list,remain_idx_list=data_line
            item_list.append(item_index)
            label_list.append(label)
            remain_list.append(remain_idx_list)

            # 整合剩余存量节点到历史节点中
            hist_buy_idx_list=list(set(hist_buy_idx_list+remain_idx_list))
            for hop in range(self.n_hop):
                # 输出结果是一个tuple(list)的个是，tuple有3个元素，每个元素分布是一个list
                ripple_set=self.get_ripple_item(self.kg,hist_buy_idx_list,hop)
                memories_h_dict[hop].append(ripple_set[0])
                memories_r_dict[hop].append(ripple_set[1])
                memories_t_dict[hop].append(ripple_set[2])

        for hop in range(self.n_hop):
            memories_h[hop]=  np.array(memories_h_dict[hop])
            memories_r[hop] = np.array(memories_r_dict[hop])
            memories_t[hop] = np.array(memories_t_dict[hop])

        """
        相加和list聚合最后的结果是不一样的，相加，会有7个item的list，x_other只有4个item的list
        memories_r本身就是list套list，所以不怕
        """
        item_list=np.array(item_list)
        remain_list=np.array(remain_list)
        label_list=np.array(label_list)
        x=[item_list] + memories_h + memories_r + memories_t+ [remain_list]
        x_other=[item_list,memories_h,memories_r,memories_t,remain_list]
        return [item_list] + memories_h + memories_r + memories_t+ [remain_list],label_list

    def train(self):
        # print("train model ...")
        # self.model.summary()
        # X, y = self.data_parse(self.train_data)
        X, y = self.choujiang_data_parse(self.train_data)
        # print('num of train sample  %.1f',len(y))
        # print('label 1 num  %.1f',sum(y))

        tensorboard = TensorBoard(log_dir=self.log_path, histogram_freq=1)
        early_stopper = EarlyStopping(patience=self.patience, verbose=1)
        model_checkpoint = ModelCheckpoint(self.save_path, verbose=1, save_best_only=True)
        learning_rate_scheduler = LearningRateScheduler(self.step_decay)
        self.model.fit(x=X,
                       y=y,
                       batch_size=self.batch_size,
                       epochs=self.epochs,
                       verbose=0,
                       validation_split=0.2,
                       callbacks=[early_stopper, model_checkpoint, learning_rate_scheduler, tensorboard])

    def evaluate(self):
        model = self.build_model()
        model.load_weights(self.save_path)
        # print("evaluate model ...")
        # X, y = self.data_parse(self.test_data)
        X, y = self.choujiang_data_parse(self.test_data)

        score = model.evaluate(X, y, batch_size=self.batch_size)
        # print("- loss: {} "
        #       # "- binary_accuracy: {} "
        #       "- auc: {} "
        #       # "- f1: {} "
        #       "- precision: {} "
        #       "- recall: {}".format(*score))
        result="- loss: {} - auc: {} - precision: {} - recall: {} ".format(*score)
        return result,score[1]

    def predict(self):
        model = self.build_model()
        model.load_weights(self.save_path)
        # X, y = self.data_parse(self.test_data)
        X, y = self.choujiang_data_parse(self.test_data)
        pred = model.predict(X, batch_size=self.batch_size)
        result = [1 if x > 0.3 else 0 for x in pred]
        return result,pred,y

    def update_item_embedding(self, item_embeddings, o, l2):
        # transformation matrix for updating item embeddings at the end of each hop
        transform_matrix = Dense(self.dim, use_bias=False, kernel_initializer='glorot_uniform', kernel_regularizer=l2,activation='relu')
        if self.item_update_mode == "replace":
            item_embeddings = o
        elif self.item_update_mode == "plus":
            item_embeddings = item_embeddings + o
        elif self.item_update_mode == "replace_transform":
            item_embeddings = transform_matrix(o)
        elif self.item_update_mode == "plus_transform":
            item_embeddings = transform_matrix(item_embeddings + o)
        elif self.item_update_mode == "plus_transform_drop_out":
            item_embeddings = transform_matrix(item_embeddings + o)
            drop_layer=tf.keras.layers.Dropout(0.2)
            item_embeddings=drop_layer(item_embeddings)
        else:
            raise Exception("Unknown item updating mode: " + self.item_update_mode)
        return item_embeddings
