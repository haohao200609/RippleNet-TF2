#coding:utf-8
import argparse
import numpy as np

RATING_FILE_NAME = dict({'movie': 'movie_ratings.dat', 'book': 'book_ratings.csv', 'news': 'ratings.txt'})
SEP = dict({'movie': '::', 'book': ';', 'news': '\t'})
THRESHOLD = dict({'movie': 4, 'book': 0, 'news': 0})
SKIP_LINE = dict({'movie': 0, 'book': 1, 'news': 0})

"""
把数据库相应的物品id，改变成为从0开始id
"""

def read_item_index_to_entity_id_file():
    file = 'data/' + DATASET + '/item_index2entity_id_rehashed.txt'
    print('reading item index to entity id file: ' + file + ' ...')
    i = 0
    for line in open(file, encoding='utf-8').readlines():
        item_index = line.strip().split('\t')[0]
        satori_id = line.strip().split('\t')[1]
        item_index_old2new[item_index] = i
        entity_id2index[satori_id] = i
        i += 1
    print('item_index_old2new:'+str(len(item_index_old2new.keys())))
    print('entity_id2index:' + str(len(entity_id2index.keys())))
    print('i:' + str(i))

"""
第一行
1::1193::5::978300760

把评分，解析成为最后输出正负样本ratings_final.txt，玩家评分超过xxx是正样本，label是1，负样本是0，没有看过的是负样本，但是对于游戏可能这个要重新分一下

"""
def convert_rating():
    file = 'data/' + DATASET + '/' + RATING_FILE_NAME[DATASET]
    skip_line = SKIP_LINE[DATASET] - 1

    print('reading rating file ...')
    item_set = set(item_index_old2new.values())
    user_pos_ratings = dict()
    user_neg_ratings = dict()

    for i, line in enumerate(open(file, encoding='utf-8').readlines()):
        if i == skip_line:
            continue
        array = line.strip().split(SEP[DATASET])

        # remove prefix and suffix quotation marks for book rating dataset
        if DATASET == 'book':
            array = list(map(lambda x: x[1:-1], array))

        item_index_old = array[1]
        if item_index_old not in item_index_old2new:  # the item is not in the final item set
            continue
        item_index = item_index_old2new[item_index_old]

        user_index_old = int(array[0])

        rating = float(array[2])
        if rating >= THRESHOLD[DATASET]:
            if user_index_old not in user_pos_ratings:
                user_pos_ratings[user_index_old] = set()
            user_pos_ratings[user_index_old].add(item_index)
        else:
            if user_index_old not in user_neg_ratings:
                user_neg_ratings[user_index_old] = set()
            user_neg_ratings[user_index_old].add(item_index)

    print('converting rating file ...')
    writer = open('data/' + DATASET + '/ratings_final.txt', 'w', encoding='utf-8')
    user_cnt = 0
    user_index_old2new = dict()
    for user_index_old, pos_item_set in user_pos_ratings.items():
        if user_index_old not in user_index_old2new:
            user_index_old2new[user_index_old] = user_cnt
            user_cnt += 1
        user_index = user_index_old2new[user_index_old]

        for item in pos_item_set:
            writer.write('%d\t%d\t1\n' % (user_index, item))
        unwatched_set = item_set - pos_item_set
        if user_index_old in user_neg_ratings:
            unwatched_set -= user_neg_ratings[user_index_old]
        for item in np.random.choice(list(unwatched_set), size=len(pos_item_set), replace=False):
            writer.write('%d\t%d\t0\n' % (user_index, item))
    writer.close()
    print('number of users: %d' % user_cnt)
    print('number of items: %d' % len(item_set))


def convert_kg():
    print('converting kg file ...')
    entity_cnt = len(entity_id2index)
    relation_cnt = 0

    writer = open('data/' + DATASET + '/kg_final.txt', 'w', encoding='utf-8')

    files = []
    if DATASET == 'movie':
        files.append(open('data/' + DATASET + '/kg_part1_rehashed.txt', encoding='utf-8'))
        files.append(open('data/' + DATASET + '/kg_part2_rehashed.txt', encoding='utf-8'))
    else:
        files.append(open('data/' + DATASET + '/kg_rehashed.txt', encoding='utf-8'))
    total_triple_count=0
    for file in files:
        for line in file:
            array = line.strip().split('\t')
            head_old = array[0]
            relation_old = array[1]
            tail_old = array[2]

            if head_old not in entity_id2index:
                entity_id2index[head_old] = entity_cnt
                entity_cnt += 1
            head = entity_id2index[head_old]

            if tail_old not in entity_id2index:
                entity_id2index[tail_old] = entity_cnt
                entity_cnt += 1
            tail = entity_id2index[tail_old]

            if relation_old not in relation_id2index:
                relation_id2index[relation_old] = relation_cnt
                relation_cnt += 1
            relation = relation_id2index[relation_old]

            writer.write('%d\t%d\t%d\n' % (head, relation, tail))
            total_triple_count += 1

    writer.close()
    print('number of entities (containing items): %d' % entity_cnt)
    print('number of relations: %d' % relation_cnt)


if __name__ == '__main__':
    np.random.seed(555)

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', type=str, default='movie', help='which dataset to preprocess')
    args = parser.parse_args()
    DATASET = args.dataset
    # 这个是KG里面的entity编号
    entity_id2index = dict()
    """
    大概就集中类型
    film.actor.film
    film.film.genre
    film.film.country
    一共就12中类型
    """
    relation_id2index = dict()
    item_index_old2new = dict()

    read_item_index_to_entity_id_file()
    convert_rating()
    # entity id和relation ID这样的三元结构，转换为从0开始的index
    convert_kg()

    print('done')
