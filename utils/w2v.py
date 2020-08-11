import gensim
import pandas as pd
from tqdm import tqdm
import os


# flag1为key，flag2选择为需要编码的词，如ad_id和creative_id，返回click_log和ad合并后的文件
def merge_file(path_train, path_test, flag1, flag2):
    click_log_train = pd.read_csv(os.path.join(path_train, 'click_log.csv'))
    ad_train = pd.read_csv(os.path.join(path_train, 'ad.csv'))
    click_log_train = click_log_train.fillna(0)
    ad_train = ad_train.fillna(0)
    click_log_train = click_log_train.merge(ad_train, on='creative_id', how='inner')
    del ad_train
    print('合并完成')
    return click_log_train

#train为输入文件，clo1为key字段，clo2为编码词字段，返回clo2组成的序列
def tosent(train,clo1,clo2):
    train = train[[clo1, clo2]]
    dic = {}
    for item in tqdm(train.values):
        if dic.get(item[0]):
            dic[item[0]].append(str(item[1]))
        else:
            dic[item[0]] = [str(item[1])]
    sentence = []
    for key in tqdm(dic.keys()):
        tmp = dic[key]
        tmp.append('/u')
        tmp.insert(0, '/s')
        sentence.append(tmp)
    return sentence

#返回合并后的文件train,词向量模型的索引，词向量模型的权值文件
def w2v(path_train,path_test,flag1,flag2,size):
    train=merge_file(path_train,path_test,flag1,flag2)
    sentence=tosent(train,flag1,flag2)
    model_w2v=gensim.models.word2vec.Word2Vec(sentence,window=10,min_count=1,workers=10,iter=10,size=size)
    model_w2v.save(os.path.join(path_train,'model_w2v_'+str(size)+'.model'))
    vocab = model_w2v.wv.vocab
    weight = model_w2v.wv.vectors
    return train,vocab,weight