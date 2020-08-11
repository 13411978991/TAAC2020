import torch
from tqdm import tqdm
import copy

#返回广告序列[ad_id1,ad_id2,...]
def df2sent(data,feature,vocab,config):
  data=data[['time','user_id',feature,'click_times']]
  data=data.groupby(['user_id','time',feature])['click_times'].sum().reset_index()
  data=data.sort_values(by=['user_id','time'],ascending=False)
  data.loc[data[feature] == '\\N', feature] = 0
  data=data.drop_duplicates()
  df={}
  for i in tqdm(data.values):
    if df.get(i[0]):
      if len(df[i[0]])<config.pad_size:
        df[i[0]][0].append(vocab[str(i[2])].index)
        df[i[0]][1].append(i[3])
    else:
      df[i[0]]=([vocab[str(i[2])].index],[i[3]])
  return df


#定义迭代器
class DatasetIterater(object):
    def __init__(self, batches, batch_size, device,pad_size):
        self.batch_size = batch_size
        self.batches = batches
        self.pad_size = pad_size
        self.n_batches = len(batches) // batch_size
        self.residue = False  # 记录batch数量是否为整数
        if len(batches) % self.n_batches != 0:
            self.residue = True
        self.index = 0
        self.device = device

    def _to_tensor(self, datas):
        data=[]
        for item in datas:
          tmp_ad=copy.deepcopy(item[0][0])
          tmp_ad_w=copy.deepcopy(item[0][1])
          ad_len=len(tmp_ad)
          if ad_len<self.pad_size:
            tmp_ad.extend([1]*(self.pad_size-ad_len))#词序列不够padsize的用索引1 ('/s)补齐
            tmp_ad_w.extend([-1e9]*(self.pad_size-ad_len))#词序列不够padsize的点击次数填充负无穷，直接填充0也可以，问题不大
          data.append(((tmp_ad[0:self.pad_size],tmp_ad_w[0:self.pad_size]),item[1]))
        x = torch.LongTensor([i[0][0] for i in data]).to(self.device)
        x1 = torch.LongTensor([i[0][1] for i in data]).to(self.device)
        y = torch.LongTensor([i[1] for i in data]).to(self.device)
        return (x,x1),y

    def __next__(self):
        if self.residue and self.index == self.n_batches:
            batches = self.batches[self.index * self.batch_size: len(self.batches)]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

        elif self.index >= self.n_batches:
            self.index = 0
            raise StopIteration
        else:
            batches = self.batches[self.index * self.batch_size: (self.index + 1) * self.batch_size]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

    def __iter__(self):
        return self

    def __len__(self):
        if self.residue:
            return self.n_batches + 1
        else:
            return self.n_batches


def build_iterator(dataset, config):
    iter = DatasetIterater(dataset, config.batch_size, config.device,config.pad_size)
    return iter