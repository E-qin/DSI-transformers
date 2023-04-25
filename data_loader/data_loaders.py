# from torchvision import datasets, transforms
from base import BaseDataLoader
import json
from torch.utils.data import Dataset, DataLoader
import torch

# class MnistDataLoader(BaseDataLoader):
#     """
#     MNIST data loading demo using BaseDataLoader
#     """
#     def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
#         trsfm = transforms.Compose([
#             transforms.ToTensor(),
#             transforms.Normalize((0.1307,), (0.3081,))
#         ])
#         self.data_dir = data_dir
#         self.dataset = datasets.MNIST(self.data_dir, train=training, download=True, transform=trsfm)
#         super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)


class LawLoader(BaseDataLoader):
    def __init__(self,  data_dir):
        # super(LawLoader, self).__init__()
        self.data_list = []
        with open(data_dir, "r", encoding="utf8") as f:
            for idx, line in enumerate(f):
                data_per_line = json.loads(line)
                law_id = list(data_per_line)[0]
                law_name, law_content = data_per_line[law_id]['name'], data_per_line[law_id]['content']
                # self.data_list.append([law_id, law_name, law_content])
                self.data_list.append([law_id + ' ' + law_name, law_content])

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        # law_id, law_name, law_content = self.data_list[index]
        # return law_id, law_name, law_content
        law_name, law_content = self.data_list[index]
        return law_name, law_content
        
class Collate:
    def __init__(self, tokenizer, max_seq_len, max_target_len):
        # self.tokenizer = BertTokenizer.from_pretrained(vocab_path)
        # self.tokenizer  = T5Tokenizer.from_pretrained(
        #                     vocab_path,
        #                     do_lower_case=True,
        #                     max_length=max_seq_len,
        #                     truncation=True,
        #                     additional_special_tokens=special_tokens,
        #                 )
        self.tokenizer = tokenizer
        self.input_length = max_seq_len
        self.target_length = max_target_len
        self.max_seq_len = max_seq_len

    def __call__(self, batch):
        # id_batch, name_batch, content_batch = zip(*batch)
        # print('batch:{}'.format(batch))
        name_batch, content_batch = zip(*batch)
        name_batch, content_batch = list(name_batch), list(content_batch)
        # print('name_batch:{}'.format(name_batch))
        # print('content_batch:{}'.format(content_batch))

        # batch_encoding = self.tokenizer.prepare_seq2seq_batch(name_batch, content_batch, max_length=self.max_seq_len, max_target_length = self.target_length)
        # return batch_encoding, content_batch
        content_batch = self.tokenizer.prepare_seq2seq_batch(content_batch, max_length=self.max_seq_len, padding='max_length',truncation=True)
        return content_batch, name_batch
    

def build_dataloader(jsonfile_path, tokenizer, batch_size, shuffle=True, num_workers=0, max_seq_len=50, max_target_len=512):
    data_generator = LawLoader(jsonfile_path)
    collate = Collate(tokenizer, max_seq_len, max_target_len)
    return DataLoader(
        data_generator,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate
    )