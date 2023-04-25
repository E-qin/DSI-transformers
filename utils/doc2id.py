from transformers import AutoModel, AutoTokenizer, RobertaTokenizer, LongformerTokenizer
import faiss
import json
from tqdm import tqdm
import numpy as np
from multiprocessing import Pool

def train_pq(dim, code_size, train_num):
    # https://github.com/facebookresearch/faiss/wiki/Faiss-building-blocks:-clustering,-PCA,-quantization
    # d = 32  # data dimension
    # cs = 4  # code size (bytes)

    # # train set 
    # nt = 10000
    xt = np.random.rand(train_num, dim).astype('float32')

    # dataset to encode (could be same as train)
    pq = faiss.ProductQuantizer(dim, code_size, 8)
    pq.train(xt)

    return  pq
    # n = 20000
    # x = np.random.rand(n, d).astype('float32')
    # # encode 
    # codes = pq.compute_codes(x)
    # print(codes)

    # # decode
    # x2 = pq.decode(codes)
    # # print(x2)
    # # compute reconstruction error
    # avg_relative_error = ((x - x2)**2).sum() / (x ** 2).sum()
    # # print(avg_relative_error)


def read_data(path_in):
    with open(path_in, 'r') as r:
        filename_law_dict = json.load(r)
        return filename_law_dict


def lawformer(tokenizer, model, text):
    # frozen lawformer embedding --> embedding --> ProductQuantizer --> id
    # PQ from https://github.com/facebookresearch/faiss/wiki/Faiss-building-blocks:-clustering,-PCA,-quantization
    
    # tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    # model = AutoModel.from_pretrained(model_name)
    inputs = tokenizer(text, return_tensors='pt')
    outputs = model(**inputs)
    # last_hidden_state = outputs.last_hidden_state
    pooler_output = outputs.pooler_output
    # print(last_hidden_state.shape)
    # print(pooler_output.shape)
    # bs, dim = outputs.shape[0], outputs.shape[-1]
    x = pooler_output.squeeze(0)
    # codes = pq.compute_codes(x)
    return x


def doc2id(path_in, path_out, tokenizer_name, model_name, max_len=4000, dim=768, code_size=8, train_num=10000):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    model = AutoModel.from_pretrained(model_name)
    pq = train_pq(dim, code_size, train_num)

    dict_data = read_data(path_in)
    new_dict = {}
    count = 0
    for filename, val in tqdm(dict_data.items()):
        path = file_prefix + filename
        with open(path, 'r') as r:
            case = json.load(r)
            if 'ajjbqk' not in case:
                continue
            fact = case['ajjbqk'][:max_len]
            embedding = lawformer(tokenizer, model, fact)
            new_dict[filename] = {}
            new_dict[filename]['crime_list'] = val
            new_dict[filename]['lawformer_embedding'] = embedding.detach().numpy()
    data_to_write = json.dumps(new_dict, ensure_ascii=False)
    with open(path_out, 'w') as w:
        w.write(data_to_write)


from multiprocessing import Pool

def process_file(params):
    filename, tokenizer, model, max_len = params
    path = file_prefix + filename
    with open(path, 'r') as r:
        case = json.load(r)
        fact = case['ajjbqk'][:max_len]
        embedding = lawformer(tokenizer, model, fact)
        return (filename, embedding.detach().numpy())

def doc2id_parallel(path_in, path_out, dim=768, code_size=8, train_num=10000, num_workers=8, max_len=4000):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    model = AutoModel.from_pretrained(model_name)
    # pq = train_pq(dim, code_size, train_num)

    dict_data = read_data(path_in)
    new_dict = {}
    count = 0

    with Pool(num_workers) as pool:
        results = pool.map(process_file, [(filename, tokenizer, model, max_len) for filename in dict_data.keys()])

    for filename, embedding in results:
        new_dict[filename] = {}
        new_dict[filename]['crime_list'] = dict_data[filename]
        new_dict[filename]['lawformer_embedding'] = embedding

    data_to_write = json.dumps(new_dict, ensure_ascii=False)
    with open(path_out, 'w') as w:
        w.write(data_to_write)





if __name__ == '__main__':
    tokenizer_name = "hfl/chinese-roberta-wwm-ext"
    model_name = "thunlp/Lawformer"
    model_path = '/home/weijie_yu/pretrain_models/lawformer'
    # text = ["任某提起诉讼，请求判令解除婚姻关系并对夫妻共同财产进行分割。", "任某偷了500块然后花掉了。"]
    text = "任某提起诉讼，请求判令解除婚姻关系并对夫妻共同财产进行分割。"
    # lawformer_pq_id(tokenizer_name, model_name, text)
    file_prefix = '/home/weijie_yu/dataset/legal/lecard/documents/'
    file_path = '/home/weijie_yu/dataset/legal/lecard/LeCaRD-main/data4lm/filename_law_dict.json'
    path_out = '/home/weijie_yu/dataset/legal/lecard/LeCaRD-main/data4lm/filename_law_code_dict.json'

    # doc2id(file_path, path_out, tokenizer_name, model_name)
    doc2id_parallel(file_path, path_out, tokenizer_name, model_name)