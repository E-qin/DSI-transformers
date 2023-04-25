from transformers import AutoTokenizer, AutoModel
tokenizer = AutoTokenizer.from_pretrained("/home/zhongxiang_sun/code/pretrain_model/THUDM--chatglm-6b", trust_remote_code=True)
model = AutoModel.from_pretrained("/home/zhongxiang_sun/code/pretrain_model/THUDM--chatglm-6b", trust_remote_code=True).half().cuda()
history = []
while 1:
    text = input("chatglm: ")
    response, history = model.chat(tokenizer, text, history=history)
    print(response)