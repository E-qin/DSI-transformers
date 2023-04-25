import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel
from transformers import T5Tokenizer, T5Config, T5ForConditionalGeneration

class MnistModel(BaseModel):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

class T5_model(BaseModel):
    def __init__(self, model, tokenizer, max_output_len, sample=True):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.sample = sample
        self.max_output_len = max_output_len

    def forward(self, inputs):
        outputs = self.model(
        input_ids = inputs['input_ids'],
        attention_mask = inputs['attention_mask'],
        labels = inputs['labels'],
        )
        loss = outputs.loss
        # print('loss:{}'.format(loss))

        logits = self.model.generate(
            input_ids = inputs['input_ids'],
            attention_mask = inputs['attention_mask'],
            # decoder_input_ids = inputs['labels'],
            max_length=32, 
            do_sample= True
            # early_stopping=True,
            )
        
        # logits = outputs.logits
        # print('logits:{}'.format(logits.shape))  # [bs, len]
        batch_pred = self.tokenizer.batch_decode(logits, skip_special_tokens=True)
        # print('batch_pred:{}'.format(batch_pred))
        return batch_pred, loss
