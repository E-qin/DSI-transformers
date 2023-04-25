import argparse
import collections
import torch
import os 
import numpy as np
# import data_loader.data_loaders as module_data
from data_loader.data_loaders import build_dataloader
import model.loss as module_loss
import model.metric as module_metric
# import model.model as module_arch
from model.model import T5_model
from parse_config import ConfigParser
from trainer import Trainer
from utils import prepare_device
from transformers import T5Tokenizer, T5Config, T5ForConditionalGeneration
from torch.nn.parallel import DistributedDataParallel


# fix random seeds for reproducibility
SEED = 42
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)

def main(config):
    # torch.cuda.set_device(args.local_rank)
    # torch.distributed.init_process_group(backend="nccl")

    logger = config.get_logger('train')

    # setup data_loader instances
    # data_loader = config.init_obj('data_loader', module_data)
    # valid_data_loader = data_loader.split_validation()
    special_tokens = ["<extra_id_{}>".format(i) for i in range(100)]
    model_tokenizer  = T5Tokenizer.from_pretrained(
                            config["model_path"],
                            do_lower_case=True,
                            max_length= config["max_len"],
                            truncation=True,
                            additional_special_tokens=special_tokens,
                        )
    model_config = T5Config.from_pretrained(config["model_path"])
    mode_model = T5ForConditionalGeneration.from_pretrained(config["model_path"], config=model_config)
    mode_model.resize_token_embeddings(len(model_tokenizer))
    model = T5_model(mode_model, model_tokenizer, config["max_len_out"])
    # model = DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank)

    data_loader = build_dataloader(config['train_dir'], model_tokenizer, config["batch_size"], config["shuffle"], config["num_workers"], config["max_len"])
    valid_data_loader = build_dataloader(config['test_dir'], model_tokenizer, config["batch_size"], config["shuffle"], config["num_workers"], config["max_len"])
    # build model architecture, then print to console
    # model = config.init_obj('arch', module_arch, model, tokenizer, config["max_len"])
    logger.info(model)

    # prepare for (multi-device) GPU training
    device, device_ids = prepare_device(config['n_gpu'])
    model = model.to(device)
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)

    # get function handles of loss and metrics
    criterion = getattr(module_loss, config['loss'])
    metrics = [getattr(module_metric, met) for met in config['metrics']]

    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = config.init_obj('optimizer', torch.optim, trainable_params)
    lr_scheduler = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer)

    trainer = Trainer(model, criterion, metrics, optimizer,
                      config=config,
                      device=device,
                      data_loader=data_loader,
                      valid_data_loader=valid_data_loader,
                      lr_scheduler=lr_scheduler)

    trainer.train()


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')
    # args.add_argument('--local_rank', default=None, type=int,
    #                   help='indices of GPUs to enable (default: all)')
    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'], type=float, target='optimizer;args;lr'),
        CustomArgs(['--bs', '--batch_size'], type=int, target='data_loader;args;batch_size')
    ]
    config = ConfigParser.from_args(args, options)
    main(config)
