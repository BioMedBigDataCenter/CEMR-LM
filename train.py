# 导入所需的库和模块
import time
import torch
import numpy as np
from pretrain_eval import train, init_network
from importlib import import_module
import argparse
from pretrain_utils import build_dataset, build_iterator, get_time_dif
import torch.nn as nn

import logging
import os
from datetime import datetime
from pprint import pformat

parser = argparse.ArgumentParser(description='sentence classification')
parser.add_argument('--model', type=str, required=True, help='model: CEMR-LM')
args = parser.parse_args()

# 主程序入口
if __name__ == '__main__':
    # 选择数据集类型
    dataset = 'CTC_Data'  # XBS_Data RYJL_Data CTC_Data

    model_name = args.model  # bert
    x = import_module('models.' + model_name)
    config = x.Config(dataset)
    
    # 设置随机种子，保证结果可重复
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True  # 保证每次结果一样

    os.makedirs('log', exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    model_name = config.model_name
    log_filename = f'log/{dataset}_{model_name}_{timestamp}.log'
    log_path = os.path.join(os.getcwd(), log_filename)
    logging.basicConfig(filename=log_path, level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger()
    logger.info("Model Configuration:\n%s", pformat(vars(config)))
    
    start_time = time.time()
    print("Loading data...")
    logger.info("Loading data...")
    
    train_data, dev_data, test_data = build_dataset(config)
    train_iter = build_iterator(train_data, config)
    dev_iter = build_iterator(dev_data, config)
    test_iter = build_iterator(test_data, config)
    
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)
    logger.info("Time usage: " + str(time_dif))
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = x.Model(config).to(device)
    if torch.cuda.device_count() > 1:
        print("Use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)

    train(config, model, train_iter, dev_iter, test_iter, logger=logger)