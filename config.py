REL_PATH = './code/data/output/rel.csv'
REL_SIZE = 18
SCHEMA_PATH = './code/data/input/duie_x/duie_schema.json'


TRAIN_JSON_PATH = './code/data/input/duie_x/duie_train.json'
TEST_JSON_PATH = './code/data/input/duie_x/duie_dev.json'
DEV_JSON_PATH = './code/data/input/duie_x/duie_test.json'

BERT_MODEL_NAME = '../hf_demo/bert-base-chinese'

import torch

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MULTI_GPU = False
if torch.cuda.device_count()>1:
    MULTI_GPU = True

BATCH_SIZE = 80
BATCH_SIZE_GPU0 = 30
BERT_DIM = 768
LR = 1e-5
EPOCH = 50
MODEL_DIR = './code/data/output/'

# 模型预测过程中，需要先匹配出 subject，再基于 subject 预测 relation object。
# 如果 subject 预测错误，后面就不可能正确；另外，数量上也有倍数关系，所以对 subject 的误差做适当加权。
CLS_WEIGHT_COEF = [1.0, 1.0]
SUB_WEIGHT_COEF = 1

SUB_HEAD_BAR = 0.5
SUB_TAIL_BAR = 0.5
OBJ_HEAD_BAR = 0.5
OBJ_TAIL_BAR = 0.5

EPS = 1e-10