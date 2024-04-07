import torch
import utility
import data
from trainer import Trainer
import warnings
import vessl
import numpy as np
from tqdm import tqdm
import yaml
import os
import argparse
from model import redcnn

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='test')
parser.add_argument('--load', type=str, default='test')
parser.add_argument('--save', type=str, default='test')
args = parser.parse_args()

# vessl.configure(organization_name='2302-AI-LEC-MEDI', project_name='Minwoo-Yu')
# vessl.init(message='test_'+'redcnn_l1')

def test():
    with open(os.path.join('configs', args.config+'.yaml'), 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        print('config loaded, config_path: {}'.format(os.path.join('configs', args.config+'.yaml')))

    loader = data.Data(config, test_only=True)
    model_test = redcnn.REDCNN().to('cuda')
    model_test.load_state_dict(torch.load(os.path.join('experiment', args.load, 'model', 'model_best.pt')))
    model_test.eval()
    data_test = loader.loader_test

    rmse_val = 0
    saver = torch.Tensor([])

    with torch.no_grad():
        for i, (ldct, ndct) in enumerate(tqdm(data_test)):
            ldct, ndct = ldct.cuda(), ndct.cuda()
            ldct = utility.normalize(ldct, -500, 500)
            denoised = model_test(ldct)
            denoised = utility.denormalize(denoised, -500, 500)
            rmse_val += utility.calc_rmse(denoised, ndct) / len(data_test)
            saver = torch.cat([saver, denoised.cpu()], 0)

        vessl.log(step = 0, payload={'rmse_val': rmse_val.item()})

if __name__ == '__main__':
    test()