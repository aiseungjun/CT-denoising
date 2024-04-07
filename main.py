import torch
import utility
import data
import model
from trainer import Trainer
import warnings
import vessl
import yaml
import os
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='train/redcnn')
    parser.add_argument('--save', type=str, default='redcnn')
    args = parser.parse_args()

    with open(os.path.join('configs', args.config+'.yaml'), 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        print('config loaded, config_path: {}'.format(os.path.join('configs', args.config+'.yaml')))
 
    torch.manual_seed(config['seed'])
    checkpoint = utility.checkpoint(args.save)

    vessl.configure(organization_name='2302-AI-LEC-MEDI', project_name='Minwoo-Yu')
    vessl.init(message=args.save)

    if checkpoint.ok:
        loader = data.Data(config)
        t = Trainer(config, loader, checkpoint)
        while not t.terminate():
            t.train()
            t.eval()

if __name__ == '__main__':
    main()
