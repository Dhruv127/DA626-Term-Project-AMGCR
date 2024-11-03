import torch.optim as optim
import torch
import random
import datetime
import os
from parser import parse_args
from utils import *
from load_data import *
from model import *
from tqdm import tqdm
from time import time
from copy import deepcopy

args = parse_args()

if args.gpu >= 0 and torch.cuda.is_available():
    args.device = "cuda:{}".format(args.gpu)
else:
    args.device = "cpu"

seed = args.seed
args.dataset = 'amazon'
args.epoch = 5
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

args.data_path = "/kaggle/input/aaaaaaaamazon-dataset/AMGCR/data/"

print(args)

if __name__ == '__main__':
    data_generator = Data(args)
    n, m = data_generator.n_users, data_generator.n_items

    adj_norm = data_generator.adj_norm
    edge_index = data_generator.edge_index

   # Define hyperparameter values for individual tuning
    # d_values = [32, 64]
    # batch_size_values = [10240]
    lambda1_values = [0.2, 0.3, 0.4, 0.5]
    lambda2_values = [0.1, 0.2, 0.3, 0.4, 0.5]

    # Function to train and evaluate the model for a given setting
    def train_and_evaluate(param_name, param_value):
        setattr(args, param_name, param_value)
       
        model = AMGCR(n, m, adj_norm, edge_index, args).to(args.device)
        optimizer = optim.Adam(model.parameters(), weight_decay=0, lr=args.lr)

        print(f"Start Training for {param_name}={param_value}")
        stopping_step = 0
        last_state_dict = None
        max_recall = 0.0
        for epoch in range(args.epoch):
            epoch_loss = 0
            epoch_loss_bpr = 0
            epoch_loss_cl = 0
            epoch_loss_pr = 0

            n_samples = data_generator.uniform_sample()
            n_batch = int(np.ceil(n_samples / args.batch_size))

            model.train()
            for idx in tqdm(range(n_batch)):
                optimizer.zero_grad()

                user, pos, neg = data_generator.mini_batch(idx)
                item = np.concatenate([pos, neg], axis=0)

                loss, loss_bpr, loss_cl, loss_pr = model(user, item, pos, neg)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.cpu().item()
                epoch_loss_bpr += loss_bpr.cpu().item()
                epoch_loss_cl += loss_cl.cpu().item()
                epoch_loss_pr += loss_pr.cpu().item()
            epoch_loss = epoch_loss/n_batch
            epoch_loss_bpr = epoch_loss_bpr/n_batch
            epoch_loss_cl = epoch_loss_cl/n_batch
            epoch_loss_pr = epoch_loss_pr/n_batch

            print('Epoch:', epoch, 'Loss:', epoch_loss, 'loss_bpr:', epoch_loss_bpr, 'loss_cl:', epoch_loss_cl, \
                'loss_pr:', epoch_loss_pr)
            if epoch % 5 == 0 or epoch == args.epoch - 1:  # valid every 10 epochs
                model.eval()
                valid_ret = eval_PyTorch(model, data_generator)
                perf_str = 'Epoch: %2d, valid-recall=[%.4f, %.4f], valid-ndcg=[%.4f, %.4f]' % \
                        (epoch, valid_ret['recall'][0], valid_ret['recall'][1], valid_ret['ndcg'][0], valid_ret['ndcg'][1])
                print(perf_str)
                if valid_ret['recall'][0] > max_recall:
                    max_recall = valid_ret['recall'][0]
                    # Ensure the directory exists
                    os.makedirs('/kaggle/working/ckpts', exist_ok=True)
                    # Save the model to the Kaggle working directory
                    torch.save(model.state_dict(), f'/kaggle/working/ckpts/{args.dataset}_{param_name}_{param_value}_best_model.pth')
                
        best_model = AMGCR(n, m, adj_norm, edge_index, args).to(args.device)
        best_model.load_state_dict(torch.load(f'/kaggle/working/ckpts/{args.dataset}_{param_name}_{param_value}_best_model.pth'))
        best_model.forward(user, item, pos, neg)
        test_ret = test_PyTorch(best_model, data_generator)
        perf_str = 'Fianl Testing: test-recall=[%.4f, %.4f], test-ndcg=[%.4f, %.4f]' % \
                (test_ret['recall'][0], test_ret['recall'][1], test_ret['ndcg'][0], test_ret['ndcg'][1])
        print(f"For {param_name}={param_value}")
        print(perf_str)

   # Save the original default values
# default_d = args.d
default_batch_size = args.batch_size
default_lambda1 = args.lambda_1
default_lambda2 = args.lambda_2

# # Tuning 'd'
# for d in d_values:
#     train_and_evaluate('d', d)

# args.d = default_d  # Reset to default 

# Tuning 'batch_size'
# for batch_size in batch_size_values:
#     train_and_evaluate('batch_size', batch_size)

# args.batch_size = default_batch_size  # Reset to default

# Tuning 'lambda_1'
for lambda1 in lambda1_values:
    train_and_evaluate('lambda_1', lambda1)

args.lambda_1 = default_lambda1  # Reset to default

# Tuning 'lambda_2'
for lambda2 in lambda2_values:
    train_and_evaluate('lambda_2', lambda2)

args.lambda_2 = default_lambda2  # Reset to default