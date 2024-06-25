import argparse
from tqdm import tqdm
import numpy as np
from detect.squeeze import reduce_precision_torch
from mydataset import cifar10_data
from model.resnet import resnet18
import torch
from attacks.auto_attack.autoattack.autoattack import AutoAttack
import os
from utlis.detection import train_detector
import torch.nn as nn
from detect.squeeze import reduce_precision_torch, median_filter_torch, non_local_means_denoising_color
from cleverhans.torch.attacks.projected_gradient_descent import projected_gradient_descent
from utlis.th_utils import th_model_eval_distance
from attacks.CW.l0_attack import CarliniL0
from attacks.CW.l2_attack import CarliniL2
from attacks.CW.li_attack import CarliniLi


# from attacks.simba_pytorch.simba import SimBA


class prediction_median_filter(nn.Module):
    def __init__(self, model, width):
        super(prediction_median_filter, self).__init__()
        self.model = model
        self.width = width

    def forward(self, x):
        x = median_filter_torch(x, self.width)
        return self.model(x)


class prediction_non_local_means_denoising_color(nn.Module):
    def __int__(self, model):
        super(prediction_non_local_means_denoising_color, self).__int__()
        self.model = model

    def forward(self, x):
        x = non_local_means_denoising_color(x)
        return self.model(x)


class prediction_non_local(nn.Module):
    def __init__(self, model):
        super(prediction_non_local, self).__init__()
        self.model = model

    def forward(self, x):
        device = x.device
        x_np = x.cpu().detach().numpy().transpose(0, 2, 3, 1).astype(np.uint8)  # CHW to HWC and to uint8
        x_np = non_local_means_denoising_color(x_np)
        x = torch.from_numpy(x_np.transpose(0, 3, 1, 2)).float().div(255.).to(device)  # HWC back to CHW, to Tensor, normalize, and to original device
        return self.model(x)

class prediction_reduce_precision(nn.Module):
    def __init__(self, model, npp=2 ** 4):
        super(prediction_reduce_precision, self).__init__()
        self.model = model
        self.npp = npp

    def forward(self, x):
        x = reduce_precision_torch(x, self.npp)
        return self.model(x)


def get_PGD_adv_examples(model, dataloader, args):
    device = args.device
    pgd_examples = []
    eps_list = args.eps_list  # list=[0,0.1,0.2]
    if args.norm == 'Linf':
        norm = np.inf
    else:
        norm = 2
    pgd_dict = {eps: [] for eps in eps_list}

    for x, y in tqdm(dataloader):
        x, y = x.to(device), y.to(device)
        for eps in eps_list:
            x_pgd = projected_gradient_descent(model, x, eps, 0.01, 40, norm)
            pgd_dict[eps].append(x_pgd.cpu().detach().numpy())

    # Convert list of arrays to single array for each epsilon
    for eps in eps_list:
        pgd_dict[eps] = np.concatenate(pgd_dict[eps], axis=0)

    return pgd_dict


def get_autoPGD_adv_examples(model, dataloader, args):
    device = args.device
    eps_list = args.eps_list
    norm = args.norm
    adv_examples_dict = {eps: [] for eps in eps_list}

    for eps in eps_list:
        adversary = AutoAttack(model, norm=norm, eps=eps, version='standard')
        x_adv_list = []

        for images, labels in tqdm(dataloader):
            images = images.to(device)
            labels = labels.to(device)  # 将 labels 转换为 9 - labels
            x_adv = adversary.run_standard_evaluation(images, labels, bs=args.batch_size)
            x_adv_list.append(x_adv.cpu().detach().numpy())

        adv_examples_dict[eps] = np.concatenate(x_adv_list, axis=0)

    return adv_examples_dict


def get_simba_adv_examples(model, dataloader, args):
    attack = SimBA(model, 'cifar', 32)
    device = args.device
    eps_list = args.eps_list

    cw_dict = {eps: [] for eps in eps_list}

    for x, y in tqdm(dataloader):
        x, y = x.to(device), y.to(device)
        x_simba = attack, probs, succs, queries, l2_norms, linf_norms = attack.simba_batch(x, y, args.max_iters,
                                                                                           args.freq_dims,
                                                                                           args.eps_list[1],
                                                                                           args.strike, args.linf_bound,
                                                                                           args.order, args.targeted,
                                                                                           args.pixel_attack)
        cw_dict[0.0].append(x.cpu().detach().numpy())
        cw_dict[0.1].append(x_simba.cpu().detach().numpy())

    for eps in eps_list:
        cw_dict[eps] = np.concatenate(cw_dict[eps], axis=0)

    return cw_dict


def get_CW_adv_examples(model, dataloader, args):
    device = args.device
    eps_list = args.eps_list
    norm = args.norm
    if norm == 'Linf':
        attack = CarliniLi(model, args)
    elif norm == 'L0':
        attack = CarliniL0(model, args)
    else:
        attack = CarliniL2(model, args)
    cw_dict = {eps: [] for eps in eps_list}

    for x, y in tqdm(dataloader):
        x, y = x.to(device), y.to(device)

        x_cw = attack.attack(x, y)
        cw_dict[0.0].append(x.cpu().detach().numpy())
        cw_dict[0.1].append(x_cw.cpu().detach().numpy())

    for eps in eps_list:
        cw_dict[eps] = np.concatenate(cw_dict[eps], axis=0)

    return cw_dict


def calculate_l1_distance(predictions_orig, predictions_squeezed, X_test_adv, args, csv_fpath):
    print("\n===Calculating L1 distance with feature squeezing...")
    eps_list = args.eps_list
    l1_dist = np.zeros((len(X_test_adv[eps_list[0]]), len(eps_list)))
    for i, eps in enumerate(eps_list):
        print("Epsilon=", eps)
        X_test_adv_ = X_test_adv[eps]
        l1_dist_vec = th_model_eval_distance(predictions_orig, predictions_squeezed, X_test_adv_, args)
        l1_dist[:, i] = l1_dist_vec

    np.savetxt(csv_fpath, l1_dist, delimiter=',')
    print("---Results are stored in ", csv_fpath, '\n')
    return l1_dist


def main(args):
    dataset = cifar10_data.CIFAR10Data(args)
    print("-------------数据加载完成-------------------")
    my_model = resnet18()
    state_dict = torch.load(
        args.model_path + args.model_name + ".pt"
    )
    my_model.load_state_dict(state_dict)

    my_model.eval()  # for evaluation
    my_model.to(args.device)
    print("-------------模型加载完成-------------------")

    test_dataloader = dataset.test_dataloader()
    # train_dataloader = dataset.train_dataloader()
    # 评估模型
    correct = 0
    total = 0
    print("---------------测试开始-------------------")
    with torch.no_grad():
        for data in test_dataloader:
            images, labels = data
            outputs = my_model(images.to(args.device))
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted.cpu().detach() == labels).sum().item()
            break
        print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))
    if args.detector == 'jiont':
        print("---------------构建对抗例子-------------------")
        if args.attack == 'PGD':
            adv_x_dict = get_PGD_adv_examples(my_model, test_dataloader, args)
        elif args.attack == 'autoPGD':
            adv_x_dict = get_autoPGD_adv_examples(my_model, test_dataloader, args)
        elif args.attack == 'CW':
            adv_x_dict = get_CW_adv_examples(my_model, test_dataloader, args)

        predictor = my_model
        predictor_squeeze_reduce = prediction_reduce_precision(my_model)
        predictor_squeeze_median = prediction_median_filter(my_model, args)
        predictor_squeeze_color = prediction_non_local_means_denoising_color(my_model, args)
        nb_examples = 10000
        l1_dist = np.zeros((len(adv_x_dict[args.eps_list[0]]), len(args.eps_list)))
        for squeeze in args.squeeze:
            if squeeze == 'reduce':
                predictor_squeeze = predictor_squeeze_reduce
            elif squeeze == 'median_filter':
                predictor_squeeze = predictor_squeeze_median
            else:
                predictor_squeeze = predictor_squeeze_color
            # Calculate L1 distance on prediction for adversarial detection.
            csv_fpath = args.model_name + f"_{args.squeeze}_{args.attack}_{args.norm}_l1distance.csv"
            csv_fpath = os.path.join(args.model_path, csv_fpath)
            print("---------------加载L1距离-------------------")
            if not os.path.isfile(csv_fpath):
                l1_dist_cur = calculate_l1_distance(predictor, predictor_squeeze, adv_x_dict, args, csv_fpath)
            else:
                l1_dist_cur = np.loadtxt(csv_fpath, delimiter=',')
            l1_dist = np.maximum(l1_dist, l1_dist_cur)

        size_train = size_val = int(nb_examples / 2)
        col_id_leg = [0]
        # Selected epsilon: 0.1, 0.2, 0.3
        col_id_adv = [i + 1 for i in range(len(args.eps_list) - 1)]

        x_train = np.hstack([l1_dist[:size_train, col_id] for col_id in col_id_leg + col_id_adv])
        y_train = np.hstack([np.zeros(size_train * len(col_id_leg)), np.ones(size_train * len(col_id_adv))])

        x_val = np.hstack([l1_dist[-size_val:, col_id] for col_id in col_id_leg + col_id_adv])
        y_val = np.hstack([np.zeros(size_val * len(col_id_leg)), np.ones(size_val * len(col_id_adv))])
        print(f"attack={args.attack} norm={args.norm} detector={args.sqeeze_list}")
        train_detector(x_train, y_train, x_val, y_val)

    else:
        predictor = my_model
        if args.detector == 'reduce':
            predictor_squeeze = prediction_reduce_precision(my_model)
        elif args.detector == 'median_filter':
            predictor_squeeze = prediction_median_filter(my_model, args.width)
        else:
            predictor_squeeze = prediction_non_local(my_model)
        nb_examples = 10000
        print("---------------构建对抗例子-------------------")
        if args.attack == 'PGD':
            adv_x_dict = get_PGD_adv_examples(my_model, test_dataloader, args)
        elif args.attack == 'autoPGD':
            adv_x_dict = get_autoPGD_adv_examples(my_model, test_dataloader, args)
        elif args.attack == 'CW':
            adv_x_dict = get_CW_adv_examples(my_model, test_dataloader, args)
        elif args.attack == 'CW':
            adv_x_dict = get_simba_adv_examples(my_model, test_dataloader, args)

        # adv_x_dict = get_autoPGD_adv_examples(sess, x, predictions, X_test, eps_list, model_name, nb_examples,
        #                                       result_folder)

        # Calculate L1 distance on prediction for adversarial detection.

        csv_fpath = args.model_name + f"_{args.detector}_{args.attack}_{args.norm}_l1distance.csv"
        csv_fpath = os.path.join(args.model_path, csv_fpath)
        print("---------------加载L1距离-------------------")
        if not os.path.isfile(csv_fpath):
            l1_dist = calculate_l1_distance(predictor, predictor_squeeze, adv_x_dict, args, csv_fpath)
        else:
            l1_dist = calculate_l1_distance(predictor, predictor_squeeze, adv_x_dict, args,
                                            csv_fpath)  # l1_dist = np.loadtxt(csv_fpath, delimiter=',')
        # l1_dist = calculate_l1_distance_PGD(predictor, predictor_squeeze, adv_x_dict, args, csv_fpath)
        size_train = size_val = int(nb_examples / 2)
        col_id_leg = [0]
        # Selected epsilon: 0.1, 0.2, 0.3
        col_id_adv = [i for i in range(len(args.eps_list) - 1)]

        x_train = np.hstack([l1_dist[:size_train, col_id] for col_id in col_id_leg + col_id_adv])
        y_train = np.hstack([np.zeros(size_train * len(col_id_leg)), np.ones(size_train * len(col_id_adv))])

        x_val = np.hstack([l1_dist[-size_val:, col_id] for col_id in col_id_leg + col_id_adv])
        y_val = np.hstack([np.zeros(size_val * len(col_id_leg)), np.ones(size_val * len(col_id_adv))])
        print(f"attack={args.attack} norm={args.norm} detector={args.detector}")
        train_detector(x_train, y_train, x_val, y_val)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate a ResNet18 model on CIFAR-10')
    parser.add_argument('--model_path', type=str, default='./model/save/state_dicts/')
    parser.add_argument('--model_name', type=str, default='resnet18')
    parser.add_argument('--device', type=torch.device, default=torch.device('cuda'))

    # parser.add_argument('--model_version', type=str, default='custom', help='Model version: "custom" or "pretrained"')
    parser.add_argument('--batch_size', type=int, default=500, help='Batch size for evaluation')
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--data_dir', type=str, default='./mydataset')

    parser.add_argument('--eps_list', type=list, default=[0.0, 0.1])
    parser.add_argument('--norm', type=str, default='Linf')
    parser.add_argument('--attack', type=str, default='CW')
    parser.add_argument('--detector', type=str, default='median_filter')
    parser.add_argument('--sqeeze_list', type=list, default=['reduce', 'median_filter'])
    parser.add_argument('--width', type=int, default=2)
    parser.add_argument('--num_iters', type=int, default=1000, help='maximum number of iterations, 0 for unlimited')
    parser.add_argument('--log_every', type=int, default=100, help='log every n iterations')
    parser.add_argument('--linf_bound', type=float, default=0.0, help='L_inf bound for frequency space attack')
    parser.add_argument('--freq_dims', type=int, default=32, help='dimensionality of 2D frequency space')
    parser.add_argument('--order', type=str, default='rand', help='(random) order of coordinate selection')
    parser.add_argument('--stride', type=int, default=7, help='stride for block order')

    args = parser.parse_args()
    main(args)
