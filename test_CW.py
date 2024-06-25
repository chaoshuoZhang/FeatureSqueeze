import numpy as np
import time
from random import random
from attacks.CW.l2_attack import CarliniL2
from mydataset.setup_cifar import CIFAR
from model.resnet import resnet18
import torch
import argparse
import argparse
from mydataset import cifar10_data
from model.resnet import resnet18
import torch
from tqdm import tqdm
import numpy as np
import torch.nn as nn
from easydict import EasyDict
from cleverhans.torch.attacks.fast_gradient_method import fast_gradient_method
from cleverhans.torch.attacks.projected_gradient_descent import projected_gradient_descent
import matplotlib.pyplot as plt

def main(args):
    dataset = cifar10_data.CIFAR10Data(args)
    test_dataloader = dataset.test_dataloader()

    my_model = resnet18()
    state_dict = torch.load(args.model_path + args.model_name + ".pt")
    my_model.load_state_dict(state_dict)

    my_model.eval()  # for evaluation
    device = "cuda" if torch.cuda.is_available() else "cpu"
    my_model = my_model.to(device)
    loss_fn = nn.CrossEntropyLoss(reduction="mean")
    optimizer = torch.optim.Adam(my_model.parameters(), lr=1e-3)

    report = EasyDict(nb_test=0, correct=0, correct_fgm=0, correct_adv=0)
    attack = CarliniL2(my_model, args)
    for x, y in tqdm(test_dataloader):
        x, y = x.to(device), y.to(device)
        # x_fgm = fast_gradient_method(my_model, x, args.eps, np.inf)
        x_adv = attack.attack(x, y)

        _, y_pred = my_model(x).max(1)
        # _, y_pred_fgm = my_model(x_fgm).max(1)
        _, y_pred_adv = my_model(x_adv).max(1)

        first_image = x[0]

        if first_image.is_cuda:
            first_image = first_image.cpu()

        first_image_np = first_image.detach().numpy()

        first_image_np = np.transpose(first_image_np, (1, 2, 0))

        if first_image.ndim == 2:
            plt.imshow(first_image_np, cmap='gray')
        else:
            plt.imshow(first_image_np)

        plt.axis('off')

        plt.savefig('first_image_CW.png', bbox_inches='tight', pad_inches=0)

        first_image_adv = x_adv[0]

        if first_image_adv.is_cuda:
            first_image_adv = first_image_adv.cpu()

        first_image_adv_np = first_image_adv.detach().numpy()

        first_image_adv_np = np.transpose(first_image_adv_np, (1, 2, 0))

        if first_image.ndim == 2:
            plt.imshow(first_image_adv_np, cmap='gray')
        else:
            plt.imshow(first_image_adv_np)

        plt.axis('off')

        plt.savefig('first_image_adv_CW.png', bbox_inches='tight', pad_inches=0)

        report.nb_test += y.size(0)
        report.correct += y_pred.eq(y).sum().item()
        # report.correct_fgm += y_pred_fgm.eq(y).sum().item()
        report.correct_adv += y_pred_adv.eq(y).sum().item()
        break

    print("Test accuracy on clean examples (%): {:.3f}".format(100 * report.correct / report.nb_test))
    # print("Test accuracy on FGM adversarial examples (%): {:.3f}".format(100 * report.correct_fgm / report.nb_test))
    print("Test accuracy on PGD adversarial examples (%): {:.3f}".format(100 * report.correct_adv / report.nb_test))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate a ResNet18 model on CIFAR-10')
    parser.add_argument('--model_path', type=str, default='./model/save/state_dicts/')
    parser.add_argument('--model_name', type=str, default='resnet18')
    parser.add_argument('--device', type=torch.device, default=torch.device('cuda'))

    # parser.add_argument('--model_version', type=str, default='custom', help='Model version: "custom" or "pretrained"')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size for evaluation')
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--data_dir', type=str, default='./mydataset')

    parser.add_argument('--eps_list', type=list, default=[0.0, 0.1])
    parser.add_argument('--norm', type=str, default='L2')
    parser.add_argument('--attack', type=str, default='CW')
    parser.add_argument('--detector', type=str, default='reduce')
    parser.add_argument('--sqeeze_list', type=list, default=['reduce', 'median_filter'])

    args = parser.parse_args()
    main(args)
