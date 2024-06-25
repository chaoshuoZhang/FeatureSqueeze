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

def main(args):
    dataset = cifar10_data.CIFAR10Data(args)
    test_dataloader = dataset.test_dataloader()

    print("-------------数据加载完成-------------------")
    my_model = resnet18()
    state_dict = torch.load(args.model_path + args.model_name + ".pt")
    my_model.load_state_dict(state_dict)

    my_model.eval()  # for evaluation
    print("-------------模型加载完成-------------------")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    my_model = my_model.to(device)
    loss_fn = nn.CrossEntropyLoss(reduction="mean")
    optimizer = torch.optim.Adam(my_model.parameters(), lr=1e-3)

    print("-------------对抗样本测试-------------------")

    # 评估模型
    report = EasyDict(nb_test=0, correct=0, correct_fgm=0, correct_pgd=0)
    for x, y in tqdm(test_dataloader):
        x, y = x.to(device), y.to(device)
        # x_fgm = fast_gradient_method(my_model, x, args.eps, np.inf)
        x_pgd = projected_gradient_descent(my_model, x, args.eps, 0.01, 40, np.inf)

        _, y_pred = my_model(x).max(1)
        # _, y_pred_fgm = my_model(x_fgm).max(1)
        _, y_pred_pgd = my_model(x_pgd).max(1)

        report.nb_test += y.size(0)
        report.correct += y_pred.eq(y).sum().item()
        # report.correct_fgm += y_pred_fgm.eq(y).sum().item()
        report.correct_pgd += y_pred_pgd.eq(y).sum().item()
        break

    print("Test accuracy on clean examples (%): {:.3f}".format(100 * report.correct / report.nb_test))
    # print("Test accuracy on FGM adversarial examples (%): {:.3f}".format(100 * report.correct_fgm / report.nb_test))
    print("Test accuracy on PGD adversarial examples (%): {:.3f}".format(100 * report.correct_pgd / report.nb_test))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate a ResNet18 model on CIFAR-10')
    parser.add_argument('--model_path', type=str, default='./model/save/state_dicts/')
    parser.add_argument('--model_name', type=str, default='resnet18')

    # parser.add_argument('--model_version', type=str, default='custom', help='Model version: "custom" or "pretrained"')
    parser.add_argument('--batch_size', type=int, default=500, help='Batch size for evaluation')
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--data_dir', type=str, default='./mydataset')

    parser.add_argument('--nb_epochs', type=int, default=8)
    parser.add_argument('--eps', type=float, default=0.3)
    args = parser.parse_args()
    main(args)
