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
from attacks.auto_attack.autoattack.autoattack import AutoAttack
import matplotlib.pyplot as plt


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
    report = EasyDict(nb_test=0, correct=0, correct_fgm=0, correct_adv=0)
    adversary = AutoAttack(my_model, norm="L1", eps=0.3, version='standard')
    x_adv_list = []

    for images, y in tqdm(test_dataloader):
        images, y = images.to(device), y.to(device)


        x_adv = adversary.run_standard_evaluation(images, y, bs=args.batch_size)

        _, y_pred = my_model(images).max(1)
        # _, y_pred_fgm = my_model(x_fgm).max(1)
        _, y_pred_adv = my_model(x_adv).max(1)

        first_image = images[0]  # 形状为 (3, 32, 32)

        if first_image.is_cuda:
            first_image = first_image.cpu()

        first_image_np = first_image.numpy()

        first_image_np = np.transpose(first_image_np, (1, 2, 0))

        if first_image.ndim == 2:
            plt.imshow(first_image_np, cmap='gray')  # 对于灰度图，使用灰度色图
        else:
            plt.imshow(first_image_np)  # 对于彩色图，直接显示

        # 隐藏坐标轴
        plt.axis('off')

        plt.savefig('first_image.png', bbox_inches='tight', pad_inches=0)  # 保存为 PNG 文件

        first_image_adv = x_adv[0]  # 形状为 (3, 32, 32)

        if first_image_adv.is_cuda:
            first_image_adv = first_image_adv.cpu()

        first_image_adv_np = first_image_adv.numpy()

        first_image_adv_np = np.transpose(first_image_adv_np, (1, 2, 0))

        if first_image.ndim == 2:
            plt.imshow(first_image_adv_np, cmap='gray')  # 对于灰度图，使用灰度色图
        else:
            plt.imshow(first_image_adv_np)  # 对于彩色图，直接显示

        plt.axis('off')

        plt.savefig('first_image_adv.png', bbox_inches='tight', pad_inches=0)  # 保存为 PNG 文件

        report.nb_test += y.size(0)
        report.correct += y_pred.eq(y).sum().item()
        # report.correct_fgm += y_pred_fgm.eq(y).sum().item()
        report.correct_adv += y_pred_adv.eq(y).sum().item()
        break

    print("Test accuracy on clean examples (%): {:.3f}".format(100 * report.correct / report.nb_test))
    # print("Test accuracy on FGM adversarial examples (%): {:.3f}".format(100 * report.correct_fgm / report.nb_test))
    print("Test accuracy on autoPGD adversarial examples (%): {:.3f}".format(100 * report.correct_adv / report.nb_test))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate a ResNet18 model on CIFAR-10')
    parser.add_argument('--model_path', type=str, default='./model/save/state_dicts/')
    parser.add_argument('--model_name', type=str, default='resnet18')

    # parser.add_argument('--model_version', type=str, default='custom', help='Model version: "custom" or "pretrained"')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size for evaluation')
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--data_dir', type=str, default='./mydataset')

    parser.add_argument('--nb_epochs', type=int, default=8)
    parser.add_argument('--eps', type=float, default=0.3)
    args = parser.parse_args()
    main(args)
