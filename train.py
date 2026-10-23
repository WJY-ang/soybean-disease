# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import argparse
import os
import time
import csv
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_recall_fscore_support

from datasets import build_dataset
import utils
from utils import str2bool
from optim_factory import create_optimizer
from timm.loss import LabelSmoothingCrossEntropy

# 使用 MLCA+MDConv 模型（保留原始 MLP）
import models.ConvNext_MLCA_MDConv as convnextv2

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def get_args_parser():
    parser = argparse.ArgumentParser('ConvNeXt V2 双模块训练 (MLCA+MDConv 并行) - 弱增强调整版', add_help=False)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--epochs', default=100, type=int)  # 增加轮数，让模型充分收敛
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--imagenet_default_mean_and_std', default=True, type=str2bool)

    # 关闭所有强增强（保持弱增强）
    parser.add_argument('--color_jitter', type=float, default=0.0)
    parser.add_argument('--aa', type=str, default='')
    parser.add_argument('--train_interpolation', type=str, default='bicubic')
    parser.add_argument('--reprob', type=float, default=0.0)
    parser.add_argument('--remode', type=str, default='pixel')
    parser.add_argument('--recount', type=int, default=1)
    parser.add_argument('--resplit', action='store_true', default=False)
    parser.add_argument('--mixup', type=float, default=0.0)
    parser.add_argument('--cutmix', type=float, default=0.0)
    parser.add_argument('--cutmix_minmax', type=float, nargs='+', default=None)
    parser.add_argument('--crop_pct', type=float, default=0.9)
    parser.add_argument('--use_amp', default=True, type=str2bool)

    # ========== 关键调整：降低正则化强度，避免双模块欠拟合 ==========
    parser.add_argument('--smoothing', type=float, default=0.0)  # 原0.1 → 0.0，去掉标签平滑
    parser.add_argument('--model', default='convnextv2_nano', type=str)
    parser.add_argument('--input_size', default=224, type=int)
    parser.add_argument('--drop_path', type=float, default=0.1)  # 原0.2 → 0.1，降低随机深度
    parser.add_argument('--weight_decay', type=float, default=0.05)  # 原0.1 → 0.05，降低权重衰减
    parser.add_argument('--lr', type=float, default=1.2e-4)  # 原1e-4 → 1.2e-4，略微提高学习率
    parser.add_argument('--min_lr', type=float, default=1e-6)
    parser.add_argument('--warmup_epochs', type=int, default=3)  # 原5 → 3，缩短预热
    parser.add_argument('--opt', default='adamw', type=str)
    # =============================================================

    # 路径
    parser.add_argument('--finetune', default=r'D:\ConvNext\ConvNeXt-V2-main\pretrained_weights\convnextv2_nano.pt')
    parser.add_argument('--data_path', default=r'D:\数据集\大豆数据集\大豆划分\train', type=str)
    parser.add_argument('--val_data_path', default=r'D:\数据集\大豆数据集\大豆划分\val', type=str)
    parser.add_argument('--test_data_path', default=r'D:\数据集\大豆数据集\大豆划分\test', type=str)
    parser.add_argument('--nb_classes', default=4, type=int)
    parser.add_argument('--data_set', default='image_folder', type=str)
    parser.add_argument('--output_dir', default='./run/MLCA+MDConv5.5', type=str)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--pin_mem', type=str2bool, default=True)
    parser.add_argument('--eval', type=str2bool, default=False)
    return parser


def train_one_epoch(model, criterion, data_loader, optimizer, device, args, scaler):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    pbar = tqdm(data_loader, desc="Train", leave=False)

    for imgs, labels in pbar:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        with torch.amp.autocast('cuda', enabled=args.use_amp):
            outputs = model(imgs)
            loss = criterion(outputs, labels)

        if torch.isnan(loss) or torch.isinf(loss):
            continue

        if scaler:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        total_correct += (outputs.argmax(1) == labels).sum().item()
        total_samples += imgs.size(0)

    return {"loss": total_loss / total_samples, "acc": total_correct / total_samples}


def evaluate(model, criterion, data_loader, device):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []
    total_samples = 0
    with torch.no_grad():
        for imgs, labels in tqdm(data_loader, desc="Val", leave=False):
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * imgs.size(0)
            all_preds.extend(outputs.argmax(1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            total_samples += imgs.size(0)
    acc = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='macro')
    return {"loss": total_loss / total_samples, "acc": acc, "precision": precision, "recall": recall, "f1": f1,
            "preds": all_preds, "labels": all_labels}


def plot_curves(epochs, train_loss, val_loss, train_acc, val_acc, save_dir):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss, label='Train Loss')
    plt.plot(epochs, val_loss, label='Val Loss')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_acc, label='Train Acc')
    plt.plot(epochs, val_acc, label='Val Acc')
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'training_curves.png'), dpi=300)
    plt.close()


def load_pretrained_main_path(model, ckpt_path, exclude_keywords=['mlca', 'mdconv', 'fusion_alpha']):
    """只加载标准 ConvNeXt‑V2 主路径的预训练权重，新增模块保持随机初始化"""
    if not os.path.exists(ckpt_path):
        print(f"⚠️ 预训练权重不存在: {ckpt_path}，将从头训练")
        return model
    ckpt = torch.load(ckpt_path, map_location='cpu')
    if 'model' in ckpt:
        ckpt = ckpt['model']
    elif 'state_dict' in ckpt:
        ckpt = ckpt['state_dict']
    new_ckpt = {}
    for k, v in ckpt.items():
        if k.startswith('module.'):
            k = k[7:]
        new_ckpt[k] = v
    ckpt = new_ckpt

    model_dict = model.state_dict()
    loaded_keys = []
    for k, v in ckpt.items():
        if k in model_dict and v.shape == model_dict[k].shape:
            if any(x in k for x in exclude_keywords):
                continue
            model_dict[k] = v
            loaded_keys.append(k)
    model.load_state_dict(model_dict, strict=False)
    print(f"✅ 成功加载 {len(loaded_keys)} 个主路径参数（新增模块保持随机初始化）")
    return model


def compute_extra_metrics(model, device, test_stats, class_names, save_dir, input_size=224, batch_size=16):
    print("\n" + "=" * 70)
    print("模型额外指标分析".center(70))
    print("=" * 70)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"📊 总参数量: {total_params / 1e6:.2f}M")
    print(f"📊 可训练参数量: {trainable_params / 1e6:.2f}M")

    try:
        from thop import profile
        input_tensor = torch.randn(1, 3, input_size, input_size).to(device)
        flops, _ = profile(model, inputs=(input_tensor,), verbose=False)
        print(f"📈 FLOPs: {flops / 1e9:.2f}G")
    except ImportError:
        print("⚠️ 未安装 thop，跳过 FLOPs 计算 (pip install thop)")

    model.eval()
    dummy = torch.randn(batch_size, 3, input_size, input_size).to(device)
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    start = time.time()
    with torch.no_grad():
        for _ in range(100):
            _ = model(dummy)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    end = time.time()
    avg_time_ms = (end - start) / 100 * 1000
    print(f"⚡ 平均推理时间: {avg_time_ms:.2f} ms/样本 (batch_size={batch_size})")
    print(f"⚡ 吞吐量: {1000 / avg_time_ms:.2f} 样本/秒")

    # 混淆矩阵
    cm = confusion_matrix(test_stats['labels'], test_stats['preds'])

    # 每个类别的指标
    per_class_metrics = []
    total_samples = len(test_stats['labels'])
    for i, name in enumerate(class_names):
        TP = cm[i, i]
        FP = cm[:, i].sum() - TP
        FN = cm[i, :].sum() - TP
        TN = cm.sum() - TP - FP - FN

        precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        specificity = TN / (TN + FP) if (TN + FP) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        accuracy = (TP + TN) / total_samples  # 整体准确率中该类预测正确的比例

        per_class_metrics.append({
            'class': name,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'Specificity': specificity,
            'F1-score': f1,
            'Support': TP + FN  # 该类真实样本数
        })

    # 打印表格
    print("\n🔍 各类别详细指标：")
    print("-" * 110)
    print(
        f"{'Class':<12} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'Specificity':<12} {'F1-score':<10} {'Support':<8}")
    print("-" * 110)
    for m in per_class_metrics:
        print(
            f"{m['class']:<12} {m['Accuracy']:.4f}     {m['Precision']:.4f}     {m['Recall']:.4f}     {m['Specificity']:.4f}       {m['F1-score']:.4f}     {m['Support']:<8}")
    print("-" * 110)

    # 保存为 CSV
    csv_path = os.path.join(save_dir, 'per_class_metrics.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['class', 'Accuracy', 'Precision', 'Recall', 'Specificity', 'F1-score',
                                               'Support'])
        writer.writeheader()
        writer.writerows(per_class_metrics)
    print(f"✅ 每类指标已保存至: {csv_path}")

    # 额外保存特异性（原有功能）
    specificities = [m['Specificity'] for m in per_class_metrics]
    print("🔍 平均特异度: {:.4f}".format(sum(specificities) / len(specificities)))

    with open(os.path.join(save_dir, 'extra_metrics.txt'), 'w') as f:
        f.write(f"总参数量: {total_params / 1e6:.2f}M\n")
        f.write(f"可训练参数量: {trainable_params / 1e6:.2f}M\n")
        f.write(f"平均推理时间: {avg_time_ms:.2f} ms/样本 (batch_size={batch_size})\n")
        for i, name in enumerate(class_names):
            f.write(f"Specificity_{name}: {specificities[i]:.4f}\n")
        f.write(f"Average Specificity: {sum(specificities) / len(specificities):.4f}\n")
        f.write("\n每类详细指标（Accuracy, Precision, Recall, Specificity, F1-score）:\n")
        for m in per_class_metrics:
            f.write(
                f"{m['class']}: Acc={m['Accuracy']:.4f}, Prec={m['Precision']:.4f}, Rec={m['Recall']:.4f}, Spec={m['Specificity']:.4f}, F1={m['F1-score']:.4f}, Support={m['Support']}\n")


def main(args):
    utils.init_distributed_mode(args)
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    cudnn.benchmark = True
    os.makedirs(args.output_dir, exist_ok=True)

    # 构建数据集
    dataset_train, _ = build_dataset(is_train=True, args=args)
    args.eval_data_path = args.val_data_path
    dataset_val, _ = build_dataset(is_train=False, args=args)

    # 手动覆盖训练集 transform 为弱增强（与验证集对齐，仅多随机翻转）
    train_transform = transforms.Compose([
        transforms.Resize(248),
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    dataset_train.transform = train_transform
    print("✅ 训练集 transform 已与验证集对齐（弱增强：Resize+CenterCrop+Flip）")

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=args.pin_mem, drop_last=True
    )
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=args.pin_mem
    )

    # 模型（并行 MLCA+MDConv）
    model = convnextv2.__dict__[args.model](
        num_classes=args.nb_classes,
        drop_path_rate=args.drop_path
    )

    # 加载预训练主路径权重（跳过新增模块）
    if args.finetune:
        model = load_pretrained_main_path(model, args.finetune, exclude_keywords=['mlca', 'mdconv', 'fusion_alpha'])

    model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"📊 模型总参数量: {total_params / 1e6:.2f}M")

    optimizer = create_optimizer(args, model)
    scaler = torch.amp.GradScaler('cuda') if args.use_amp else None
    criterion = LabelSmoothingCrossEntropy(
        smoothing=args.smoothing) if args.smoothing > 0 else torch.nn.CrossEntropyLoss()

    epochs_list, train_losses, train_accs = [], [], []
    val_losses, val_accs, val_precs, val_recs, val_f1s = [], [], [], [], []
    best_val_acc = 0.0
    csv_path = os.path.join(args.output_dir, 'results.csv')
    with open(csv_path, 'w') as f:
        f.write('epoch,train_loss,train_acc,val_loss,val_acc,val_precision,val_recall,val_f1\n')

    class_names = ["healthy", "grey spot", "macular", "mosaic"]
    print("=" * 70)
    print("开始训练 MLCA+MDConv 并行双模块模型（弱增强 + 低正则化）".center(70))
    print("=" * 70)

    for epoch in range(args.epochs):
        start = time.time()
        train_stats = train_one_epoch(model, criterion, data_loader_train, optimizer, device, args, scaler)
        val_stats = evaluate(model, criterion, data_loader_val, device)
        epoch_time = time.time() - start

        epochs_list.append(epoch)
        train_losses.append(train_stats['loss'])
        train_accs.append(train_stats['acc'])
        val_losses.append(val_stats['loss'])
        val_accs.append(val_stats['acc'])
        val_precs.append(val_stats['precision'])
        val_recs.append(val_stats['recall'])
        val_f1s.append(val_stats['f1'])

        with open(csv_path, 'a') as f:
            f.write(
                f"{epoch},{train_stats['loss']:.5f},{train_stats['acc']:.5f},{val_stats['loss']:.5f},{val_stats['acc']:.5f},{val_stats['precision']:.5f},{val_stats['recall']:.5f},{val_stats['f1']:.5f}\n")

        if val_stats['acc'] > best_val_acc:
            best_val_acc = val_stats['acc']
            torch.save({'model': model.state_dict()}, os.path.join(args.output_dir, "checkpoint-best.pth"))

        print(f"Epoch {epoch:2d} | Train Loss {train_stats['loss']:.4f} Acc {train_stats['acc']:.4f} | "
              f"Val Loss {val_stats['loss']:.4f} Acc {val_stats['acc']:.4f} | Prec {val_stats['precision']:.4f} Rec {val_stats['recall']:.4f} F1 {val_stats['f1']:.4f} | Time {epoch_time:.1f}s")

        if (epoch + 1) % 10 == 0 or epoch == args.epochs - 1:
            plot_curves(epochs_list, train_losses, val_losses, train_accs, val_accs, args.output_dir)

    # 测试
    print("\n" + "=" * 70)
    print("加载最佳模型测试".center(70))
    print("=" * 70)
    model.load_state_dict(torch.load(os.path.join(args.output_dir, "checkpoint-best.pth"), map_location='cpu')['model'])
    args.eval_data_path = args.test_data_path
    dataset_test, _ = build_dataset(is_train=False, args=args)
    test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False,
                                              num_workers=args.num_workers, pin_memory=args.pin_mem)
    test_stats = evaluate(model, criterion, test_loader, device)
    print(f"🎉 最终测试精度: {test_stats['acc']:.4f} ({test_stats['acc']:.2%})")
    print(f"   精确率 (Precision): {test_stats['precision']:.4f}")
    print(f"   召回率 (Recall/Sensitivity): {test_stats['recall']:.4f}")
    print(f"   F1-score: {test_stats['f1']:.4f}")

    cm = confusion_matrix(test_stats['labels'], test_stats['preds'])
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.savefig(os.path.join(args.output_dir, 'confusion_matrix.png'), dpi=300)
    plt.close()

    # 保存最终报告（含每类详细指标）
    with open(os.path.join(args.output_dir, 'final_report.txt'), 'w') as f:
        f.write("最终测试报告（MLCA+MDConv 并行双模块模型 - 弱增强低正则化版）\n")
        f.write(f"最佳验证精度: {best_val_acc:.4f}\n")
        f.write(f"最终测试精度: {test_stats['acc']:.4f}\n")
        f.write(f"宏观平均精确率: {test_stats['precision']:.4f}\n")
        f.write(f"宏观平均召回率: {test_stats['recall']:.4f}\n")
        f.write(f"宏观平均F1: {test_stats['f1']:.4f}\n\n")
        f.write("各类别详细指标（Accuracy为整体准确率中该类预测正确的比例）：\n")
        f.write(classification_report(test_stats['labels'], test_stats['preds'], target_names=class_names, digits=4))
        f.write("\n额外计算的特异度（Specificity）等见 per_class_metrics.csv 和 extra_metrics.txt\n")

    # 计算并输出每个类别的完整指标（包含Accuracy, Precision, Recall, Specificity, F1-score）
    compute_extra_metrics(model, device, test_stats, class_names, args.output_dir,
                          input_size=args.input_size, batch_size=args.batch_size)

    print(f"\n所有结果保存至: {args.output_dir}")


if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)