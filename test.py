# test_script.py
# 输出 Excel 表格：每张图片的 Accuracy 和 置信度
# 【修正版】class_names 顺序与文件夹字母顺序一致，避免标签错位

import os
import csv
import shutil
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
from PIL import Image
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support

import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

import models.ConvNext_MLCA_MDConv as convnextv2


class ImageFolderDataset(Dataset):
    """按训练时的 class_names 顺序加载测试集"""
    def __init__(self, data_path, transform=None, class_names=None):
        self.data_path = data_path
        self.transform = transform
        self.samples = []

        if class_names is not None:
            # ⭐ 强制使用传入的类别顺序（必须与训练时一致）
            self.class_to_idx = {cls: idx for idx, cls in enumerate(class_names)}
            self.class_names = class_names
        else:
            classes = sorted([d for d in os.listdir(data_path)
                              if os.path.isdir(os.path.join(data_path, d))])
            self.class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
            self.class_names = classes

        for cls in self.class_names:
            cls_path = os.path.join(data_path, cls)
            if not os.path.exists(cls_path):
                print(f"⚠️ 跳过不存在的类别目录: {cls_path}")
                continue
            img_files = sorted([f for f in os.listdir(cls_path)
                                if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))])
            for img_name in img_files:
                self.samples.append((
                    os.path.join(cls_path, img_name),
                    self.class_to_idx[cls],
                    cls,
                    img_name
                ))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label, class_name, img_name = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label, img_path, class_name, img_name


def get_test_transform():
    return transforms.Compose([
        transforms.Resize(248),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def load_model(model_path, num_classes=4, device='cuda'):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"❌ 权重文件不存在: {model_path}\n"
                                f"   绝对路径: {os.path.abspath(model_path)}")

    model = convnextv2.__dict__['convnextv2_nano'](
        num_classes=num_classes,
        drop_path_rate=0.1
    )

    checkpoint = torch.load(model_path, map_location='cpu')
    state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()

    print(f"✅ 模型加载成功: {model_path}")
    return model


def compute_per_class_metrics(cm, class_names):
    metrics = []
    total_samples = cm.sum()
    for i, name in enumerate(class_names):
        TP = cm[i, i]
        FP = cm[:, i].sum() - TP
        FN = cm[i, :].sum() - TP
        TN = cm.sum() - TP - FP - FN

        accuracy = (TP + TN) / total_samples if total_samples > 0 else 0.0
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        specificity = TN / (TN + FP) if (TN + FP) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        metrics.append({
            'class': name,
            'Accuracy': round(accuracy, 4),
            'Precision': round(precision, 4),
            'Recall': round(recall, 4),
            'Specificity': round(specificity, 4),
            'F1-score': round(f1, 4),
            'Support': int(TP + FN)
        })
    return metrics


def evaluate_test_set(model, test_loader, device, class_names):
    all_preds, all_labels = [], []
    all_img_names, all_true_classes, all_pred_classes = [], [], []
    all_correct, all_probs, all_conf = [], [], []

    with torch.no_grad():
        for images, labels, img_paths, cls_names, img_names in tqdm(test_loader, desc="测试评估"):
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            preds = outputs.argmax(1).cpu().numpy()
            labels_np = labels.numpy()

            all_preds.extend(preds)
            all_labels.extend(labels_np)
            all_img_names.extend(img_names)
            all_true_classes.extend([class_names[l] for l in labels_np])
            all_pred_classes.extend([class_names[p] for p in preds])
            all_correct.extend([(p == l) for p, l in zip(preds, labels_np)])
            all_probs.extend(probs)
            # 置信度 = 最大 softmax 概率
            all_conf.extend([float(np.max(p)) for p in probs])

    acc = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='macro', zero_division=0
    )
    cm = confusion_matrix(all_labels, all_preds)
    per_class_metrics = compute_per_class_metrics(cm, class_names)

    per_image_results = []
    for i in range(len(all_img_names)):
        per_image_results.append({
            'image_name': all_img_names[i],
            'true_class': all_true_classes[i],
            'pred_class': all_pred_classes[i],
            'correct': all_correct[i],
            'accuracy': 1.0 if all_correct[i] else 0.0,
            'confidence': all_conf[i],
            'probabilities': all_probs[i].tolist()
        })

    return {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm,
        'per_class_metrics': per_class_metrics,
        'per_image_results': per_image_results,
        'class_names': class_names
    }


# ============================================================
# 导出 Excel
# ============================================================
def export_to_excel(results, output_dir, class_names):
    """
    导出结果到 Excel：
      - Sheet1: 汇总指标
      - Sheet2: 所有图片（含 Accuracy、置信度）
      - Sheet3~: 每个类别一个 Sheet（含该类别每张图的 Accuracy 和 置信度）
    """
    print("\n" + "=" * 70)
    print("导出 Excel 结果".center(70))
    print("=" * 70)

    excel_path = os.path.join(output_dir, '测试结果.xlsx')

    # ---------- Sheet: 汇总 ----------
    summary_rows = []
    for m in results['per_class_metrics']:
        summary_rows.append({
            '类别': m['class'],
            'Accuracy': m['Accuracy'],
            'Precision': m['Precision'],
            'Recall': m['Recall'],
            'Specificity': m['Specificity'],
            'F1-score': m['F1-score'],
            'Support': m['Support']
        })
    summary_rows.append({
        '类别': '总体(macro)',
        'Accuracy': round(results['accuracy'], 4),
        'Precision': round(results['precision'], 4),
        'Recall': round(results['recall'], 4),
        'Specificity': '',
        'F1-score': round(results['f1'], 4),
        'Support': sum(m['Support'] for m in results['per_class_metrics'])
    })
    df_summary = pd.DataFrame(summary_rows)

    # ---------- Sheet: 所有图片 ----------
    all_rows = []
    for idx, r in enumerate(results['per_image_results'], 1):
        row = {
            '序号': idx,
            '图片名称': r['image_name'],
            '真实类别': r['true_class'],
            '预测类别': r['pred_class'],
            '是否正确': '✓' if r['correct'] else '✗',
            'Accuracy': r['accuracy'],
            '置信度': round(r['confidence'], 4)
        }
        for i, c in enumerate(class_names):
            row[f'P({c})'] = round(r['probabilities'][i], 4)
        all_rows.append(row)
    df_all = pd.DataFrame(all_rows)

    # ---------- 每个类别一个 Sheet ----------
    class_results = {name: [] for name in class_names}
    for r in results['per_image_results']:
        class_results[r['true_class']].append(r)

    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        # 汇总
        df_summary.to_excel(writer, sheet_name='汇总指标', index=False)
        # 所有图片
        df_all.to_excel(writer, sheet_name='所有图片', index=False)

        # 每个类别
        for cls_name in class_names:
            items = class_results[cls_name]
            rows = []
            correct_count = sum(1 for r in items if r['correct'])
            total = len(items)
            cls_acc = correct_count / total if total > 0 else 0.0

            for idx, r in enumerate(items, 1):
                row = {
                    '序号': idx,
                    '图片名称': r['image_name'],
                    '真实类别': r['true_class'],
                    '预测类别': r['pred_class'],
                    '是否正确': '✓' if r['correct'] else '✗',
                    'Accuracy': r['accuracy'],
                    '置信度': round(r['confidence'], 4)
                }
                for i, c in enumerate(class_names):
                    row[f'P({c})'] = round(r['probabilities'][i], 4)
                rows.append(row)

            df_cls = pd.DataFrame(rows)

            # Sheet 名不能超过 31 字符，且不能含特殊字符
            sheet_name = cls_name.replace(" ", "_")[:31]

            # 写入类别明细
            df_cls.to_excel(writer, sheet_name=sheet_name, index=False, startrow=0)

            # 在明细后面追加汇总信息
            worksheet = writer.sheets[sheet_name]
            start_row = len(df_cls) + 2
            worksheet.cell(row=start_row, column=1, value=f"{cls_name} 类别汇总")
            worksheet.cell(row=start_row + 1, column=1, value="图片总数")
            worksheet.cell(row=start_row + 1, column=2, value=total)
            worksheet.cell(row=start_row + 2, column=1, value="预测正确")
            worksheet.cell(row=start_row + 2, column=2, value=correct_count)
            worksheet.cell(row=start_row + 3, column=1, value="预测错误")
            worksheet.cell(row=start_row + 3, column=2, value=total - correct_count)
            worksheet.cell(row=start_row + 4, column=1, value="该类Accuracy")
            worksheet.cell(row=start_row + 4, column=2, value=round(cls_acc, 4))

            print(f"  ✅ Sheet [{sheet_name}] 已写入 ({total} 张图片)")

    print(f"\n✅ Excel 已保存: {excel_path}")
    return excel_path


def save_confusion_matrix(cm, class_names, save_path):
    plt.figure(figsize=(8, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"✅ 混淆矩阵已保存: {save_path}")


def save_per_image_results_by_class(results, output_dir, class_names):
    """按类别分文件夹保存（CSV + 复制图片）"""
    class_results = {name: [] for name in class_names}
    for r in results['per_image_results']:
        class_results[r['true_class']].append(r)

    for cls_name in class_names:
        items = class_results[cls_name]
        safe_name = cls_name.replace(" ", "_")
        cls_dir = os.path.join(output_dir, safe_name)
        os.makedirs(cls_dir, exist_ok=True)

        # CSV
        csv_path = os.path.join(cls_dir, f'{safe_name}_predictions.csv')
        with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            writer.writerow(['序号', '图片名称', '真实类别', '预测类别', '是否正确', 'Accuracy', '置信度'])
            for idx, r in enumerate(items, 1):
                writer.writerow([
                    idx, r['image_name'], r['true_class'], r['pred_class'],
                    '✓' if r['correct'] else '✗', r['accuracy'], round(r['confidence'], 4)
                ])

        # 复制图片
        correct_dir = os.path.join(cls_dir, 'correct')
        wrong_dir = os.path.join(cls_dir, 'wrong')
        os.makedirs(correct_dir, exist_ok=True)
        os.makedirs(wrong_dir, exist_ok=True)
        for r in items:
            dst_dir = correct_dir if r['correct'] else wrong_dir
            dst = os.path.join(dst_dir, r['image_name'])
            try:
                if not os.path.exists(dst):
                    shutil.copy2(r['image_path'], dst)
            except Exception:
                pass

        print(f"  ✅ {cls_name}: {len(items)} 张图片 -> {cls_dir}")


def main():
    # ========== 配置参数 ==========
    WEIGHT_PATH = r'D:\ConvNext\ConvNeXt-V2-main\run\MLCA+MDConv5.5\checkpoint-best.pth'
    TEST_DATA_PATH = r'D:\数据集\大豆数据集\大豆划分\test'
    OUTPUT_DIR = r'D:\预测结果\大豆\Acc\MLCA+MDConv5.5'
    NUM_CLASSES = 4
    BATCH_SIZE = 16
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    # ==================================

    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    # ⭐⭐⭐ 关键修改：类别顺序必须与文件夹字母顺序一致 ⭐⭐⭐
    # 文件夹顺序（按字母）：grey spot → healthy → macular → mosaic
    class_names = ["grey spot", "healthy", "macular", "mosaic"]

    print("=" * 70)
    print("模型测试脚本".center(70))
    print("=" * 70)
    print(f"设备: {DEVICE}")
    print(f"测试数据路径: {TEST_DATA_PATH}")
    print(f"权重路径: {WEIGHT_PATH}")
    print(f"输出目录: {OUTPUT_DIR}")
    print(f"类别顺序: {class_names}")
    print("=" * 70 + "\n")

    # 1. 加载模型
    model = load_model(WEIGHT_PATH, num_classes=NUM_CLASSES, device=DEVICE)

    # 2. 加载测试数据（强制训练时的类别顺序）
    transform = get_test_transform()
    dataset = ImageFolderDataset(TEST_DATA_PATH, transform=transform, class_names=class_names)
    test_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=0, pin_memory=True)

    print(f"📊 测试集样本数: {len(dataset)}")
    print(f"📊 类别映射: {dataset.class_to_idx}\n")

    # 3. 评估
    results = evaluate_test_set(model, test_loader, DEVICE, class_names)

    # 4. 控制台输出
    print("\n" + "=" * 70)
    print("评估结果".center(70))
    print("=" * 70)
    print(f"🎯 总体准确率 (Accuracy): {results['accuracy']:.4f} ({results['accuracy'] * 100:.2f}%)")
    print(f"🎯 宏观平均精确率 (Precision): {results['precision']:.4f}")
    print(f"🎯 宏观平均召回率 (Recall): {results['recall']:.4f}")
    print(f"🎯 宏观平均F1-score: {results['f1']:.4f}")
    print("-" * 70)
    print("\n📊 各类别详细指标:")
    print(f"{'类别':<12} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} "
          f"{'Specificity':<14} {'F1-score':<12} {'Support':<8}")
    print("-" * 80)
    for m in results['per_class_metrics']:
        print(f"{m['class']:<12} {m['Accuracy']:<12.4f} {m['Precision']:<12.4f} "
              f"{m['Recall']:<12.4f} {m['Specificity']:<14.4f} "
              f"{m['F1-score']:<12.4f} {m['Support']:<8}")

    # 5. 保存
    print("\n" + "=" * 70)
    print("保存结果".center(70))
    print("=" * 70)

    # 5.1 混淆矩阵
    save_confusion_matrix(
        results['confusion_matrix'], class_names,
        os.path.join(OUTPUT_DIR, 'test_confusion_matrix.png')
    )

    # 5.2 Excel 输出（包含每张图片的 Accuracy 和 置信度）
    export_to_excel(results, OUTPUT_DIR, class_names)

    # 5.3 按类别分文件夹
    save_per_image_results_by_class(results, OUTPUT_DIR, class_names)

    print("\n" + "=" * 70)
    print("测试完成！".center(70))
    print(f"📁 Excel 结果: {os.path.join(OUTPUT_DIR, '测试结果.xlsx')}")
    print(f"📁 全部结果: {OUTPUT_DIR}")
    print("=" * 70)


if __name__ == '__main__':
    main()