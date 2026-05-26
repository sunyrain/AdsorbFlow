#!/usr/bin/env python3
"""
提取训练日志中的 loss 曲线和 pos_mae 指标，保存为 CSV 并绘制图表。
用法:
    python extract_training_metrics.py <log_file_path> [--output_dir OUTPUT_DIR]

示例:
    python extract_training_metrics.py logs/wandb/.../output.log
    python extract_training_metrics.py logs/wandb/.../output.log --output_dir ./results
"""

import argparse
import re
import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端


def parse_training_log(log_path: str) -> tuple:
    """
    解析训练日志文件，提取训练loss和验证指标。

    Args:
        log_path: 日志文件路径

    Returns:
        train_data: 训练数据列表
        val_data: 验证数据列表
    """
    # 训练日志格式: loss: 6.10e+00, loss_unweighted: 6.10e+00, lr: 2.00e-04, epoch: 7.01e-04, step: 1.00e+00
    train_pattern = re.compile(
        r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}).*'
        r'loss:\s*([\d.e+-]+),\s*'
        r'loss_unweighted:\s*([\d.e+-]+),\s*'
        r'lr:\s*([\d.e+-]+),\s*'
        r'epoch:\s*([\d.e+-]+),\s*'
        r'step:\s*([\d.e+-]+)'
    )

    # 验证日志格式: loss: 5.4672, loss_unweighted: 5.4672, pos_mae: 3.4656, epoch: 1.0000
    val_pattern = re.compile(
        r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}).*'
        r'loss:\s*([\d.e+-]+),\s*'
        r'loss_unweighted:\s*([\d.e+-]+),\s*'
        r'pos_mae:\s*([\d.e+-]+),\s*'
        r'epoch:\s*([\d.e+-]+)'
    )

    train_data = []
    val_data = []

    with open(log_path, 'r', encoding='utf-8') as f:
        for line in f:
            # 尝试匹配验证数据（含pos_mae）
            val_match = val_pattern.search(line)
            if val_match:
                timestamp, loss, loss_unweighted, pos_mae, epoch = val_match.groups()
                val_data.append({
                    'timestamp': timestamp,
                    'epoch': float(epoch),
                    'loss': float(loss),
                    'loss_unweighted': float(loss_unweighted),
                    'pos_mae': float(pos_mae)
                })
                continue

            # 尝试匹配训练数据
            train_match = train_pattern.search(line)
            if train_match:
                timestamp, loss, loss_unweighted, lr, epoch, step = train_match.groups()
                train_data.append({
                    'timestamp': timestamp,
                    'step': int(float(step)),
                    'epoch': float(epoch),
                    'loss': float(loss),
                    'loss_unweighted': float(loss_unweighted),
                    'lr': float(lr)
                })

    return train_data, val_data


def save_to_csv(train_data: list, val_data: list, output_dir: str, prefix: str = ""):
    """保存数据到CSV文件"""
    os.makedirs(output_dir, exist_ok=True)

    train_csv_path = os.path.join(output_dir, f"{prefix}train_metrics.csv")
    val_csv_path = os.path.join(output_dir, f"{prefix}val_metrics.csv")

    if train_data:
        train_df = pd.DataFrame(train_data)
        train_df.to_csv(train_csv_path, index=False)
        print(f"训练数据已保存到: {train_csv_path}")
        print(f"  - 共 {len(train_data)} 条记录")

    if val_data:
        val_df = pd.DataFrame(val_data)
        val_df.to_csv(val_csv_path, index=False)
        print(f"验证数据已保存到: {val_csv_path}")
        print(f"  - 共 {len(val_data)} 条记录")

    return train_csv_path if train_data else None, val_csv_path if val_data else None


def plot_metrics(train_data: list, val_data: list, output_dir: str, prefix: str = ""):
    """绘制训练曲线图"""
    os.makedirs(output_dir, exist_ok=True)

    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. 训练 Loss vs Step
    if train_data:
        train_df = pd.DataFrame(train_data)
        ax1 = axes[0, 0]
        ax1.plot(train_df['step'], train_df['loss'], 'b-', linewidth=0.8, alpha=0.7)
        ax1.set_xlabel('Step')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training Loss vs Step')
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')

    # 2. 训练 Loss vs Epoch
    if train_data:
        ax2 = axes[0, 1]
        ax2.plot(train_df['epoch'], train_df['loss'], 'b-', linewidth=0.8, alpha=0.7)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.set_title('Training Loss vs Epoch')
        ax2.grid(True, alpha=0.3)
        ax2.set_yscale('log')

    # 3. 验证 Loss vs Epoch
    if val_data:
        val_df = pd.DataFrame(val_data)
        ax3 = axes[1, 0]
        ax3.plot(val_df['epoch'], val_df['loss'], 'r-o', linewidth=1.5, markersize=4)
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Validation Loss')
        ax3.set_title('Validation Loss vs Epoch')
        ax3.grid(True, alpha=0.3)

    # 4. Pos MAE vs Epoch
    if val_data:
        ax4 = axes[1, 1]
        ax4.plot(val_df['epoch'], val_df['pos_mae'], 'g-o', linewidth=1.5, markersize=4)
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Position MAE')
        ax4.set_title('Position MAE vs Epoch')
        ax4.grid(True, alpha=0.3)

        # 标注最小值
        min_idx = val_df['pos_mae'].idxmin()
        min_epoch = val_df.loc[min_idx, 'epoch']
        min_mae = val_df.loc[min_idx, 'pos_mae']
        ax4.annotate(f'Min: {min_mae:.4f}\n@ Epoch {min_epoch:.1f}',
                    xy=(min_epoch, min_mae),
                    xytext=(min_epoch + 1, min_mae + 0.1),
                    arrowprops=dict(arrowstyle='->', color='red'),
                    fontsize=9, color='red')

    plt.tight_layout()

    # 保存图片
    plot_path = os.path.join(output_dir, f"{prefix}training_curves.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"训练曲线图已保存到: {plot_path}")

    plt.close()

    # 额外：绘制综合对比图
    if train_data and val_data:
        fig2, ax = plt.subplots(figsize=(12, 6))

        train_df = pd.DataFrame(train_data)
        val_df = pd.DataFrame(val_data)

        # 绘制训练loss (按epoch聚合)
        train_by_epoch = train_df.groupby(train_df['epoch'].astype(int))['loss'].mean()
        ax.plot(train_by_epoch.index, train_by_epoch.values, 'b-',
                linewidth=1.5, label='Train Loss', alpha=0.8)

        # 绘制验证loss
        ax.plot(val_df['epoch'], val_df['loss'], 'r-o',
                linewidth=1.5, markersize=5, label='Val Loss')

        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.set_title('Training vs Validation Loss', fontsize=14)
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

        combined_path = os.path.join(output_dir, f"{prefix}loss_comparison.png")
        plt.savefig(combined_path, dpi=150, bbox_inches='tight')
        print(f"Loss对比图已保存到: {combined_path}")
        plt.close()

    return plot_path


def print_summary(train_data: list, val_data: list):
    """打印训练摘要"""
    print("\n" + "="*60)
    print("训练摘要")
    print("="*60)

    if train_data:
        train_df = pd.DataFrame(train_data)
        print(f"\n训练数据:")
        print(f"  总步数: {train_df['step'].max()}")
        print(f"  总Epoch: {train_df['epoch'].max():.2f}")
        print(f"  初始Loss: {train_df['loss'].iloc[0]:.4f}")
        print(f"  最终Loss: {train_df['loss'].iloc[-1]:.4f}")
        print(f"  最小Loss: {train_df['loss'].min():.4f}")

    if val_data:
        val_df = pd.DataFrame(val_data)
        print(f"\n验证数据:")
        print(f"  验证次数: {len(val_data)}")
        print(f"  初始Val Loss: {val_df['loss'].iloc[0]:.4f}")
        print(f"  最终Val Loss: {val_df['loss'].iloc[-1]:.4f}")
        print(f"  最小Val Loss: {val_df['loss'].min():.4f} (Epoch {val_df.loc[val_df['loss'].idxmin(), 'epoch']:.1f})")
        print(f"  初始Pos MAE: {val_df['pos_mae'].iloc[0]:.4f}")
        print(f"  最终Pos MAE: {val_df['pos_mae'].iloc[-1]:.4f}")
        print(f"  最小Pos MAE: {val_df['pos_mae'].min():.4f} (Epoch {val_df.loc[val_df['pos_mae'].idxmin(), 'epoch']:.1f})")

    print("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(description='提取训练日志中的loss和pos_mae指标')
    parser.add_argument('log_path', type=str, help='日志文件路径')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='输出目录 (默认: 日志所在目录)')
    parser.add_argument('--prefix', type=str, default='',
                       help='输出文件名前缀')

    args = parser.parse_args()

    log_path = args.log_path
    if not os.path.exists(log_path):
        print(f"错误: 日志文件不存在: {log_path}")
        return

    # 设置输出目录
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = os.path.dirname(log_path)
        if not output_dir:
            output_dir = '.'

    print(f"解析日志文件: {log_path}")
    print(f"输出目录: {output_dir}")

    # 解析日志
    train_data, val_data = parse_training_log(log_path)

    if not train_data and not val_data:
        print("未找到任何训练或验证数据!")
        return

    # 保存CSV
    save_to_csv(train_data, val_data, output_dir, args.prefix)

    # 绘制图表
    plot_metrics(train_data, val_data, output_dir, args.prefix)

    # 打印摘要
    print_summary(train_data, val_data)


if __name__ == '__main__':
    main()
