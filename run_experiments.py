"""
PSESC 项目主运行脚本
在 ECG 心电图数据集上运行分类和异常检测实验
"""

import sys
import os

# 确保导入路径正确
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

print("""
╔══════════════════════════════════════════════════════════════════╗
║                                                                  ║
║       PSESC 时间序列分类与异常检测实验系统                        ║
║       基于 Shapelet 隶属度的紧凑特征提取                         ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
""")

def main():
    """主函数：运行所有实验"""
    
    print("请选择要运行的实验:")
    print("  [1] ECG 心电图分类实验（决策树 + 随机森林）")
    print("  [2] ECG 心电图异常检测实验（Isolation Forest + SVM）")
    print("  [3] 运行所有实验")
    print("  [4] 退出")
    print("  [5] 工业数据异常检测实验模板（SWaT/SMD/MSL）")
    
    choice = input("\n请输入选项 (1-4): ").strip()
    
    if choice == '1':
        print("\n启动分类实验...")
        from experiments.ecg_classification_experiment import run_classification_experiment
        try:
            run_classification_experiment('ECGFiveDays')
            print("\n✓ 分类实验完成！")
        except Exception as e:
            print(f"\n❌ 分类实验失败: {e}")
            import traceback
            traceback.print_exc()
    
    elif choice == '2':
        print("\n启动异常检测实验...")
        from experiments.ecg_anomaly_detection_experiment import run_anomaly_detection_experiment
        try:
            run_anomaly_detection_experiment('ECGFiveDays', normal_label=1)
            print("\n✓ 异常检测实验完成！")
        except Exception as e:
            print(f"\n❌ 异常检测实验失败: {e}")
            import traceback
            traceback.print_exc()
    
    elif choice == '3':
        print("\n启动所有实验...")
        
        # 实验1：分类
        print("\n" + "="*70)
        print("  第 1/2 项：分类实验")
        print("="*70)
        from experiments.ecg_classification_experiment import run_classification_experiment
        try:
            run_classification_experiment('ECGFiveDays')
            print("\n✓ 分类实验完成！")
        except Exception as e:
            print(f"\n❌ 分类实验失败: {e}")
            import traceback
            traceback.print_exc()
        
        # 实验2：异常检测
        print("\n" + "="*70)
        print("  第 2/2 项：异常检测实验")
        print("="*70)
        from experiments.ecg_anomaly_detection_experiment import run_anomaly_detection_experiment
        try:
            run_anomaly_detection_experiment('ECGFiveDays', normal_label=1)
            print("\n✓ 异常检测实验完成！")
        except Exception as e:
            print(f"\n❌ 异常检测实验失败: {e}")
            import traceback
            traceback.print_exc()
        
        print("\n" + "="*70)
        print("  所有实验已完成！")
        print("="*70)
        print("\n结果保存位置:")
        print("  - 分类结果: results/classification/")
        print("  - 异常检测结果: results/anomaly_detection/")
    
    elif choice == '4':
        print("\n再见！")
        return

    elif choice == '5':
        print("\n请在终端使用以下命令运行工业数据模板:")
        print("python experiments/industrial_anomaly_experiment.py --dataset swat --data-dir <your_data_dir>")
        print("python experiments/industrial_anomaly_experiment.py --dataset smd --data-dir <your_data_dir> --smd-per-machine")
        print("python experiments/industrial_anomaly_experiment.py --dataset msl --data-dir <your_data_dir>")
    
    else:
        print("\n无效选项，请重新运行程序。")


if __name__ == '__main__':
    main()
