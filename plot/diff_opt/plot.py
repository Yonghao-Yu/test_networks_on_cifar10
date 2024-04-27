import os
import json
import matplotlib.pyplot as plt

model = ['mlp', 'vgg19', 'resnet34']
opt = ['sgd', 'adamw']
lr = ['0.005', '0.0005', '5e-05']
log_file = 'log.txt'

for m in model:
    for l in lr:
        log_files = [
            os.path.join(m, opt[0], l, log_file),
            os.path.join(m, opt[1], l, log_file)
        ]

        test_acc1_data = [[], []]

        # 加载并解析每个日志文件
        for i, file_name in enumerate(log_files):
            with open(file_name, 'r') as file:
                for line in file:  # 逐行读取
                    if line.strip():  # 跳过空行
                        log = json.loads(line)  # 解析每一行
                        test_acc1_data[i].append(log['test_acc1'])  # 提取test_acc1数据

        # 绘制每个日志文件的test_acc1数据
        plt.figure(figsize=(10, 6))
        for lbl, data in zip(opt, test_acc1_data):
            plt.plot(range(1, len(data) + 1), data, label=f'model={lbl}')

        # 添加图例和标题
        title = f'{m}_{l}'
        plt.title(title)
        plt.xlabel('Epoch')
        plt.ylabel('test_acc1')
        plt.legend()

        plt.savefig(os.path.join('plot', 'diff_opt', title + '.png'), dpi=300, bbox_inches='tight')
        # plt.show()
