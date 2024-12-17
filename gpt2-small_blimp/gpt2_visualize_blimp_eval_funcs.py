import pandas as pd
import matplotlib.pyplot as plt

""" 各モデルのaccuracyを単純に比較 """
def acc_comparison(data: pd.DataFrame, model_name: str, file_name: str) -> None:
    # モデルごとにフィルタリングして可視化
    models = data['Model'].unique()
    fig, ax = plt.subplots(figsize=(15, 10))

    for model in models:
        subset = data[data['Model'] == model]
        ax.barh(subset['Task'], subset['Accuracy'], label=model, alpha=0.7)

    # グラフの装飾
    ax.set_xlabel('Accuracy')
    ax.set_title('Task-wise Accuracy Comparison across Models')
    ax.legend(title='Model')

    # 画像ファイルとして保存
    plt.savefig(f'images/{model_name}/accuracy_comparison_{file_name}.png', bbox_inches='tight')


""" 各モデルごと """
def multiple_models_acc_comparison(data: pd.DataFrame, model_name: str, file_name: str) -> None:
    models = data['Model'].unique()
    # サブプロットの作成
    num_models = len(models)
    fig, axes = plt.subplots(num_models, 1, figsize=(15, 5 * num_models), sharex=True)

    for ax, model in zip(axes, models):
        subset = data[data['Model'] == model]
        ax.barh(subset['Task'], subset['Accuracy'], alpha=0.7)
        ax.set_title(model)
        ax.set_xlabel('Accuracy')

    # グラフの装飾
    plt.suptitle('Task-wise Accuracy Comparison across Models', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # サブプロットのタイトルのスペースを調整

    # 画像ファイルとして保存
    plt.savefig(f'images/{model_name}/multiple_models_accuracy_comparison_{file_name}.png', bbox_inches='tight')


""" 一番性能の良かったモデルのみを表示 """
def models_above_base_model(data: pd.DataFrame, model_name: str, file_name: str) -> None:
    # GPT-2の精度を取得
    baseline_accuracy = data.loc[data['Model'] == f'{model_name}', 'Accuracy'].values[0]
    print(baseline_accuracy)

    # GPT-2を上回るモデルがあるタスクを特定
    tasks_with_above_baseline = data[data['Accuracy'] > baseline_accuracy]['Task'].unique()

    # 可視化するためのデータをフィルタリング
    filtered_data = data[data['Task'].isin(tasks_with_above_baseline) & (data['Model'] != 'gpt2')]

    # モデルごとにフィルタリングして可視化
    models = filtered_data['Model'].unique()
    fig, ax = plt.subplots(figsize=(15, 10))

    for model in models:
        subset = filtered_data[filtered_data['Model'] == model]
        ax.barh(subset['Task'], subset['Accuracy'], label=model, alpha=0.7)

    # 各モデルがGPT-2を上回ったタスクの数をカウント
    task_count = len(tasks_with_above_baseline)
    above_baseline_counts = {model: 0 for model in models}

    # GPT-2のカウントを追加
    gpt2_count = 0
    for task in tasks_with_above_baseline:
        if any(filtered_data[(filtered_data['Task'] == task) & (filtered_data['Model'] != 'gpt2')]['Accuracy'] > baseline_accuracy):
            best_model = filtered_data[filtered_data['Task'] == task].sort_values(by='Accuracy', ascending=False).iloc[0]
            above_baseline_counts[best_model['Model']] += 1
        if baseline_accuracy > filtered_data[filtered_data['Task'] == task]['Accuracy'].max():
            gpt2_count += 1

    # 各モデルのタスク数を割合として計算
    above_baseline_ratios = {model: (count / task_count * 100) for model, count in above_baseline_counts.items()}

    # GPT-2の割合も追加
    above_baseline_ratios['gpt2'] = (gpt2_count / task_count * 100)

    # グラフの装飾
    ax.set_xlabel('Accuracy', fontsize=12)
    ax.set_title('Task-wise Accuracy Comparison (Models Outperforming GPT-2)', fontsize=16)
    ax.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')

    # モデルがGPT-2を上回る性能を発揮したタスクの割合をグラフの下に表示
    plt.figtext(0.5, 0.01,
                '\n'.join([f'{model}: {above_baseline_ratios[model]:.2f}% of tasks' for model in above_baseline_ratios]),
                ha='center', fontsize=10)

    # 画像ファイルとして保存
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])  # 下部のマージンを調整
    plt.savefig(f'images/{model_name}/models_above_gpt2_task_comparison_with_task_ratios_{file_name}.png', bbox_inches='tight')
