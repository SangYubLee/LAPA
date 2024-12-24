import wandb
import pandas as pd
import matplotlib.pyplot as plt
import os
def load_wandb_logs(run_path):
    """
    wandb 로그를 불러와서 데이터프레임으로 변환하는 함수
    
    Args:
        run_path (str): wandb run 경로 (예: "username/project/run_id")
        
    Returns:
        pd.DataFrame: wandb 로그 데이터
    """
    api = wandb.Api()
    run = api.run(run_path)
    history_df = pd.DataFrame(run.scan_history())
    return history_df

def plot_wandb_metrics(df, metrics, title=None, figsize=(12,6)):
    """
    wandb 메트릭을 그래프로 시각화하는 함수
    
    Args:
        df (pd.DataFrame): wandb 로그 데이터프레임
        metrics (list): 그래프로 그릴 메트릭 이름 리스트
        title (str, optional): 그래프 제목
        figsize (tuple, optional): 그래프 크기
    """
    plt.figure(figsize=figsize)
    
    for metric in metrics:
        if metric in df.columns:
            plt.plot(df['_step'], df[metric], label=metric)
    
    plt.xlabel('Step')
    plt.ylabel('Value') 
    if title:
        plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

    os.makedirs('wandb', exist_ok=True)
    if title:
        plt.savefig(os.path.join('wandb/', f'{title}.png'), dpi=300, bbox_inches='tight')
    else:
        plt.savefig(os.path.join('wandb/', 'metrics.png'), dpi=300, bbox_inches='tight')

if __name__ == "__main__":
    # 사용 예시
    run_path = "pissaitworks-seoul-national-university/phenaki_cnn/kchom5vu"
    
    
    # 로그 데이터 불러오기
    logs_df = load_wandb_logs(run_path)
    
    # 메트릭 시각화
    metrics_to_plot = ['num_unique_indices']  # 시각화할 메트릭 지정
    plot_wandb_metrics(logs_df, metrics_to_plot, title='num_unique_indices')