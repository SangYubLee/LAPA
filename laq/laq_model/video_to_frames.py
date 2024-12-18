import os
import subprocess
from pathlib import Path
from tqdm import tqdm

def extract_frames(video_dir, output_dir, fps=12):
    """
    비디오 파일들을 프레임 이미지로 변환
    Args:
        video_dir (str): webm 비디오 파일들이 있는 디렉토리 경로
        output_dir (str): 프레임들을 저장할 새로운 디렉토리 경로
        fps (int): 추출할 프레임 레이트 (기본값: 12fps)
    """
    # 비디오 파일 리스트 가져오기
    video_files = list(Path(video_dir).glob('*.webm'))
    
    # 출력 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)
    
    for video_path in tqdm(video_files, desc="비디오 프레임 추출 중"):
        # 비디오 파일명 추출 (확장자 제외)
        video_name = video_path.stem
        
        # 새로운 위치에 프레임 저장 디렉토리 생성
        frame_dir = Path(output_dir) / video_name
        os.makedirs(frame_dir, exist_ok=True)
        
        # ffmpeg 명령어 실행
        output_pattern = str(frame_dir / f"frame_%06d.jpg")
        cmd = [
            'ffmpeg', '-i', str(video_path),
            '-r', str(fps),
            '-q:v', '1',
            output_pattern
        ]
        
        try:
            subprocess.run(cmd, check=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            print(f"Error processing {video_path}: {e}")

if __name__ == "__main__":
    VIDEO_DIR = "/media/hdd/sangyub/data/sthv2/20bn-something-something-v2_video&fps30"
    OUTPUT_DIR = "/media/hdd/sangyub/data/sthv2/20bn-something-something-v2"
    extract_frames(VIDEO_DIR, OUTPUT_DIR, fps=12)