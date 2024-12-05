import os
import subprocess

def run_script(script_path):
    try:
        subprocess.run(['python', script_path], check=True)
        print(f"Successfully ran {script_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error running {script_path}: {e}")
        exit(1)

def main():
    run_script(os.path.join('scripts', 'download_stock_data.py'))
    run_script(os.path.join('scripts', 'preprocess_stock_data.py'))
    run_script(os.path.join('training', 'train_neural_cde.py'))
    run_script(os.path.join('training', 'train_linear_regression.py'))
    run_script(os.path.join('training', 'train_lstm.py'))
    
    run_script(os.path.join('evaluation', 'evaluate_models.py'))

if __name__ == '__main__':
    main()
