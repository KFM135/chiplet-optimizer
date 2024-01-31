import os
import subprocess
import sys
import datetime

start_time = datetime.datetime.now()
child_script_dir = os.path.dirname(os.path.abspath(__file__))
child_script_path = os.path.join(child_script_dir, 'RL_64_chiplet.py')
output_dir = 'RL_64_chiplet'
# episode_len = [2, 5, 10]
# en_coeff = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

isExist = os.path.exists(output_dir)
if not isExist:
    os.makedirs(output_dir)

for i in range(0, 10):
    print(f'Running script {i} of 10')
    out_file_name = 'RL_3D_' + str(i) + '.txt'
    out_file_path = os.path.join(output_dir, out_file_name)
    command = [sys.executable, child_script_path, f'1>{out_file_path}', str(i)]
    subprocess.run(command, shell=True, capture_output=True, text=True)

end_time = datetime.datetime.now()
print(f'Run time:{end_time - start_time}')