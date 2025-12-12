%%%%%%%%%%%%%%%%%%%Code specification and Running%%%%%%%%%%%%%%%%%%%%%%%%%%



%%%%%%%%%%%%%%%%%%%Experiment environments%%%%%%%%%%%%%%%%%%%%%%%%%%
1.Compute resource:
-- Ubantu: 22.04.5 LTS
-- CPU: Intel(R) Xeon(R) Gold 6330 CPU @ 2.00GHz
-- GPU: A100 * 8
-- NVIDIA-SMI: 550.127.05
-- CUDA: 12.4

2.Software resouce
-- Platform: PyCharm2019.3
-- Language: Python3.8


3.Package for Tabular MDP
Package         Version
--------------- -----------
addict          2.4.0
cycler          0.11.0
et-xmlfile      1.1.0
kiwisolver      1.3.1
matplotlib      3.3.4
mmcv            1.7.2
numpy           1.19.5
openpyxl        3.1.3
packaging       21.3
pandas          1.1.5
Pillow          8.4.0
pip             21.3.1
pyarrow         6.0.1
pyparsing       3.0.7
python-dateutil 2.9.0.post0
pytz            2024.2
scipy           1.5.4
seaborn         0.8.1
setuptools      39.0.1
six             1.16.0
xlrd            1.2.0
yapf            0.32.0

4.Package for discrete-action DRL
Package                  Version
------------------------ -----------
cloudpickle              1.2.2
cmake                    3.31.6
contourpy                1.1.1
cycler                   0.12.1
filelock                 3.16.1
fonttools                4.56.0
future                   1.0.0
gym                      0.15.3
importlib_resources      6.4.5
Jinja2                   3.1.6
kiwisolver               1.4.7
lit                      18.1.8
MarkupSafe               2.1.5
matplotlib               3.7.1
mpmath                   1.3.0
networkx                 3.1
numpy                    1.19.5
nvidia-cublas-cu11       11.10.3.66
nvidia-cuda-cupti-cu11   11.7.101
nvidia-cuda-nvrtc-cu11   11.7.99
nvidia-cuda-runtime-cu11 11.7.99
nvidia-cudnn-cu11        8.5.0.96
nvidia-cufft-cu11        10.9.0.58
nvidia-curand-cu11       10.2.10.91
nvidia-cusolver-cu11     11.4.0.1
nvidia-cusparse-cu11     11.7.4.91
nvidia-nccl-cu11         2.14.3
nvidia-nvtx-cu11         11.7.91
opencv-python            4.11.0.86
packaging                24.2
pandas                   1.1.3
pillow                   10.4.0
pip                      24.2
protobuf                 5.29.4
psutil                   7.0.0
pyarrow                  12.0.0
pygame                   2.4.0
pyglet                   1.3.2
pyparsing                3.1.4
python-dateutil          2.9.0.post0
pytz                     2025.1
scipy                    1.10.1
seaborn                  0.11.0
setuptools               75.1.0
six                      1.17.0
sympy                    1.13.3
tensorboardX             2.6.2.2
torch                    2.0.1
tqdm                     4.63.0
triton                   2.0.0
typing_extensions        4.12.2
tzdata                   2025.2
wheel                    0.44.0
zipp                     3.20.2

5.Package for continuous-action DRL
Package                  Version
------------------------ ----------
cffi                     1.17.1
cloudpickle              1.3.0
Cython                   0.29.37
diffalgos                0.0.1
fasteners                0.19
filelock                 3.16.1
fsspec                   2024.12.0
future                   1.0.0
glfw                     2.8.0
gym                      0.17.2
imageio                  2.35.1
Jinja2                   3.1.5
joblib                   1.4.2
MarkupSafe               2.1.5
mpmath                   1.3.0
mujoco-py                2.1.2.14
networkx                 3.1
numpy                    1.24.4
nvidia-cublas-cu12       12.1.3.1
nvidia-cuda-cupti-cu12   12.1.105
nvidia-cuda-nvrtc-cu12   12.1.105
nvidia-cuda-runtime-cu12 12.1.105
nvidia-cudnn-cu12        9.1.0.70
nvidia-cufft-cu12        11.0.2.54
nvidia-curand-cu12       10.3.2.106
nvidia-cusolver-cu12     11.4.5.107
nvidia-cusparse-cu12     12.1.0.106
nvidia-ml-py             12.570.86
nvidia-nccl-cu12         2.20.5
nvidia-nvjitlink-cu12    12.6.85
nvidia-nvtx-cu12         12.1.105
nvitop                   1.5.0
pillow                   10.4.0
pip                      24.2
psutil                   7.0.0
pycparser                2.22
pyglet                   1.5.0
redq                     0.0.1
scipy                    1.10.1
setuptools               75.1.0
sympy                    1.13.3
torch                    2.4.1
triton                   3.0.0
typing_extensions        4.12.2
wheel                    0.44.0






%%%%%%%%%%%%%%%%%%%Code and Running%%%%%%%%%%%%%%%%%%%%%%%%%%
The folder <AdaOrder-Q-learning> includes four sub-folders,
TableExperiment, DeepExperiment, and ContinualExperiments.

1.TableExperiment
TableExperiment is used to show performance of our OrderQ and AdaorderQ in Tabular MDP settings.

We run the following bash:

python XX/XX/XXMain.py

can get the result.

Then, we run:

python Figure/XXTableQ.py

can get the TableQ.pdf and SuppleTableQ.pdf


3.DeepExperiment
DeepExperiment is used to show performance of our OrderDQN and AdaOrderDQN in discrete-action DRL settings.

We run the following bash:

python Main.py

can get the result.

Then, we run:

python figure/XXDeepQ.py

can get the DeepQ.pdf and SuppleDeepQ.pdf


4.ContinualExperiments
ContinualExperiments is used to show performance of our OrderDDPG and AdaOrderDDPG in continuous-action DRL settings.

We run the following bash:

python experiments/train_XX.py

can get the result.

Then, we run:

python figure/XXDeepC.py

can get the DeepC.pdf and SuppleDeepC.pdf
