# goldenfish_detect
detect golden fish
### 簡單來說 best.pt是我訓練出來的模型 然後python檔案會用它來偵測大家丟進去的檔案
### 阿不過要下載torch 以及cv2
### 數據集下載可以用以下的code


!pip install roboflow

from roboflow import Roboflow
rf = Roboflow(api_key="pcRddkCMQeAgwiLZvdIN")
project = rf.workspace("new-workspace-nfkue").project("fish-xgreb")
version = project.version(1)
dataset = version.download("yolov5")

