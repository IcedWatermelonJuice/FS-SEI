## Main Experiment 
* This folder is used as "main experiment" in our paper.
* The "[pretext](./pretext)" folder is pretext task which is used to pretrain encoder.
* The "[downstream](./downstream)" folder is downstream task which is used for fine-tuning and testing.

## Pretext Task
* If you need to perform pretext task, switch the directory to "[pretext](./pretext)" folder.
* Before pretraining encoder, configure the experiment parameters in "[config/Rot_Predict_CVCNN.py](./pretext/config/Rot_Predict_CVCNN.py)".
* The "[main.py](./pretext/main.py)" is used to pretrain encoder. You can execute it by using `python main.py`.
* The results of the experiment are stored in "[_experiments](./pretext/_experiments)" folder.

## Downstream Task
* If you need to perform pretext task, switch the directory to "[downstream](./downstream)" folder.
* Before pretraining encoder, configure the experiment parameters in "[config/config.yaml](./downstream/config/config.yaml)"
* The "[train.py](./downstream/train.py)" is used for fine-tuning and testing. For examples, you can execute `python train.py -shot=5` to perform a 5-shot FS-SEI fine-tuning and testing task or execute `python train.py -shot=5 pretrain=0` to perform a 5-shot FS-SEI task without pretraining.
* The experiment logs are stored in "[logs](./downstream/logs)" folder, the fine-tuning files are backed up in "[runs](./downstream/runs)" folder, and the test results are stored in "[test_result](./downstream/test_result)" folder as excel files.
* The "[average.py](./downstream/average.py)" and "[boxplot.py](./downstream/boxplot.py)" in "[visualization](./downstream/visualization)" folder is used to draw box plot and line chart of average accuracy.

## ContactL
* E-mail: [1023010411@njupt.edu.cn](mailto:1023010411@njupt.edu.cn)
