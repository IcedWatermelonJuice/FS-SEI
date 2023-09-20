## Ablation Experiment
* This folder is used as "ablation experiment" in our paper.
* The "[pretext](./pretext)" folder is used to pretrain encoder without specified mini-task in pretext task.
* The "[downstream](./downstream)" folder is downstream task which is used for fine-tuning and testing.

## Pretext Task
* If you need to perform pretext task, switch the directory to "[pretext](./pretext)" folder.
* Before pretraining encoder, configure the experiment parameters in "[config/Ablate_CLS.py](./pretext/config/Ablate_CLS.py)", "[config/Ablate_MSE.py](./pretext/config/Ablate_MSE.py)" and "[config/Ablate_NCE.py](./pretext/config/Ablate_NCE.py)".
* The "[main.py](./pretext/main.py)" is used to pretrain encoder. For a example, you can execute `python main.py -exp=Ablate_CLS` to perform a pretraining task for ablating CLS experiment. Similarly, `python main.py -exp=Ablate_MSE` and `python main.py -exp=Ablate_NCE` are used for ablating MSE and NCE experiments.
* The results of the experiment are stored in "[_experiments](./pretext/_experiments)" folder.

## Downstream Task
* The downstream task here is similar to the one in main experiment.
* For more Info., please go to the "[main-experiment](../main-experiment)" folder.

## ContactL
* E-mail: [1023010411@njupt.edu.cn](mailto:1023010411@njupt.edu.cn)
