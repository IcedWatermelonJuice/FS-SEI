## FS-SEI Based on RFD
* The code corresponds to the paper "[Few-Shot Specific Emitter Identification Method Using Rotation Feature Decoupling for Secure 6G](https://ieeexplore.ieee.org/document/10419686)".

```
@INPROCEEDINGS{10419686,
  author={Xu, Lai and Shi, Wenjuan and Fu, Xue and Xu, Hua and Wang, Yu and Adebisi, Bamidele and Gui, Guan},
  booktitle={2023 IEEE 23rd International Conference on Communication Technology (ICCT)}, 
  title={Few-Shot Specific Emitter Identification Method Using Rotation Feature Decoupling for Secure 6G}, 
  year={2023},
  volume={},
  number={},
  pages={490-494},
  keywords={Training;6G mobile communication;Wireless communication;Simulation;Feature extraction;Stability analysis;Task analysis;Specific emitter identification (SEI);few-shot learning;self-supervised learning;rotation prediction},
  doi={10.1109/ICCT59356.2023.10419686}}
```

* The "[main-experiment](./main-experiment)" folder is used for pretext task and downstream task.
* The "[ablation-experiment](./ablation-experiment)" folder is used for verifing the impact of mini-target in pretext task to downstream task.
* The [Enhanced Few-Shot Specific Emitter Identification via Phase Shift Prediction and Decoupling](https://ieeexplore.ieee.org/document/10614374/) and PSPD project is the continuation and improvement of this project.

## Contact
* E-mail: [geminixl615@gmail.com](mailto:geminixl615@gmail.com).

## Requirement
* [Python](https://www.python.org/) 3.8.5  
* [PyTorch](https://github.com/pytorch/pytorch) 1.11.0

## Framework
![Framework of FS-SEI Method Using RFD](./fig/framework.png)

## Dataset
* The dataset we used is from [https://genesys-lab.org/oracle](https://genesys-lab.org/oracle).
* We cut this 16-class Wi-Fi dataset into 2 different datasets. Specifically, we divide the first 10 classes as
an auxiliary dataset for pretext task, and the last 6 classes as a few-shot dataset for downstream task.
* Info of the dataset is displayed in the following table.

| Parameters | Value |
| :---: | :---: |
|Dataset | WiFi |
|Standard | [IEEE 802.11a](https://standards.ieee.org/ieee/802.11a/1165/) |
|Dimension | (2, 6000) |
|Format | I/Q |
|Center frequency | 2.45 GHz |
|Sample rate | 5 Mbps |
|Transmitter | 16 USRP X310 |
|Receiver | 1 USRP B210 |
|Distance between transmitters | 6 in feet |
|Distance between transmitter and receiver | 62 in feet |

## Performance
<!-- * FS-SEI based on RFD  
![FS-SEI based on RFD](./fig/unfreezed_pretrained.png)  
* FS-SEI without pretrain  
![FS-SEI without pretrain](./fig/unfreezed_untrained.png)  
* Average Line Chart: FS-SEI based on RFD VS FS-SEI without pretrain  
![Average Line Chart: FS-SEI using RFD VS FS-SEI without pretrain](./fig/average_comparison.png)  
* Abaltion experiment  
![Abaltion experiment](./fig/comparation_of_ablation.png) -->
The performance of our method is demonstrated in `subsection B` and `subsection C` of `section IV` in our paper.
