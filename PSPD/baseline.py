import os
from utils.config import finetune_config
from finetune import downstream

if __name__ == "__main__":
    # os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    # for s in [1, 5, 10, 15, 20, 25, 30]:
    # for s in [1, 5, 30]:
    #     ft_config = finetune_config(shot=s, max_epoch=300, pretrain_epoch=0, input_type='cwd', dataset_name="radar_10db")
    #     downstream(ft_config)
    # ft_config = finetune_config(shot=10, encoder_name="DenseNet")
    # downstream(ft_config)
    from copy import deepcopy

    def_bs_config = finetune_config(dataset_name="wifi_ft50", max_epoch=300, pretrain_epoch=0, shot=20)
    shots = [def_bs_config['dataset']['shot']] if isinstance(def_bs_config['dataset']['shot'], int) else def_bs_config['dataset']['shot']
    for s in shots:
        bs_config = deepcopy(def_bs_config)
        bs_config['dataset']['shot'] = s
        downstream(bs_config)
