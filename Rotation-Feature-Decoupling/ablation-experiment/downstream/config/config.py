import argparse
import yaml


def get_config(config_root):
    parser = argparse.ArgumentParser()
    parser.add_argument('--encoder', type=str, default="CVCNN", help='the name of encoder')
    parser.add_argument('--freeze', type=int, default=0, help='freeze the encoder (1: freeze, 0: not freeze)')
    parser.add_argument('--shot', type=int, default=10, help='the length of a shot from dataset')
    parser.add_argument('--iteration', type=int, default=100, help='the number of monte carlo experiments')
    parser.add_argument('--pretrain', type=int, default=1, help='use pretrained encoder (1: pretrain, 0: not pretrain)')
    parser.add_argument('--pretrain_encoder', type=str, default="feature_net_epoch245", help='the name of pretrain encoder')
    args_opt = parser.parse_args()

    config = yaml.load(open(config_root, "r"), Loader=yaml.FullLoader)
    config["trainer"]["encoder"] = args_opt.encoder
    config["trainer"]["freeze"] = args_opt.freeze == 1
    config["trainer"]["k_shot"] = args_opt.shot
    config["iteration"] = args_opt.iteration
    config["trainer"]["pretrain"] = args_opt.pretrain == 1
    config["trainer"]["pretrained_path"] = f"../pretext/_experiments/Rot_Predict_{args_opt.encoder}/{args_opt.pretrain_encoder}"
    config["full_name"]=f"{args_opt.encoder}_{'freezed' if args_opt.freeze == 1 else 'unfreezed'}_{'pretrained' if args_opt.pretrain == 1 else 'untrained'}_{args_opt.shot}shot_{args_opt.iteration}iteration"


    return config

if __name__ == "__main__":
    get_config("config.yaml")
