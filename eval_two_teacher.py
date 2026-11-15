import os
import torch


import network_RelaVPR_two_teacher


import parser_RelaVPR_two_teacher
import parser_RelaVPR
import logging
from os.path import join
from datetime import datetime

import test
import test
import test_two_teacher

import util
import commons
import datasets_ws

import warnings

warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

if __name__ == '__main__':

    ######################################### SETUP #########################################
    places = ['sped','msls_val', 'amstertime','pitts30k', 'nordland','pitts250k']
    args = parser_RelaVPR_two_teacher.parse_arguments_win()
    model = network_RelaVPR_two_teacher.VPRmodel_small(args)

    model = model.to(args.device)
    model.load_state_dict(torch.load('./RelaVPR_two_teacher.pth')['model_state_dict'])
    total_params = sum(p.numel() for p in model.parameters())
    print(f"The entire parameters: {total_params / 1e6:.4f}M")

 
    result_file = "./RelaVPR_two_teacher.txt"

    with open(result_file, "w") as f:
        f.write("RelaVPR_two_teacher Evaluation Results\n")
        f.write("=" * 60 + "\n")

    for place in places:

        start_time = datetime.now()
        args.save_dir = join("test", args.save_dir, start_time.strftime('%Y-%m-%d_%H-%M-%S'))
        commons.setup_logging(args.save_dir)
        commons.make_deterministic(args.seed)

        args.features_dim = 3072
        logging.info(f"Arguments: {args}")
        logging.info(f"The outputs are being saved in {args.save_dir}")

        ######################################### MODEL #########################################


        # args.pca_dim=6144
        pca = None


        ######################################### DATASETS #########################################
        test_ds = datasets_ws.BaseDataset(args, args.eval_datasets_folder, place, "test")
        logging.info(f"Test set: {test_ds}")

        ######################################### TEST on TEST SET #########################################
        recalls, recalls_str = test_two_teacher.test(args, test_ds, model, args.test_method, pca)
        logging.info(f"Recalls on {test_ds}: {recalls_str}")

        # ================= 保存结果 =================
        with open(result_file, "a") as f:
            f.write(f"\nDataset: {place}\n")
            f.write(f"Checkpoint: best_model.pth\n")
            f.write(f"Time: {datetime.now()}\n")
            f.write(f"Results: {recalls_str}\n")
            f.write("-" * 60 + "\n")

        logging.info(f"Finished in {str(datetime.now() - start_time)[:-7]}")
