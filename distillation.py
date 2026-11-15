import torch
import logging
import numpy as np
from tqdm import tqdm,trange

import multiprocessing
from os.path import join
from datetime import datetime
from torch.utils.data.dataloader import DataLoader


import parser_ImAge

import parser_RelaVPR
import test
from network_ImAge import VPRmodel

from network_RelaVPR import  VPRmodel_small

torch.backends.cudnn.benchmark = True  # Provides a speedup

import util


import commons
import datasets_ws

from loss import loss_function, RKDConfidenceHardLoss_final
from dataloaders.GSVCities import get_GSVCities

import warnings
warnings.filterwarnings("ignore")
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
from multiprocessing import freeze_support
if __name__ == '__main__':
    # 在 Windows 系统下，使用多进程时需要调用 freeze_support()


    freeze_support()
    #### Initial setup: parser, logging...
    args_teacher = parser_ImAge.parse_arguments_win()
    args=parser_RelaVPR.parse_arguments_win()
    start_time = datetime.now()
    args.save_dir = join("logs", args.save_dir, start_time.strftime('%Y-%m-%d_%H-%M-%S'))
    commons.setup_logging(args.save_dir)
    commons.make_deterministic(args.seed)
    logging.info(f"Arguments: {args}")
    logging.info(f"The outputs are being saved in {args.save_dir}")
    logging.info(f"Using {torch.cuda.device_count()} GPUs and {multiprocessing.cpu_count()} CPUs")

    #### Creation of Datasets
    logging.debug(f"Loading dataset {args.eval_dataset_name} from folder {args.eval_datasets_folder}")

    test_ds_pitts30k = datasets_ws.BaseDataset(args, args.eval_datasets_folder, 'pitts30k', "val")
    test_ds_msls = datasets_ws.BaseDataset(args, args.eval_datasets_folder, 'msls_val', "test")
    # args.features_dim = 4096
    #### Initialize model

    model = VPRmodel_small(args )

    model = model.to(args.device)
    distill_head=RKDConfidenceHardLoss_final().to(args.device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    aggregator_params = sum(p.numel() for p in model.aggregator.parameters()) if model.aggregator else 0

    print(f"The entire parameters: {total_params / 1e6:.2f}M")
    print(f"The trainable parameters: {trainable_params / 1e6:.2f}M")
    print(f"The aggregator parameters: {aggregator_params / 1e6:.2f}M")

    args.features_dim = 3072






    if args.teacher == True:
        teacher =VPRmodel(args_teacher).eval().to(args.device)
        if args.resume is not None:
            logging.info(f"Resuming model from {args_teacher.resume}")
            teacher = util.resume_model(args_teacher, teacher)

    #### Setup Optimizer and Loss
    if args.optim == "adam":
        # optimizer = torch.optim.Adam(model.parameters()+distill_head.parameters(), lr=args.lr)
        optimizer = torch.optim.Adam(
            list(model.parameters()) + list(distill_head.parameters()),
            lr=args.lr
        )
    elif args.optim == "sgd":
        optimizer = torch.optim.SGD(list(model.parameters()) + list(distill_head.parameters()), lr=args.lr,
                                    momentum=0.9, weight_decay=0.001)


    best_r1 = start_epoch_num = not_improved_num = 0
    logging.info(f"Output dimension of the model is {args.features_dim}")

    #### Getting GSVCities
    train_dataset = get_GSVCities()

    train_loader_config = {
        'batch_size': args.train_batch_size,
        'num_workers': args.num_workers,
        'drop_last': False,
        'pin_memory': True,
       'shuffle': False}

    #### Training loop
    ds = DataLoader(dataset=train_dataset, **train_loader_config)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=len(ds)*3, gamma=0.5, last_epoch=-1)



    for epoch_num in range(start_epoch_num, args.epochs_num):
        logging.info(f"Start training epoch: {epoch_num:02d}")

        epoch_start_time = datetime.now()
        epoch_losses = np.zeros((0,1), dtype=np.float32)

        model = model.train()
        epoch_losses=[]
        for images, place_id in tqdm(ds):
            BS, N, ch, h, w = images.shape

            # reshape places and labels
            images = images.view(BS*N, ch, h, w)
            #
            labels = place_id.view(-1)

            descriptors,_,_= model(images.to(args.device))

            if args.teacher:
                with torch.no_grad():
                    teacher_descriptors = teacher(images.to(args.device))
                # print(descriptors.shape)
                # print(teacher_descriptors.shape)

                loss =distill_head(descriptors,teacher_descriptors,labels.to(args.device))


            else:
                loss = loss_function (descriptors, labels)



            del descriptors

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # scheduler.step()

            # Keep track of all losses by appending them to epoch_losses
            batch_loss = loss.item()
            epoch_losses = np.append(epoch_losses, batch_loss)
            del loss

        logging.info(f"Finished epoch {epoch_num:02d} in {str(datetime.now() - epoch_start_time)[:-7]}, "
                     f"average epoch triplet loss = {epoch_losses.mean():.4f}")

        # Compute recalls on validation set
        recalls_pitts30k, recalls_str_pitts30k = test.test(args, test_ds_pitts30k, model)
        logging.info(f"Recalls on val set {test_ds_pitts30k}: {recalls_str_pitts30k}")

        recalls_msls, recalls_str_msls =test.test(args, test_ds_msls, model, args.test_method)
        logging.info(f"Recalls on val set {test_ds_msls}: {recalls_str_msls}")


        is_best = recalls_msls[0] + recalls_pitts30k[0] > best_r1

        # Save checkpoint, which contains all training parameters
        util.save_checkpoint(args, {"epoch_num": epoch_num, "model_state_dict": model.state_dict(),
                                    "optimizer_state_dict": optimizer.state_dict(), "recalls": recalls_msls,
                                    "best_r1": best_r1,
                                    "not_improved_num": not_improved_num
                                    }, is_best, filename="last_model.pth")


        # If recall@5 did not improve for "many" epochs, stop training
        if is_best:
            logging.info(
                f"Improved: previous best R@5 = {best_r1:.1f}, current R@5 = {(recalls_msls[0] + recalls_pitts30k[0]):.1f}")
            best_r1 = (recalls_msls[0] + recalls_pitts30k[0])
            not_improved_num = 0
        else:
            not_improved_num += 1
            logging.info(
                f"Not improved: {not_improved_num} / {args.patience}: best R@5 = {best_r1:.1f}, current R@5 = {(recalls_msls[0] + recalls_pitts30k[0]):.1f}")
            if not_improved_num >= args.patience:
                logging.info(f"Performance did not improve for {not_improved_num} epochs. Stop training.")
                break

    logging.info(f"Best R@5: {best_r1:.1f}")
    logging.info(f"Trained for {epoch_num + 1:02d} epochs, in total in {str(datetime.now() - start_time)[:-7]}")

