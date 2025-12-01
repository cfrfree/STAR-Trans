import logging
import os
import time
import torch
import torch.nn as nn
from utils.meter import AverageMeter
from utils.metrics import R1_mAP_eval
from torch.cuda import amp
import torch.distributed as dist
from loss import clip_loss
from loss.cmt_loss import ModalityConsistencyLoss
from loss.topo_loss import TopologicalConsistencyLoss


def do_train_pair(cfg, model, train_loader_pair, optimizer, scheduler, local_rank, start_epoch=1):
    log_period = cfg.SOLVER.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD

    device = "cuda"
    epochs = cfg.SOLVER.MAX_EPOCHS

    logger = logging.getLogger("transreid.train")
    logger.info("start training")
    _LOCAL_PROCESS_GROUP = None

    if device:
        model.to(local_rank)
        if torch.cuda.device_count() > 1 and cfg.MODEL.DIST_TRAIN:
            print("Using {} GPUs for training".format(torch.cuda.device_count()))
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True)

    loss_meter = AverageMeter()
    scaler = amp.GradScaler()

    # train pair
    if cfg.MODEL.PAIR:
        if torch.cuda.device_count() > 1 and cfg.MODEL.DIST_TRAIN:
            model.module.train_with_pair()
        else:
            model.train_with_pair()
        for epoch in range(start_epoch, epochs + 1):
            start_time = time.time()
            loss_meter.reset()
            scheduler.step(epoch)
            model.train()
            for n_iter, (img, vid, target_cam) in enumerate(train_loader_pair):
                optimizer.zero_grad()
                img = img.to(device)
                target = vid.to(device)
                target_cam = target_cam.to(device)
                with amp.autocast(enabled=True):
                    logits_per_sar = model(img, target, cam_label=target_cam)
                    loss = clip_loss(logits_per_sar)

                scaler.scale(loss).backward()

                scaler.step(optimizer)
                scaler.update()

                loss_meter.update(loss.item(), img.shape[0])

                torch.cuda.synchronize()
                if (n_iter + 1) % log_period == 0:
                    logger.info(
                        "Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Base Lr: {:.2e}".format(
                            epoch, (n_iter + 1), len(train_loader_pair), loss_meter.avg, scheduler._get_lr(epoch)[0]
                        )
                    )

            end_time = time.time()
            time_per_batch = (end_time - start_time) / (n_iter + 1)
            if cfg.MODEL.DIST_TRAIN:
                pass
            else:
                logger.info(
                    "Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]".format(
                        epoch, time_per_batch, train_loader_pair.batch_size / time_per_batch
                    )
                )

            if epoch % checkpoint_period == 0:
                if cfg.MODEL.DIST_TRAIN:
                    if dist.get_rank() == 0:
                        torch.save(model.state_dict(), os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + "_{}.pth".format(epoch)))
                else:
                    torch.save(model.state_dict(), os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + "_{}.pth".format(epoch)))


def do_train(cfg, model, center_criterion, train_loader, val_loader, optimizer, optimizer_center, scheduler, loss_fn, num_query, local_rank):
    log_period = cfg.SOLVER.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    eval_period = cfg.SOLVER.EVAL_PERIOD

    device = "cuda"
    epochs = cfg.SOLVER.MAX_EPOCHS

    if "eva02" in cfg.MODEL.PRETRAIN_PATH:
        patch_size = 14
    else:
        patch_size = 16  # 默认值

    h_num = cfg.INPUT.SIZE_TRAIN[0] // patch_size
    w_num = cfg.INPUT.SIZE_TRAIN[1] // patch_size

    print(f"初始化 TopoLoss: Image={cfg.INPUT.SIZE_TRAIN}, Patch={patch_size} -> Grid=({h_num}, {w_num})")

    cmt_loss_fn = ModalityConsistencyLoss()
    topo_loss_fn = TopologicalConsistencyLoss(h_num=h_num, w_num=w_num)

    logger = logging.getLogger("transreid.train")
    logger.info("start training")

    if device:
        model.to(local_rank)
        if torch.cuda.device_count() > 1 and cfg.MODEL.DIST_TRAIN:
            print("Using {} GPUs for training".format(torch.cuda.device_count()))
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True)

    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    scaler = amp.GradScaler()

    # train
    if torch.cuda.device_count() > 1 and cfg.MODEL.DIST_TRAIN:
        model.module.train_with_single()
    else:
        model.train_with_single()
    for epoch in range(1, epochs + 1):
        start_time = time.time()
        loss_meter.reset()
        acc_meter.reset()
        evaluator.reset()
        scheduler.step(epoch)
        # is_stage_one = epoch <= cfg.INPUT.GRAYSCALE_EPOCH
        model.train()
        for n_iter, (img, vid, target_cam, target_view, img_wh) in enumerate(train_loader):
            # if is_stage_one:
            #     # 找到光学图片的索引
            #     rgb_idx = target_cam == 0
            #     if rgb_idx.any():
            #         # 转灰度
            #         # img[rgb_idx] = TF.rgb_to_grayscale(img[rgb_idx], num_output_channels=3)
            #         # 或者简单的通道平均
            #         gray = img[rgb_idx].mean(dim=1, keepdim=True).repeat(1, 3, 1, 1)
            #         img[rgb_idx] = gray
            optimizer.zero_grad()
            optimizer_center.zero_grad()
            img = img.to(device)
            target = vid.to(device)
            target_cam = target_cam.to(device)
            img_wh = img_wh.to(device)
            with amp.autocast(enabled=True):
                outputs = model(img, target, cam_label=target_cam, img_wh=img_wh)

                loss_cyc = torch.tensor(0.0).to(device)
                loss_topo = torch.tensor(0.0).to(device)
                # loss_cpm = ... (删除)

                # === 修改处 1: 判断条件改为 5 ===
                if isinstance(outputs, tuple) and len(outputs) == 5:
                    # 解包 5 个变量
                    cls_score, f_final, f_comp, saliency, attn_map = outputs

                    # 1. 基础 Loss
                    loss_base = loss_fn(cls_score, f_final, target, target_cam)

                    # 2. ST-CMT: 模态一致性 Loss (带显著性)
                    if f_comp is not None:
                        loss_cyc = cmt_loss_fn(f_final, f_comp, target, target_cam, saliency_scores=saliency)

                    # 3. ST-CMT: 拓扑一致性 Loss
                    if attn_map is not None:
                        loss_topo = topo_loss_fn(attn_map)

                    # === 修改处 2: 总 Loss ===
                    loss = loss_base + cfg.MODEL.CYC_LOSS_WEIGHT * loss_cyc + cfg.MODEL.TOPO_LOSS_WEIGHT * loss_topo
                    # + loss_cpm (已删除)

                elif isinstance(outputs, tuple) and len(outputs) == 2:
                    # Baseline 情况
                    cls_score, f_final = outputs
                    loss = loss_fn(cls_score, f_final, target, target_cam)
                else:
                    loss = torch.tensor(0.0, requires_grad=True).to(device)

            scaler.scale(loss).backward()

            scaler.step(optimizer)
            scaler.update()

            if "center" in cfg.MODEL.METRIC_LOSS_TYPE:
                for param in center_criterion.parameters():
                    param.grad.data *= 1.0 / cfg.SOLVER.CENTER_LOSS_WEIGHT
                scaler.step(optimizer_center)
                scaler.update()

            # === 修改开始：把 score 改为 cls_score ===
            if isinstance(cls_score, list):
                acc = (cls_score[0].max(1)[1] == target).float().mean()
            else:
                acc = (cls_score.max(1)[1] == target).float().mean()
            # === 修改结束 ===

            loss_meter.update(loss.item(), img.shape[0])
            acc_meter.update(acc, 1)

            torch.cuda.synchronize()
            if (n_iter + 1) % log_period == 0:
                logger.info(
                    "Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Acc: {:.3f}, Base Lr: {:.2e}".format(
                        epoch, (n_iter + 1), len(train_loader), loss_meter.avg, acc_meter.avg, scheduler._get_lr(epoch)[0]
                    )
                )

        end_time = time.time()
        time_per_batch = (end_time - start_time) / (n_iter + 1)
        if cfg.MODEL.DIST_TRAIN:
            pass
        else:
            logger.info(
                "Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]".format(
                    epoch, time_per_batch, train_loader.batch_size / time_per_batch
                )
            )

        if epoch % checkpoint_period == 0:
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    torch.save(model.state_dict(), os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + "_{}.pth".format(epoch)))
            else:
                torch.save(model.state_dict(), os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + "_{}.pth".format(epoch)))

        if epoch % eval_period == 0:
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    model.eval()
                    for n_iter, (img, vid, camid, camids, target_view, _, img_wh) in enumerate(val_loader):
                        with torch.no_grad():
                            img = img.to(device)
                            camids = camids.to(device)
                            target_view = target_view.to(device)
                            img_wh = img_wh.to(device)
                            feat = model(img, cam_label=camids, img_wh=img_wh)
                            evaluator.update((feat, vid, camid))
                    cmc, mAP, _, _, _, _, _ = evaluator.compute()
                    logger.info("Validation Results - Epoch: {}".format(epoch))
                    logger.info("mAP: {:.1%}".format(mAP))
                    for r in [1, 5, 10]:
                        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
                    torch.cuda.empty_cache()
            else:
                model.eval()
                for n_iter, (img, vid, camid, camids, target_view, _, img_wh) in enumerate(val_loader):
                    with torch.no_grad():
                        img = img.to(device)
                        camids = camids.to(device)
                        img_wh = img_wh.to(device)
                        feat = model(img, cam_label=camids, img_wh=img_wh)
                        evaluator.update((feat, vid, camid))
                cmc, mAP, _, _, _, _, _ = evaluator.compute()
                logger.info("Validation Results - Epoch: {}".format(epoch))
                logger.info("mAP: {:.1%}".format(mAP))
                for r in [1, 5, 10]:
                    logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
                torch.cuda.empty_cache()


def do_inference(cfg, model, val_loader, num_query):
    device = "cuda"
    logger = logging.getLogger("transreid.test")
    logger.info("Enter inferencing")

    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)

    evaluator.reset()

    if device:
        if torch.cuda.device_count() > 1:
            print("Using {} GPUs for inference".format(torch.cuda.device_count()))
            model = nn.DataParallel(model)
        model.to(device)

    model.eval()

    for n_iter, (img, pid, camid, camids, _, _, img_wh) in enumerate(val_loader):
        with torch.no_grad():
            img = img.to(device)
            camids = camids.to(device)
            img_wh = img_wh.to(device)
            feat = model(img, cam_label=camids, img_wh=img_wh)
            evaluator.update((feat, pid, camid))

    cmc, mAP, _, _, _, _, _ = evaluator.compute()
    logger.info("Inference Results ")
    logger.info("mAP: {:.1%}".format(mAP))
    for r in [1, 5, 10]:
        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
    return cmc[0], cmc[4]
