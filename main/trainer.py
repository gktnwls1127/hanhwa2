import os
import signal
import shutil
import json
from config import Config
from Command_corpus import Command_Corpus
from Command_dataset import Command_Train_Dataset
from util import AvgMetric
from util import compute_scores
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


class Trainer:
    def __init__(self, model: nn.Module, config: Config, command_corpus: Command_Corpus, run_index: int):
        self.model = model
        self.config = config
        self.epoch = config.epoch
        self.batch_size = config.batch_size
        self.max_history_num = config.max_history_num
        self.negative_sample_num = config.negative_sample_num
        self.loss = self.negative_log_softmax if config.click_predictor in ['dot_product', 'mlp', 'FIM'] else self.negative_log_sigmoid
        self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=config.lr, weight_decay=config.weight_decay)
        self._dataset = config.dataset
        self.command_corpus = command_corpus
        self.train_dataset = Command_Train_Dataset(command_corpus, config)
        self.run_index = run_index
        self.model_dir = config.model_dir + '/#' + str(self.run_index)
        self.best_model_dir = config.best_model_dir + '/#' + str(self.run_index)
        self.dev_res_dir = config.dev_res_dir + '/#' + str(self.run_index)
        self.result_dir = config.result_dir
        if not os.path.exists(self.model_dir):
            os.mkdir(self.model_dir)
        if not os.path.exists(self.best_model_dir):
            os.mkdir(self.best_model_dir)
        if not os.path.exists(self.dev_res_dir):
            os.mkdir(self.dev_res_dir)
        with open(config.config_dir + '/#' + str(self.run_index) + '.json', 'w', encoding='utf-8') as f:
            json.dump(config.attribute_dict, f)
        if self._dataset == 'large':
            self.prediction_dir = config.prediction_dir + '/#' + str(self.run_index)
            os.mkdir(self.prediction_dir)
        self.dev_criterion = config.dev_criterion
        self.early_stopping_epoch = config.early_stopping_epoch
        self.auc_results = []
        self.mrr_results = []
        self.ndcg5_results = []
        self.ndcg10_results = []
        self.best_dev_epoch = 0
        self.best_dev_auc = 0
        self.best_dev_mrr = 0
        self.best_dev_ndcg5 = 0
        self.best_dev_ndcg10 = 0
        self.best_dev_avg = AvgMetric(0, 0, 0, 0)
        self.epoch_not_increase = 0
        self.gradient_clip_norm = config.gradient_clip_norm
        if config.gpu_available:
            self.model.cuda()
        else:
            self.model.cpu()
        print('Running : ' + self.model.model_name + '\t#' + str(self.run_index))

    def negative_log_softmax(self, logits):
        loss = (-torch.log_softmax(logits, dim=1).select(dim=1, index=0)).mean()
        return loss

    def negative_log_sigmoid(self, logits):
        positive_sigmoid = torch.clamp(torch.sigmoid(logits[:, 0]), min=1e-15, max=1)
        negative_sigmoid = torch.clamp(torch.sigmoid(-logits[:, 1:]), min=1e-15, max=1)
        loss = -(torch.log(positive_sigmoid).sum() + torch.log(negative_sigmoid).sum()) / logits.numel()
        return loss

    def train(self):
        model = self.model
        for e in tqdm(range(1, self.epoch + 1)):
            self.train_dataset.negative_sampling()
            train_dataloader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=max(0, self.batch_size // 16), pin_memory=True)
            model.train()
            epoch_loss = 0
            for (cmd_title_text, cmd_title_mask, cmd_content_text, cmd_content_mask, cmd_time_text, cmd_time_mask, cmd_category, \
                    cand_user_ID, cand_dept, cand_pos, cand_rank, cand_unit, cand_title_text, cand_title_mask, cand_content_text, cand_content_mask, cand_time_text, cand_time_mask, \
                        cand_hist_category, cand_hist_mask, cand_hist_graph, cand_cat_mask, cand_cat_idx, pos_index) in train_dataloader:
                if self.config.gpu_available:
                    cmd_title_text   = cmd_title_text.cuda(non_blocking=True)
                    cmd_title_mask   = cmd_title_mask.cuda(non_blocking=True)
                    cmd_content_text = cmd_content_text.cuda(non_blocking=True)
                    cmd_content_mask = cmd_content_mask.cuda(non_blocking=True)
                    cmd_time_text    = cmd_time_text.cuda(non_blocking=True)
                    cmd_time_mask    = cmd_time_mask.cuda(non_blocking=True)
                    cmd_category     = cmd_category.cuda(non_blocking=True)

                    cand_user_ID = cand_user_ID.cuda(non_blocking=True)
                    cand_dept    = cand_dept.cuda(non_blocking=True)
                    cand_pos     = cand_pos.cuda(non_blocking=True)
                    cand_rank    = cand_rank.cuda(non_blocking=True)
                    cand_unit    = cand_unit.cuda(non_blocking=True)

                    cand_title_text   = cand_title_text.cuda(non_blocking=True)
                    cand_title_mask   = cand_title_mask.cuda(non_blocking=True)
                    cand_content_text = cand_content_text.cuda(non_blocking=True)
                    cand_content_mask = cand_content_mask.cuda(non_blocking=True)
                    cand_time_text    = cand_time_text.cuda(non_blocking=True)
                    cand_time_mask    = cand_time_mask.cuda(non_blocking=True)

                    cand_hist_category = cand_hist_category.cuda(non_blocking=True)
                    cand_hist_mask     = cand_hist_mask.cuda(non_blocking=True)

                    # graph류는 모델에 따라 None일 수도 있으니 안전 처리
                    if cand_hist_graph is not None: cand_hist_graph = cand_hist_graph.cuda(non_blocking=True)
                    if cand_cat_mask is not None:   cand_cat_mask   = cand_cat_mask.cuda(non_blocking=True)
                    if cand_cat_idx is not None:    cand_cat_idx    = cand_cat_idx.cuda(non_blocking=True)

                logits = model(cmd_title_text, cmd_title_mask, cmd_content_text, cmd_content_mask, cmd_time_text, cmd_time_mask, cmd_category,
                                cand_user_ID, cand_dept, cand_pos, cand_rank, cand_unit, cand_title_text, cand_title_mask, cand_content_text, cand_content_mask, cand_time_text, cand_time_mask, cand_hist_category, cand_hist_mask, cand_hist_graph, cand_cat_mask, cand_cat_idx)
                
                loss = self.loss(logits)
                if model.report_encoder.auxiliary_loss is not None:
                    report_auxiliary_loss = model.report_encoder.auxiliary_loss.mean()
                    loss += report_auxiliary_loss
                if model.user_encoder.auxiliary_loss is not None:
                    user_encoder_auxiliary_loss = model.user_encoder.auxiliary_loss.mean()
                    loss += user_encoder_auxiliary_loss
                epoch_loss += float(loss) * cand_user_ID.size(0)
                self.optimizer.zero_grad()
                loss.backward()
                if self.gradient_clip_norm > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), self.gradient_clip_norm)
                self.optimizer.step()
            print('Epoch %d : train done' % e)
            print('loss =', epoch_loss / len(self.train_dataset))

            # validation
            auc, mrr, ndcg5, ndcg10 = compute_scores(model, self.command_corpus, self.config, self.batch_size * 3 // 2, 'dev', self.dev_res_dir + '/' + model.model_name + '-' + str(e) + '.txt')
            self.auc_results.append(auc)
            self.mrr_results.append(mrr)
            self.ndcg5_results.append(ndcg5)
            self.ndcg10_results.append(ndcg10)
            print('Epoch %d : dev done\nDev criterions' % e)
            print('AUC = {:.4f}\nMRR = {:.4f}\nnDCG@5 = {:.4f}\nnDCG@10 = {:.4f}'.format(auc, mrr, ndcg5, ndcg10))
            if self.dev_criterion == 'auc':
                if auc >= self.best_dev_auc:
                    self.best_dev_auc = auc
                    self.best_dev_epoch = e
                    with open(self.result_dir + '/#' + str(self.run_index) + '-dev', 'w') as result_f:
                        result_f.write('#' + str(self.run_index) + '\t' + str(auc) + '\t' + str(mrr) + '\t' + str(ndcg5) + '\t' + str(ndcg10) + '\n')
                    self.epoch_not_increase = 0
                else:
                    self.epoch_not_increase += 1
            elif self.dev_criterion == 'mrr':
                if mrr >= self.best_dev_mrr:
                    self.best_dev_mrr = mrr
                    self.best_dev_epoch = e
                    with open(self.result_dir + '/#' + str(self.run_index) + '-dev', 'w') as result_f:
                        result_f.write('#' + str(self.run_index) + '\t' + str(auc) + '\t' + str(mrr) + '\t' + str(ndcg5) + '\t' + str(ndcg10) + '\n')
                    self.epoch_not_increase = 0
                else:
                    self.epoch_not_increase += 1
            elif self.dev_criterion == 'ndcg5':
                if ndcg5 >= self.best_dev_ndcg5:
                    self.best_dev_ndcg5 = ndcg5
                    self.best_dev_epoch = e
                    with open(self.result_dir + '/#' + str(self.run_index) + '-dev', 'w') as result_f:
                        result_f.write('#' + str(self.run_index) + '\t' + str(auc) + '\t' + str(mrr) + '\t' + str(ndcg5) + '\t' + str(ndcg10) + '\n')
                    self.epoch_not_increase = 0
                else:
                    self.epoch_not_increase += 1
            elif self.dev_criterion == 'ndcg10':
                if ndcg10 >= self.best_dev_ndcg10:
                    self.best_dev_ndcg10 = ndcg10
                    self.best_dev_epoch = e
                    with open(self.result_dir + '/#' + str(self.run_index) + '-dev', 'w') as result_f:
                        result_f.write('#' + str(self.run_index) + '\t' + str(auc) + '\t' + str(mrr) + '\t' + str(ndcg5) + '\t' + str(ndcg10) + '\n')
                    self.epoch_not_increase = 0
                else:
                    self.epoch_not_increase += 1
            else:
                avg = AvgMetric(auc, mrr, ndcg5, ndcg10)
                if avg >= self.best_dev_avg:
                    self.best_dev_avg = avg
                    self.best_dev_epoch = e
                    with open(self.result_dir + '/#' + str(self.run_index) + '-dev', 'w') as result_f:
                        result_f.write('#' + str(self.run_index) + '\t' + str(auc) + '\t' + str(mrr) + '\t' + str(ndcg5) + '\t' + str(ndcg10) + '\n')
                    self.epoch_not_increase = 0
                else:
                    self.epoch_not_increase += 1

            print('Best epoch :', self.best_dev_epoch)
            print('Best ' + self.dev_criterion + ' : ' + str(getattr(self, 'best_dev_' + self.dev_criterion)))
            if self.config.gpu_available:
                torch.cuda.empty_cache()
            if self.epoch_not_increase == 0:
                torch.save({model.model_name: model.state_dict()}, self.model_dir + '/' + model.model_name + '-' + str(self.best_dev_epoch))
            if self.epoch_not_increase == self.early_stopping_epoch:
                break

        with open('%s/%s-%s-dev_log.txt' % (self.dev_res_dir, model.model_name, self._dataset), 'w', encoding='utf-8') as f:
            f.write('Epoch\tAUC\tMRR\tnDCG@5\tnDCG@10\n')
            for i in range(len(self.auc_results)):
                f.write('%d\t%.4f\t%.4f\t%.4f\t%.4f\n' % (i + 1, self.auc_results[i], self.mrr_results[i], self.ndcg5_results[i], self.ndcg10_results[i]))
        shutil.copy(self.model_dir + '/' + model.model_name + '-' + str(self.best_dev_epoch), self.best_model_dir + '/' + model.model_name)
        print('Training : ' + model.model_name + ' #' + str(self.run_index) + ' completed\nDev criterions:')
        print('AUC : %.4f' % self.auc_results[self.best_dev_epoch - 1])
        print('MRR : %.4f' % self.mrr_results[self.best_dev_epoch - 1])
        print('nDCG@5 : %.4f' % self.ndcg5_results[self.best_dev_epoch - 1])
        print('nDCG@10 : %.4f' % self.ndcg10_results[self.best_dev_epoch - 1])


def negative_log_softmax(logits):
    loss = (-torch.log_softmax(logits, dim=1).select(dim=1, index=0)).mean()
    return loss

def negative_log_sigmoid(logits):
    positive_sigmoid = torch.clamp(torch.sigmoid(logits[:, 0]), min=1e-15, max=1)
    negative_sigmoid = torch.clamp(torch.sigmoid(-logits[:, 1:]), min=1e-15, max=1)
    loss = -(torch.log(positive_sigmoid).sum() + torch.log(negative_sigmoid).sum()) / logits.numel()
    return loss

def distributed_train(rank, model: nn.Module, config: Config, command_corpus: Command_Corpus, run_index: int):
    world_size = config.world_size
    model_name = model.model_name

    dist.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
    config.device_id = rank
    config.set_cuda()

    model.cuda()
    model = DDP(model, device_ids=[rank])

    loss_fn = negative_log_softmax if config.click_predictor in ['dot_product', 'mlp', 'FIM'] else negative_log_sigmoid
    epoch = config.epoch
    batch_size = config.batch_size // world_size
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.module.parameters()),
        lr=config.lr,
        weight_decay=config.weight_decay
    )
    gradient_clip_norm = config.gradient_clip_norm

    train_dataset = Command_Train_Dataset(command_corpus, config)

    # rank0만 디렉토리/로그 담당
    if rank == 0:
        model_dir = config.model_dir + '/#' + str(run_index)
        best_model_dir = config.best_model_dir + '/#' + str(run_index)
        dev_res_dir = config.dev_res_dir + '/#' + str(run_index)
        result_dir = config.result_dir

        os.makedirs(model_dir, exist_ok=True)
        os.makedirs(best_model_dir, exist_ok=True)
        os.makedirs(dev_res_dir, exist_ok=True)

        with open(config.config_dir + '/#' + str(run_index) + '.json', 'w', encoding='utf-8') as f:
            json.dump(config.attribute_dict, f)

        dev_criterion = config.dev_criterion
        early_stopping_epoch = config.early_stopping_epoch

        auc_results, mrr_results, ndcg5_results, ndcg10_results = [], [], [], []
        best_dev_epoch = 0
        best_dev_auc = best_dev_mrr = best_dev_ndcg5 = best_dev_ndcg10 = 0
        best_dev_avg = AvgMetric(0, 0, 0, 0)
        epoch_not_increase = 0

        print('Running : ' + model_name + '\t#' + str(run_index))

    for e in tqdm(range(1, epoch + 1)):
        # epoch마다 negative sampling
        train_dataset.negative_sampling(rank=rank)

        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset, num_replicas=world_size, rank=rank, shuffle=True
        )
        train_sampler.set_epoch(e)

        train_dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            num_workers=max(0, batch_size // 16),
            pin_memory=True,
            sampler=train_sampler
        )

        model.train()
        epoch_loss = 0.0

        # ✅ Command_Train_Dataset 반환값(= 단일 GPU train 루프와 동일)으로 unpack
        for (cmd_title_text, cmd_title_mask, cmd_content_text, cmd_content_mask, cmd_time_text, cmd_time_mask, cmd_category,
             cand_user_ID, cand_dept, cand_pos, cand_rank, cand_unit,
             cand_title_text, cand_title_mask, cand_content_text, cand_content_mask, cand_time_text, cand_time_mask,
             cand_hist_category, cand_hist_mask, cand_hist_graph, cand_cat_mask, cand_cat_idx,
             pos_index) in train_dataloader:

            # to cuda
            cmd_title_text   = cmd_title_text.cuda(non_blocking=True)
            cmd_title_mask   = cmd_title_mask.cuda(non_blocking=True)
            cmd_content_text = cmd_content_text.cuda(non_blocking=True)
            cmd_content_mask = cmd_content_mask.cuda(non_blocking=True)
            cmd_time_text    = cmd_time_text.cuda(non_blocking=True)
            cmd_time_mask    = cmd_time_mask.cuda(non_blocking=True)
            cmd_category     = cmd_category.cuda(non_blocking=True)

            cand_user_ID = cand_user_ID.cuda(non_blocking=True)
            cand_dept    = cand_dept.cuda(non_blocking=True)
            cand_pos     = cand_pos.cuda(non_blocking=True)
            cand_rank    = cand_rank.cuda(non_blocking=True)
            cand_unit    = cand_unit.cuda(non_blocking=True)

            cand_title_text   = cand_title_text.cuda(non_blocking=True)
            cand_title_mask   = cand_title_mask.cuda(non_blocking=True)
            cand_content_text = cand_content_text.cuda(non_blocking=True)
            cand_content_mask = cand_content_mask.cuda(non_blocking=True)
            cand_time_text    = cand_time_text.cuda(non_blocking=True)
            cand_time_mask    = cand_time_mask.cuda(non_blocking=True)

            cand_hist_category = cand_hist_category.cuda(non_blocking=True)
            cand_hist_mask     = cand_hist_mask.cuda(non_blocking=True)

            # graph류는 항상 텐서로 들어오는 전제(네 Dataset 구현 기준)라 그냥 cuda 처리
            cand_hist_graph = cand_hist_graph.cuda(non_blocking=True)
            cand_cat_mask   = cand_cat_mask.cuda(non_blocking=True)
            cand_cat_idx    = cand_cat_idx.cuda(non_blocking=True)

            # ✅ forward 인자 수/순서를 단일 GPU와 동일하게 맞춤 (pos_index는 모델에 전달 X)
            logits = model(
                cmd_title_text, cmd_title_mask,
                cmd_content_text, cmd_content_mask,
                cmd_time_text, cmd_time_mask,
                cmd_category,

                cand_user_ID, cand_dept, cand_pos, cand_rank, cand_unit,
                cand_title_text, cand_title_mask,
                cand_content_text, cand_content_mask,
                cand_time_text, cand_time_mask,
                cand_hist_category, cand_hist_mask,
                cand_hist_graph, cand_cat_mask, cand_cat_idx
            )  # [B, 1+neg]

            loss = loss_fn(logits)

            # auxiliary loss
            if getattr(model.module.report_encoder, "auxiliary_loss", None) is not None:
                loss = loss + model.module.report_encoder.auxiliary_loss.mean()
            if getattr(model.module.user_encoder, "auxiliary_loss", None) is not None:
                loss = loss + model.module.user_encoder.auxiliary_loss.mean()

            epoch_loss += float(loss) * cand_user_ID.size(0)

            optimizer.zero_grad()
            loss.backward()
            if gradient_clip_norm > 0:
                nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
            optimizer.step()

        # 로그
        if rank == 0:
            print(f'Epoch {e} : train done')
        dist.barrier()

        # ✅ dev는 rank0만
        if rank == 0:
            auc, mrr, ndcg5, ndcg10 = compute_scores(
                model.module, command_corpus, config, batch_size * 3 // 2,
                'dev', dev_res_dir + '/' + model_name + '-' + str(e) + '.txt'
            )
            auc_results.append(auc); mrr_results.append(mrr); ndcg5_results.append(ndcg5); ndcg10_results.append(ndcg10)

            print('Epoch %d : dev done\nDev criterions' % e)
            print('AUC = {:.4f}\nMRR = {:.4f}\nnDCG@5 = {:.4f}\nnDCG@10 = {:.4f}'.format(auc, mrr, ndcg5, ndcg10))

            # early-stopping 업데이트(기존 로직 유지)
            improved = False
            if dev_criterion == 'auc' and auc >= best_dev_auc:
                best_dev_auc = auc; improved = True
            elif dev_criterion == 'mrr' and mrr >= best_dev_mrr:
                best_dev_mrr = mrr; improved = True
            elif dev_criterion == 'ndcg5' and ndcg5 >= best_dev_ndcg5:
                best_dev_ndcg5 = ndcg5; improved = True
            elif dev_criterion == 'ndcg10' and ndcg10 >= best_dev_ndcg10:
                best_dev_ndcg10 = ndcg10; improved = True
            elif dev_criterion not in ['auc', 'mrr', 'ndcg5', 'ndcg10']:
                avg = AvgMetric(auc, mrr, ndcg5, ndcg10)
                if avg >= best_dev_avg:
                    best_dev_avg = avg; improved = True

            if improved:
                best_dev_epoch = e
                epoch_not_increase = 0
                with open(result_dir + '/#' + str(run_index) + '-dev', 'w') as result_f:
                    result_f.write('#' + str(run_index) + '\t' + str(auc) + '\t' + str(mrr) + '\t' + str(ndcg5) + '\t' + str(ndcg10) + '\n')
                torch.save({model_name: model.module.state_dict()}, model_dir + '/' + model_name + '-' + str(best_dev_epoch))
            else:
                epoch_not_increase += 1

            print('Best epoch :', best_dev_epoch)
            torch.cuda.empty_cache()

            if epoch_not_increase >= early_stopping_epoch:
                break

        dist.barrier()

    # rank0 마무리
    if rank == 0:
        with open('%s/%s-%s-dev_log.txt' % (dev_res_dir, model_name, config.dataset), 'w', encoding='utf-8') as f:
            f.write('Epoch\tAUC\tMRR\tnDCG@5\tnDCG@10\n')
            for i in range(len(auc_results)):
                f.write('%d\t%.4f\t%.4f\t%.4f\t%.4f\n' % (i + 1, auc_results[i], mrr_results[i], ndcg5_results[i], ndcg10_results[i]))

        shutil.copy(model_dir + '/' + model_name + '-' + str(best_dev_epoch), best_model_dir + '/' + model_name)
        print('Training : ' + model_name + ' #' + str(run_index) + ' completed')
        print('AUC : %.4f' % auc_results[best_dev_epoch - 1])
        print('MRR : %.4f' % mrr_results[best_dev_epoch - 1])
        print('nDCG@5 : %.4f' % ndcg5_results[best_dev_epoch - 1])
        print('nDCG@10 : %.4f' % ndcg10_results[best_dev_epoch - 1])

        os.kill(os.getpid(), signal.SIGKILL)
