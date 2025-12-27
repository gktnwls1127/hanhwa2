import os
import torch
import torch.nn as nn
import numpy as np
from Command_corpus import Command_Corpus
from Command_dataset import Command_DevTest_Dataset
from torch.utils.data import DataLoader
from evaluate import scoring


def compute_scores(model: nn.Module, command_corpus: Command_Corpus, config, batch_size: int, mode: str, result_file: str):
    assert mode in ['dev', 'test'], 'mode must be chosen from \'dev\' or \'test\''
    dataloader = DataLoader(Command_DevTest_Dataset(command_corpus, config, mode), batch_size=batch_size, shuffle=False, num_workers=max(0, batch_size // 16), pin_memory=True)
    indices = (command_corpus.dev_indices if mode == 'dev' else command_corpus.test_indices)
    
    # scores dict: sample_idx -> score
    scores = {}
    
    if config.gpu_available:
        torch.cuda.empty_cache()
    model.eval()
    with torch.no_grad():
        for (user_ID, user_dept, user_pos, user_rank, user_unit, user_title_text, user_content_text, user_time_text, user_history_mask, user_history_graph, user_history_category_mask, user_history_category_indices, \
             report_title, report_title_mask, report_content_text, report_content_mask, report_time_text, report_time_mask, report_category, user_history_category, sample_idx) in dataloader:
            if config.gpu_available:
                user_ID = user_ID.cuda(non_blocking=True)
                user_dept = user_dept.cuda(non_blocking=True)
                user_pos = user_pos.cuda(non_blocking=True)
                user_rank = user_rank.cuda(non_blocking=True)
                user_unit = user_unit.cuda(non_blocking=True)
                user_title_text = user_title_text.cuda(non_blocking=True)
                user_content_text = user_content_text.cuda(non_blocking=True)
                user_time_text = user_time_text.cuda(non_blocking=True)
                user_history_mask = user_history_mask.cuda(non_blocking=True)
                user_history_graph = user_history_graph.cuda(non_blocking=True)
                user_history_category_mask = user_history_category_mask.cuda(non_blocking=True)
                user_history_category_indices = user_history_category_indices.cuda(non_blocking=True)

                report_title = report_title.cuda(non_blocking=True)
                report_title_mask = report_title_mask.cuda(non_blocking=True)
                report_content_text = report_content_text.cuda(non_blocking=True)
                report_content_mask = report_content_mask.cuda(non_blocking=True)
                report_time_text = report_time_text.cuda(non_blocking=True)
                report_time_mask = report_time_mask.cuda(non_blocking=True)
                report_category = report_category.cuda(non_blocking=True)
                user_history_category = user_history_category.cuda(non_blocking=True)

            batch_size = user_ID.size(0)
            report_title = report_title.unsqueeze(dim=1)
            report_title_mask = report_title_mask.unsqueeze(dim=1)
            report_content_text = report_content_text.unsqueeze(dim=1)
            report_content_mask = report_content_mask.unsqueeze(dim=1)
            report_time_text = report_time_text.unsqueeze(dim=1)
            report_time_mask = report_time_mask.unsqueeze(dim=1)
            report_category = report_category.unsqueeze(dim=1)
            batch_scores = model(user_ID, user_dept, user_pos, user_rank, user_unit, user_title_text, user_content_text, user_time_text, user_history_mask, user_history_graph, user_history_category_mask, user_history_category_indices, None, \
                                                    report_title, report_title_mask, report_content_text, report_content_mask, report_time_text, report_time_mask, report_category, user_history_category, sample_idx).squeeze(dim=1) # [batch_size]
            
            # sample_idx를 CPU로 변환하고 batch_scores와 함께 저장
            batch_sample_idx = sample_idx.cpu().numpy() if isinstance(sample_idx, torch.Tensor) else sample_idx
            batch_scores_np = batch_scores.cpu().numpy() if isinstance(batch_scores, torch.Tensor) else batch_scores
            
            for idx, score in zip(batch_sample_idx, batch_scores_np):
                scores[int(idx)] = float(score)
    
    # indices는 이제 dict: sample_idx -> (impression_id, pos, label)
    impression_scores = {}  # impression_id -> {pos: score}
    
    for sample_idx in scores.keys():
        if sample_idx in indices:
            impression_id, pos, label = indices[sample_idx]
            
            if impression_id not in impression_scores:
                impression_scores[impression_id] = {}
            impression_scores[impression_id][pos] = scores[sample_idx]
    
    with open(result_file, 'w', encoding='utf-8') as result_f:
        max_impression_id = max(impression_scores.keys()) if impression_scores else 0
        for imp_id in range(1, max_impression_id + 1):
            if imp_id not in impression_scores:
                result_f.write(('' if imp_id == 1 else '\n') + str(imp_id) + ' []')
                continue

            pos2score = impression_scores[imp_id]
            n = max(pos2score.keys()) + 1
            score_list = [pos2score.get(p, float('-inf')) for p in range(n)]

            # score -> rank (1이 best)
            order = sorted(range(n), key=lambda k: score_list[k], reverse=True)
            rank = [0]*n
            for r, k in enumerate(order, start=1):
                rank[k] = r

            result_f.write(('' if imp_id == 1 else '\n') + str(imp_id) + ' ' + str(rank).replace(' ', ''))
    if config.dataset != 'large' or mode != 'test':
        with open(mode + '/ref/truth-%s.txt' % config.dataset, 'r', encoding='utf-8') as truth_f, open(result_file, 'r', encoding='utf-8') as result_f:
            auc, mrr, ndcg5, ndcg10 = scoring(truth_f, result_f)
        return auc, mrr, ndcg5, ndcg10
    else:
        return None, None, None, None


def get_run_index(result_dir: str):
    assert os.path.exists(result_dir), 'result directory does not exist'
    max_index = 0
    for result_file in os.listdir(result_dir):
        if result_file.strip()[0] == '#' and result_file.strip()[-4:] == '-dev':
            index = int(result_file.strip()[1:-4])
            max_index = max(index, max_index)
    with open(result_dir + '/#' + str(max_index + 1) + '-dev', 'w', encoding='utf-8') as result_f:
        pass
    return max_index + 1


class AvgMetric:
    def __init__(self, auc, mrr, ndcg5, ndcg10):
        self.auc = auc
        self.mrr = mrr
        self.ndcg5 = ndcg5
        self.ndcg10 = ndcg10
        self.avg = (self.auc + self.mrr + (self.ndcg5 + self.ndcg10) / 2) / 3

    def __gt__(self, value):
        return self.avg > value.avg

    def __ge__(self, value):
        return self.avg >= value.avg

    def __lt__(self, value):
        return self.avg < value.avg

    def __le__(self, value):
        return self.avg <= value.avg

    def __str__(self):
        return '%.4f\nAUC = %.4f\nMRR = %.4f\nnDCG@5 = %.4f\nnDCG@10 = %.4f' % (self.avg, self.auc, self.mrr, self.ndcg5, self.ndcg10)