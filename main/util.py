import os
import torch
import torch.nn as nn
import numpy as np
import json
from Command_corpus import Command_Corpus
from Command_dataset import Command_DevTest_Dataset
from torch.utils.data import DataLoader
from evaluate import scoring


def compute_scores(model: nn.Module, command_corpus: Command_Corpus, config, batch_size: int, mode: str, result_file: str):
    assert mode in ['dev', 'test'], "mode must be chosen from 'dev' or 'test'"

    dataset = config.dataset
    truth_path = f"{mode}/ref/truth-{dataset}.txt"

    # truth 라인 수(=impression 수) 확보 + 각 impression의 후보 수(labels 길이)도 확보
    with open(truth_path, "r", encoding="utf-8") as tf:
        truth_lines = tf.readlines()
    impression_num = len(truth_lines)
    truth_sizes = []
    for line in truth_lines:
        _impid, labels_json = line.strip("\n").split()
        truth_sizes.append(len(json.loads(labels_json)))

    # corpus indices (pair 단위) 사용
    indices = command_corpus.dev_indices if mode == "dev" else command_corpus.test_indices
    if len(indices) == 0:
        raise RuntimeError(f"[compute_scores] {mode}_indices is empty")

    # DataLoader: shuffle=False 중요 (corpus.dev_userDataset/test_userDataset 순서 유지)
    dataloader = DataLoader(
        Command_DevTest_Dataset(command_corpus, config, mode),
        batch_size=batch_size,
        shuffle=False,
        num_workers=max(0, batch_size // 16),
        pin_memory=True
    )

    # ------------------------------------------------------------------
    # 1) dataloader 순서대로 점수(score_list) 누적
    #    이 순서가 indices(=pair 단위 dev_ID/test_ID 반복) 순서와 같아야 함
    # ------------------------------------------------------------------
    score_list = []

    if config.gpu_available:
        torch.cuda.empty_cache()

    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            (user_idx, user_dept, user_pos, user_rank, user_unit,
             cand_title_text, cand_title_mask,
             cand_content_text, cand_content_mask,
             cand_time_text, cand_time_mask,
             cand_hist_category, cand_hist_mask,
             cand_hist_graph, cand_cat_mask, cand_cat_idx,
             cmd_title_text, cmd_title_mask,
             cmd_content_text, cmd_content_mask,
             cmd_time_text, cmd_time_mask,
             cmd_category,
             sample_idx) = batch  # sample_idx는 들어오지만, 여기서는 corpus.indices만 사용

            # Dev/Test pair 평가: user_idx는 [B,1] 유지 (Model.forward가 K차원 필요)
            if user_idx.dim() == 1:
                user_idx  = user_idx.unsqueeze(1)
                user_dept = user_dept.unsqueeze(1)
                user_pos  = user_pos.unsqueeze(1)
                user_rank = user_rank.unsqueeze(1)
                user_unit = user_unit.unsqueeze(1)

                cand_title_text   = cand_title_text.unsqueeze(1)
                cand_title_mask   = cand_title_mask.unsqueeze(1)
                cand_content_text = cand_content_text.unsqueeze(1)
                cand_content_mask = cand_content_mask.unsqueeze(1)
                cand_time_text    = cand_time_text.unsqueeze(1)
                cand_time_mask    = cand_time_mask.unsqueeze(1)
                cand_hist_category = cand_hist_category.unsqueeze(1)
                cand_hist_mask     = cand_hist_mask.unsqueeze(1)
                if cand_hist_graph is not None: cand_hist_graph = cand_hist_graph.unsqueeze(1)
                if cand_cat_mask  is not None:  cand_cat_mask  = cand_cat_mask.unsqueeze(1)
                if cand_cat_idx   is not None:  cand_cat_idx   = cand_cat_idx.unsqueeze(1)
            elif user_idx.dim() == 2:
                if user_idx.size(1) != 1:
                    raise RuntimeError(f"[compute_scores] expected user_idx [B,1], got {tuple(user_idx.shape)}")
            else:
                raise RuntimeError(f"[compute_scores] unexpected user_idx dim={user_idx.dim()} shape={tuple(user_idx.shape)}")

            if config.gpu_available:
                user_idx = user_idx.cuda(non_blocking=True)
                user_dept = user_dept.cuda(non_blocking=True)
                user_pos = user_pos.cuda(non_blocking=True)
                user_rank = user_rank.cuda(non_blocking=True)
                user_unit = user_unit.cuda(non_blocking=True)

                cand_title_text = cand_title_text.cuda(non_blocking=True)
                cand_title_mask = cand_title_mask.cuda(non_blocking=True)
                cand_content_text = cand_content_text.cuda(non_blocking=True)
                cand_content_mask = cand_content_mask.cuda(non_blocking=True)
                cand_time_text = cand_time_text.cuda(non_blocking=True)
                cand_time_mask = cand_time_mask.cuda(non_blocking=True)

                cand_hist_category = cand_hist_category.cuda(non_blocking=True)
                cand_hist_mask = cand_hist_mask.cuda(non_blocking=True)

                cand_hist_graph = cand_hist_graph.cuda(non_blocking=True) if cand_hist_graph is not None else None
                cand_cat_mask   = cand_cat_mask.cuda(non_blocking=True) if cand_cat_mask is not None else None
                cand_cat_idx    = cand_cat_idx.cuda(non_blocking=True) if cand_cat_idx is not None else None

                cmd_title_text = cmd_title_text.cuda(non_blocking=True)
                cmd_title_mask = cmd_title_mask.cuda(non_blocking=True)
                cmd_content_text = cmd_content_text.cuda(non_blocking=True)
                cmd_content_mask = cmd_content_mask.cuda(non_blocking=True)
                cmd_time_text = cmd_time_text.cuda(non_blocking=True)
                cmd_time_mask = cmd_time_mask.cuda(non_blocking=True)
                cmd_category = cmd_category.cuda(non_blocking=True)

            # forward -> [B,1] 또는 [B] (모델 구현에 따라)
            batch_scores = model(
                cmd_title_text, cmd_title_mask,
                cmd_content_text, cmd_content_mask,
                cmd_time_text, cmd_time_mask,
                cmd_category,
                user_idx, user_dept, user_pos, user_rank, user_unit,
                cand_title_text, cand_title_mask,
                cand_content_text, cand_content_mask,
                cand_time_text, cand_time_mask,
                cand_hist_category, cand_hist_mask,
                cand_hist_graph, cand_cat_mask, cand_cat_idx
            )

            # 항상 1차원으로 누적 (pair라서 원소 수 = B)
            batch_scores_np = batch_scores.detach().cpu().numpy().reshape(-1)
            score_list.extend([float(x) for x in batch_scores_np])

    # 길이 검증: indices는 pair 단위, score_list도 pair 단위여야 함
    if len(score_list) != len(indices):
        raise RuntimeError(
            f"[compute_scores] length mismatch: scores={len(score_list)} vs indices={len(indices)}\n"
            f"DataLoader 순서(샘플 개수)와 corpus.{mode}_indices 길이가 다릅니다."
        )

    # ------------------------------------------------------------------
    # 2) impression별로 score 모으기 (후보 순서 = append 순서)
    # ------------------------------------------------------------------
    sub_scores = [[] for _ in range(impression_num)]
    for s, imp_idx in zip(score_list, indices):
        imp_idx = int(imp_idx)
        if not (0 <= imp_idx < impression_num):
            raise RuntimeError(f"[compute_scores] imp_idx out of range: {imp_idx} (truth lines={impression_num})")
        sub_scores[imp_idx].append([s, len(sub_scores[imp_idx])])  # [score, original_position]

    # 후보 수 검증 (조용히 틀린 지표 방지)
    for i in range(impression_num):
        if len(sub_scores[i]) != truth_sizes[i]:
            raise RuntimeError(
                f"[compute_scores] candidate count mismatch at impression(line) {i+1}: "
                f"pred={len(sub_scores[i])} vs truth={truth_sizes[i]}"
            )

    # ------------------------------------------------------------------
    # 3) result 파일 작성: impid는 무조건 1..N (truth와 일치)
    # ------------------------------------------------------------------
    os.makedirs(os.path.dirname(result_file), exist_ok=True)
    with open(result_file, "w", encoding="utf-8") as result_f:
        for i, sub_score in enumerate(sub_scores):
            sub_score.sort(key=lambda x: x[0], reverse=True)
            result = [0] * len(sub_score)
            for j in range(len(sub_score)):
                result[sub_score[j][1]] = j + 1
            result_f.write(("" if i == 0 else "\n") + f"{i+1} " + str(result).replace(" ", ""))

    # ------------------------------------------------------------------
    # 4) scoring
    # ------------------------------------------------------------------
    if dataset != "large" or mode != "test":
        with open(truth_path, "r", encoding="utf-8") as truth_f, \
             open(result_file, "r", encoding="utf-8") as result_f:
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
