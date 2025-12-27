from Command_corpus import Command_Corpus
import time
import numpy as np
from config import Config
import torch.utils.data as data
from numpy.random import randint


# =========================================================
# Train Dataset (Command → User)
# =========================================================
class Command_Train_Dataset(data.Dataset):
    def __init__(self, corpus: Command_Corpus, config: Config):
        self.config = config
        self.negative_sample_num = corpus.negative_sample_num
        self.user_num = corpus.user_num

        # ---------- command (report) ----------
        self.report_title_text   = corpus.report_title_text
        self.report_title_mask   = corpus.report_title_mask
        self.report_content_text = corpus.report_content_text
        self.report_content_mask = corpus.report_content_mask
        self.report_time_text    = corpus.report_time_text
        self.report_time_mask    = corpus.report_time_mask
        self.report_category     = corpus.report_category

        # ---------- user meta ----------
        self.user_department = corpus.user_department
        self.user_position   = corpus.user_position
        self.user_rank       = corpus.user_rank
        self.user_unit       = corpus.user_unit

        # ---------- user history graph ----------
        self.user_history_graph            = corpus.train_user_history_graph
        self.user_history_category_mask    = corpus.train_user_history_category_mask
        self.user_history_category_indices = corpus.train_user_history_category_indices
        self.useridx_to_graphrow           = corpus.train_useridx_to_graphrow

        # ---------- corpus samples ----------
        # sample = [cmd_idx, pos_user, neg_users, behavior_index]
        self.samples = corpus.train_userDataset
        self.num = len(self.samples)

        # sampled users (pos + neg)
        self.train_samples = [
            [0 for _ in range(1 + self.negative_sample_num)]
            for _ in range(self.num)
        ]

    # -----------------------------------------------------
    # Negative sampling (user dimension)
    # -----------------------------------------------------
    def negative_sampling(self):
        print(f"\nBegin negative sampling, training sample num : {self.num}")
        start_time = time.time()

        for i, sample in enumerate(self.samples):
            _, pos_user, neg_candidates, _ = sample

            self.train_samples[i][0] = pos_user

            if neg_candidates:
                cand_num = len(neg_candidates)
                for j in range(self.negative_sample_num):
                    self.train_samples[i][j + 1] = neg_candidates[j % cand_num]
            else:
                used = {pos_user, 0}
                for j in range(self.negative_sample_num):
                    while True:
                        uid = randint(1, self.user_num)
                        if uid not in used:
                            self.train_samples[i][j + 1] = uid
                            used.add(uid)
                            break

        print(f"End negative sampling, used time : {time.time() - start_time:.3f}s")

    # -----------------------------------------------------
    # Get item
    # -----------------------------------------------------
    def __getitem__(self, index):
        cmd_idx, pos_user, _, behavior_index = self.samples[index]

        users = np.asarray(self.train_samples[index], dtype=np.int64)
        graph_row = self.useridx_to_graphrow.get(pos_user, 0)

        return (
            # -------- user attributes (B = 1+neg) --------
            self.user_department[users],
            self.user_position[users],
            self.user_rank[users],
            self.user_unit[users],

            # -------- user history graph --------
            self.user_history_graph[graph_row],
            self.user_history_category_mask[graph_row],
            self.user_history_category_indices[graph_row],

            # -------- candidate command --------
            self.report_title_text[cmd_idx],
            self.report_title_mask[cmd_idx],
            self.report_content_text[cmd_idx],
            self.report_content_mask[cmd_idx],
            self.report_time_text[cmd_idx],
            self.report_time_mask[cmd_idx],
            self.report_category[cmd_idx],
        )

    def __len__(self):
        return self.num


# =========================================================
# Dev / Test Dataset (Command → User)
# =========================================================
class Command_DevTest_Dataset(data.Dataset):
    def __init__(self, corpus: Command_Corpus, config: Config, mode: str):
        assert mode in ['dev', 'test']
        self.mode = mode

        # ---------- command ----------
        self.report_title_text   = corpus.report_title_text
        self.report_title_mask   = corpus.report_title_mask
        self.report_content_text = corpus.report_content_text
        self.report_content_mask = corpus.report_content_mask
        self.report_time_text    = corpus.report_time_text
        self.report_time_mask    = corpus.report_time_mask
        self.report_category     = corpus.report_category

        # ---------- user meta ----------
        self.user_department = corpus.user_department
        self.user_position   = corpus.user_position
        self.user_rank       = corpus.user_rank
        self.user_unit       = corpus.user_unit

        if mode == 'dev':
            self.user_history_graph            = corpus.dev_user_history_graph
            self.user_history_category_mask    = corpus.dev_user_history_category_mask
            self.user_history_category_indices = corpus.dev_user_history_category_indices
            self.samples = corpus.dev_userDataset
            self.useridx_to_graphrow = corpus.dev_useridx_to_graphrow
        else:
            self.user_history_graph            = corpus.test_user_history_graph
            self.user_history_category_mask    = corpus.test_user_history_category_mask
            self.user_history_category_indices = corpus.test_user_history_category_indices
            self.samples = corpus.test_userDataset
            self.useridx_to_graphrow = corpus.test_useridx_to_graphrow

        self.num = len(self.samples)

    def __getitem__(self, index):
        # sample = [cmd_idx, user_idx, behavior_index]
        cmd_idx, user_idx, behavior_index = self.samples[index]
        graph_row = self.useridx_to_graphrow.get(user_idx, 0)

        return (
            user_idx,
            self.user_department[user_idx],
            self.user_position[user_idx],
            self.user_rank[user_idx],
            self.user_unit[user_idx],

            self.user_history_graph[graph_row],
            self.user_history_category_mask[graph_row],
            self.user_history_category_indices[graph_row],

            self.report_title_text[cmd_idx],
            self.report_title_mask[cmd_idx],
            self.report_content_text[cmd_idx],
            self.report_content_mask[cmd_idx],
            self.report_time_text[cmd_idx],
            self.report_time_mask[cmd_idx],
            self.report_category[cmd_idx],

            behavior_index
        )

    def __len__(self):
        return self.num
