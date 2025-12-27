from Command_corpus import Command_Corpus
import time
import os
import json
from config import Config
import torch.utils.data as data
from numpy.random import randint
from torch.utils.data import DataLoader


class Command_Train_Dataset(data.Dataset):
    def __init__(self, corpus: Command_Corpus, config: Config):
        self.config = config
        self.negative_sample_num = corpus.negative_sample_num
        self.user_num = corpus.user_num

        # report (command)
        self.report_title_text = corpus.report_title_text
        self.report_title_mask = corpus.report_title_mask
        self.report_content_text = corpus.report_content_text
        self.report_content_mask = corpus.report_content_mask
        self.report_time_text = corpus.report_time_text
        self.report_time_mask = corpus.report_time_mask
        self.report_category = corpus.report_category

        # user meta
        self.user_department = corpus.user_department
        self.user_position = corpus.user_position
        self.user_rank = corpus.user_rank
        self.user_unit = corpus.user_unit

        # graph
        self.user_history_graph = corpus.train_user_history_graph
        self.user_history_category_mask = corpus.train_user_history_category_mask
        self.user_history_category_indices = corpus.train_user_history_category_indices
        self.useridx_to_graphrow = corpus.train_useridx_to_graphrow

        self.train_userDataset = corpus.train_userDataset
        self.num = len(self.train_userDataset)

        # [pos_user + neg_users]
        self.train_samples = [[0] * (1 + self.negative_sample_num) for _ in range(self.num)]

    def negative_sampling(self):
        print(f"\nBegin negative sampling, training sample num : {self.num}")
        start_time = time.time()

        for i, sample in enumerate(self.train_userDataset):
            pos_user = sample[0]
            neg_candidates = sample[4]

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

    def __getitem__(self, index):
        sample = self.train_userDataset[index]

        user_idx = sample[0]
        history_index = sample[1]
        history_mask = sample[2]
        cmd_idx = sample[3]                 
        sampled_users = self.train_samples[index]

        graph_row = self.useridx_to_graphrow.get(user_idx, 0)
        

        return (
            # target users
            sampled_users,

            # user attributes
            self.user_department[sampled_users],
            self.user_position[sampled_users],
            self.user_rank[sampled_users],
            self.user_unit[sampled_users],

            # user history
            self.report_title_text[history_index],
            self.report_content_text[history_index],
            self.report_time_text[history_index],
            history_mask,
            self.user_history_graph[graph_row],
            self.user_history_category_mask[graph_row],
            self.user_history_category_indices[graph_row],

            # command (same for all candidates)
            self.report_title_text[cmd_idx],
            self.report_title_mask[cmd_idx],
            self.report_content_text[cmd_idx],
            self.report_content_mask[cmd_idx],
            self.report_time_text[cmd_idx],
            self.report_time_mask[cmd_idx],
            self.report_category[cmd_idx]
        )

    def __len__(self):
        return self.num


class Command_DevTest_Dataset(data.Dataset):
    def __init__(self, corpus: Command_Corpus, config: Config, mode: str):
        assert mode in ['dev', 'test'], 'mode must be chosen from \'dev\' or \'test\''
        self.config = config
        self.mode = mode

        self.report_title_text = corpus.report_title_text
        self.report_title_mask = corpus.report_title_mask
        self.report_content_text = corpus.report_content_text
        self.report_content_mask = corpus.report_content_mask
        self.report_time_text = corpus.report_time_text
        self.report_time_mask = corpus.report_time_mask
        self.report_category = corpus.report_category

        self.user_department = corpus.user_department
        self.user_position = corpus.user_position
        self.user_rank = corpus.user_rank
        self.user_unit = corpus.user_unit

        if mode == 'dev':
            self.user_history_graph = corpus.dev_user_history_graph
            self.user_history_category_mask = corpus.dev_user_history_category_mask
            self.user_history_category_indices = corpus.dev_user_history_category_indices
            self.dataset = corpus.dev_userDataset
            self.useridx_to_graphrow = corpus.dev_useridx_to_graphrow
        else:
            self.user_history_graph = corpus.test_user_history_graph
            self.user_history_category_mask = corpus.test_user_history_category_mask
            self.user_history_category_indices = corpus.test_user_history_category_indices
            self.dataset = corpus.test_userDataset
            self.useridx_to_graphrow = corpus.test_useridx_to_graphrow

        self.num = len(self.dataset)


    def __getitem__(self, index):
        sample = self.dataset[index]
        user_idx = sample[0]
        history_index = sample[1]
        history_mask = sample[2]
        candidate_index = sample[3]  # cmd_idx (report index)
        group_id = sample[4]         # dev_ID or test_ID (grouping용)

        graph_row_index = self.useridx_to_graphrow.get(user_idx, 0)

        user_dept = self.user_department[user_idx]
        user_pos  = self.user_position[user_idx]
        user_rank = self.user_rank[user_idx]
        user_unit = self.user_unit[user_idx]

        return (
            user_idx, user_dept, user_pos, user_rank, user_unit,
            self.report_title_text[history_index],
            self.report_content_text[history_index],
            self.report_time_text[history_index],
            history_mask,
            self.user_history_graph[graph_row_index],
            self.user_history_category_mask[graph_row_index],
            self.user_history_category_indices[graph_row_index],
            self.report_title_text[candidate_index],
            self.report_title_mask[candidate_index],
            self.report_content_text[candidate_index],
            self.report_content_mask[candidate_index],
            self.report_time_text[candidate_index],
            self.report_time_mask[candidate_index],
            self.report_category[candidate_index],
            group_id
        )

    def __len__(self):
        return self.num


if __name__ == '__main__':
    start_time = time.time()
    config = Config()
    command_corpus = Command_Corpus(config)

    print('user_num :', len(command_corpus.user_ID_dict))
    print('report_num :', command_corpus.report_num)

    command_train_dataset = Command_Train_Dataset(command_corpus, config)
    command_dev_dataset = Command_DevTest_Dataset(command_corpus, config, 'dev')
    command_test_dataset = Command_DevTest_Dataset(command_corpus, config, 'test')

    command_train_dataset.negative_sampling()

    end_time = time.time()
    print('load time : %.3fs' % (end_time - start_time))
    print('Command_Train_Dataset :', len(command_train_dataset))
    print('Command_Dev_Dataset :', len(command_dev_dataset))
    print('Command_Test_Dataset :', len(command_test_dataset))

    train_dataloader = DataLoader(command_train_dataset, batch_size=config.batch_size, shuffle=True,
                                  num_workers=max(0, config.batch_size // 16))
    dev_dataloader = DataLoader(command_dev_dataset, batch_size=config.batch_size, shuffle=False,
                                num_workers=max(0, config.batch_size // 16))
    test_dataloader = DataLoader(command_test_dataset, batch_size=config.batch_size, shuffle=False,
                                 num_workers=max(0, config.batch_size // 16))