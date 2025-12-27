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
        self.report_num = corpus.report_num

        self.report_title_text = corpus.report_title_text
        self.report_title_mask = corpus.report_title_mask
        self.report_time_text = corpus.report_time_text
        self.report_time_mask = corpus.report_time_mask
        self.report_content_text = corpus.report_content_text
        self.report_content_mask = corpus.report_content_mask
        self.report_category = corpus.report_category

        self.report_valid_until = corpus.report_valid_until
        self.report_security_level = corpus.report_security_level
        
        self.user_history_graph = corpus.train_user_history_graph
        self.user_history_category_mask = corpus.train_user_history_category_mask
        self.user_history_category_indices = corpus.train_user_history_category_indices

        self.user_department = corpus.user_department
        self.user_position = corpus.user_position
        self.user_rank = corpus.user_rank
        self.user_unit = corpus.user_unit

        self.train_userDataset = corpus.train_userDataset
        self.num = len(self.train_userDataset)
        self.train_samples = [[0 for _ in range(1 + self.negative_sample_num)] for __ in range(self.num)]

        self.useridx_to_graphrow = corpus.train_useridx_to_graphrow

    def negative_sampling(self, rank=None):
        print('\n%sBegin negative sampling, training sample num : %d' % ('' if rank is None else ('rank ' + str(rank) + ' : '), self.num))
        start_time = time.time()
        for i, sample in enumerate(self.train_userDataset):
            pos_report = sample[3]
            neg_candidates = sample[4]
            self.train_samples[i][0] = pos_report

            if neg_candidates and len(neg_candidates) > 0:
                report_num = len(neg_candidates)
                if report_num <= self.negative_sample_num:
                    for j in range(self.negative_sample_num):
                        self.train_samples[i][j + 1] = neg_candidates[j % report_num]
                else:
                    used = set()
                    for j in range(self.negative_sample_num):
                        while True:
                            k = randint(0, report_num)
                            if k not in used:
                                self.train_samples[i][j + 1] = neg_candidates[k]
                                used.add(k)
                                break
            else:
                used = {pos_report, 0}
                for j in range(self.negative_sample_num):
                    while True:
                        rid = randint(0, self.report_num)  # 0 ~ report_num-1
                        if rid not in used:
                            self.train_samples[i][j + 1] = rid
                            used.add(rid)
                            break

        end_time = time.time()
        print('%sEnd negative sampling, used time : %.3fs' % ('' if rank is None else ('rank ' + str(rank) + ' : '), end_time - start_time))

    def __getitem__(self, index):
        train_userDataset = self.train_userDataset[index]
        user_idx = train_userDataset[0]
        history_index = train_userDataset[1]     # [max_history_num] (report index list)
        history_mask = train_userDataset[2]      # [max_history_num] bool
        sample_index = self.train_samples[index] # [1 + neg_num]
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
            self.report_title_text[sample_index],
            self.report_title_mask[sample_index],
            self.report_content_text[sample_index],
            self.report_content_mask[sample_index],
            self.report_time_text[sample_index],
            self.report_time_mask[sample_index],
            self.report_category[sample_index]
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
        candidate_index = sample[3]
        sample_idx = sample[5]

        graph_row_index = self.useridx_to_graphrow.get(user_idx, 0)

        user_dept = self.user_department[user_idx]
        user_pos = self.user_position[user_idx]
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
            self.user_history_category_indices[graph_row_index],
            sample_idx
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

    train_dataloader = DataLoader(command_train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=max(0, config.batch_size // 16))
    dev_dataloader = DataLoader(command_dev_dataset, batch_size=config.batch_size, shuffle=False, num_workers=max(0, config.batch_size // 16))
    test_dataloader = DataLoader(command_test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=max(0, config.batch_size // 16))
    