import os
import json
import pickle
import collections
import re
import csv
from nltk.tokenize import word_tokenize
try:
    import sentencepiece as spm
except Exception:
    spm = None
from torchtext.vocab import GloVe
from config import Config
import torch
import numpy as np


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

pat = re.compile(r"[\w]+|[.,!?;|]")


class Command_Corpus:
    @staticmethod
    def _load_commands(path):
        commands = []
        if not os.path.exists(path):
            return commands
        with open(path, "r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f, delimiter="\t", quotechar='"', escapechar='\\')
            for row in reader:
                if len(row) < 7:
                    continue
                commands.append(
                    {
                        "dataId": row[0],
                        "validUntil": row[1],
                        "securityLevel": row[2],
                        "title": row[3],
                        "reportTime": row[4],
                        "body": row[5],
                        "category": row[6],
                    }
                )
        return commands

    @staticmethod
    def _load_users(path):
        users = []
        if not os.path.exists(path):
            return users
        with open(path, "r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f, delimiter="\t", quotechar='"', escapechar='\\')
            for row in reader:
                if len(row) < 7:
                    continue
                users.append(
                    {
                        "userId": row[0],
                        "name": row[1],
                        "department": row[2],
                        "position": row[3],
                        "rank": row[4],
                        "unit": row[5],
                        "history": row[6],
                    }
                )
        return users
    
    @staticmethod
    def _load_behaviors(path):
        """behaviors.tsv 로드 (USER 추천 방식)
        columns: ImpressionID, CommandID, ReportTime, Impressions
        Impressions: userId1-label1 userId2-label2 ... (label=1이면 해당 user가 명령을 읽음)
        """
        behaviors = []
        if not os.path.exists(path):
            return behaviors
        with open(path, "r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f, delimiter="\t", quotechar='"', escapechar='\\')
            for row in reader:
                if len(row) < 4:
                    continue
                behaviors.append(
                    {
                        "ImpressionID": row[0],
                        "CommandID": row[1],
                        "ReportTime": row[2],
                        "Impressions": row[3],
                    }
                )
        return behaviors

    @staticmethod
    def _history_from_str(history_str):
        if history_str is None:
            return []
        return [h.strip() for h in history_str.split(" ") if h.strip()]

    @staticmethod
    def _build_useridx_to_graphrow(order, user_ID_dict):
        mapping = {}
        for i, uid in enumerate(order):
            uid = "" if uid is None else str(uid).strip()
            if uid == "":
                continue
            mapping[user_ID_dict.get(uid, 0)] = i
        return mapping

    @staticmethod
    def preprocess(config: Config):
        user_ID_file = 'user_ID-%s.json' % config.dataset
        report_ID_file = 'report_ID-%s.json' % config.dataset
        category_file = 'category-%s.json' % config.dataset
        vocabulary_file = 'vocabulary-' + str(config.word_threshold) + '-' + config.tokenizer + '-' + str(config.max_title_length) + '-' + str(config.max_content_length) + '-' + str(config.max_time_length) + '-' + config.dataset + '.json'
        word_embedding_file = 'word_embedding-' + str(config.word_threshold) + '-' + str(config.word_embedding_dim) + '-' + config.tokenizer + '-' + str(config.max_title_length) + '-' + str(config.max_content_length) + '-' + str(config.max_time_length) + '-' + config.dataset + '.pkl'
        user_history_graph_file = 'user_history_graph-' + str(config.max_history_num) + ('' if config.no_self_connection else '-self') + ('' if config.no_adjacent_normalization else '-normalize-' + config.gcn_normalization_type) + '-' + config.dataset + '.pkl'
        
        department_file = 'department-%s.json' % config.dataset
        position_file   = 'position-%s.json' % config.dataset
        rank_file       = 'rank-%s.json' % config.dataset
        unit_file       = 'unit-%s.json' % config.dataset
        
        preprocessed_data_files = [user_ID_file, report_ID_file, category_file, vocabulary_file, word_embedding_file, user_history_graph_file, department_file, position_file, rank_file, unit_file]

        if not all(list(map(os.path.exists, preprocessed_data_files))):
            user_ID_dict = {'<UNK>': 0}
            report_ID_dict = {'<PAD>': 0}
            category_dict = {}
            word_dict = {'<PAD>': 0, '<UNK>': 1, '<NUM>': 2}
            word_counter = collections.Counter()
            report_category_dict = {}

            department_dict = {'<UNK>': 0}
            position_dict = {'<UNK>': 0}
            rank_dict = {'<UNK>': 0}
            unit_dict = {'<UNK>': 0}

            # 1. user ID dictionay
            all_users = []
            for prefix in [config.train_root, config.dev_root, config.test_root]:
                all_users.extend(Command_Corpus._load_users(os.path.join(prefix, 'users.tsv')))
                for user in all_users:
                    user_id = user.get("userId", None)
                    if user_id is None:
                        continue
                    user_id = str(user_id).strip()
                    if user_id == "":
                        continue

                    if user_id not in user_ID_dict:
                        user_ID_dict[user_id] = len(user_ID_dict)

                    dept = str(user.get("department", "<UNK>")).strip() or "<UNK>"
                    pos  = str(user.get("position", "<UNK>")).strip()  or "<UNK>"
                    rnk  = str(user.get("rank", "<UNK>")).strip()      or "<UNK>"
                    unt  = str(user.get("unit", "<UNK>")).strip()      or "<UNK>"

                    if dept not in department_dict:
                        department_dict[dept] = len(department_dict)
                    if pos not in position_dict:
                        position_dict[pos] = len(position_dict)
                    if rnk not in rank_dict:
                        rank_dict[rnk] = len(rank_dict)
                    if unt not in unit_dict:
                        unit_dict[unt] = len(unit_dict)

            with open(user_ID_file, 'w', encoding='utf-8') as user_ID_f:
                json.dump(user_ID_dict, user_ID_f)
            with open(department_file, 'w', encoding='utf-8') as department_f:
                json.dump(department_dict, department_f)
            with open(position_file, 'w', encoding='utf-8') as position_f:
                json.dump(position_dict, position_f)
            with open(rank_file, 'w', encoding='utf-8') as rank_f:
                json.dump(rank_dict, rank_f)
            with open(unit_file, 'w', encoding='utf-8') as unit_f:
                json.dump(unit_dict, unit_f)

            # 2. report ID dictionay & report category dictionay
            for prefix in [config.train_root, config.dev_root, config.test_root]:
                reports = Command_Corpus._load_commands(os.path.join(prefix, 'commands.tsv'))
                for report in reports:
                    report_ID = report.get('dataId', None)
                    if report_ID is None:
                        continue
                    report_ID = str(report_ID).strip()
                    if report_ID == "":
                        continue

                    category = report.get('category', 'UNK')
                    category = str(category).strip() if category is not None else 'UNK'
                    if category == "":
                        category = "UNK"

                    title     = report.get('title', '')
                    content   = report.get('body', '')
                    time_str  = report.get('reportTime', '')

                    title = "" if title is None else str(title)
                    content = "" if content is None else str(content)
                    time_str = "" if time_str is None else str(time_str)

                    if report_ID not in report_ID_dict:
                        report_ID_dict[report_ID] = len(report_ID_dict)

                    if category not in category_dict:
                        category_dict[category] = len(category_dict)

                    report_category_dict[report_ID] = category_dict[category]
                    words = word_tokenize(title.lower())
                    for word in words:
                        if is_number(word):
                            word_counter['<NUM>'] += 1
                        else:
                            word_counter[word] += 1
                    words = word_tokenize(content.lower())
                    for word in words:
                        if is_number(word):
                            word_counter['<NUM>'] += 1
                        else:
                            word_counter[word] += 1
                    words = word_tokenize(time_str.lower())
                    for word in words:
                        if is_number(word):
                            word_counter['<NUM>'] += 1
                        else:
                            word_counter[word] += 1

            with open(report_ID_file, 'w', encoding='utf-8') as report_ID_f:
                json.dump(report_ID_dict, report_ID_f, ensure_ascii=False)
            with open(category_file, 'w', encoding='utf-8') as category_f:
                json.dump(category_dict, category_f, ensure_ascii=False)

            # 3. word dictionay
            word_counter_list = [[word, word_counter[word]] for word in word_counter]
            word_counter_list.sort(key=lambda x: x[1], reverse=True) # sort by word frequency
            filtered_word_counter_list = list(filter(lambda x: x[1] >= config.word_threshold, word_counter_list))

            start_index = len(word_dict)
            add_i = 0

            for w, _cnt in filtered_word_counter_list:
                if w in word_dict:
                    continue
                word_dict[w] = start_index + add_i
                add_i += 1
            with open(vocabulary_file, 'w', encoding='utf-8') as vocabulary_f:
                json.dump(word_dict, vocabulary_f, ensure_ascii=False)

            # If SentencePiece tokenizer is requested, train a SentencePiece model
            if config.tokenizer == 'SentencePiece':
                if spm is None:
                    print('Warning: sentencepiece is not installed. Install it via `pip install sentencepiece` to use SentencePiece tokenizer.')
                else:
                    corpus_path = f'spm_corpus_{config.dataset}.txt'
                    with open(corpus_path, 'w', encoding='utf-8') as corpus_f:
                        for prefix in [config.train_root, config.dev_root, config.test_root]:
                            reports = Command_Corpus._load_commands(os.path.join(prefix, 'commands.tsv'))
                            for report in reports:
                                t = report.get('title', '') or ''
                                b = report.get('body', '') or ''
                                ti = report.get('reportTime', '') or ''
                                corpus_f.write((str(t) + '\n' + str(b) + '\n' + str(ti) + '\n'))

                        model_prefix = f'spm_{config.dataset}'
                        # choose a reasonable requested vocab size
                        requested_vocab = 8000
                        # compute unique whitespace tokens in corpus as an upper bound
                        unique_tokens = set()
                        try:
                            with open(corpus_path, 'r', encoding='utf-8') as cf:
                                for line in cf:
                                    for tok in line.strip().split():
                                        if tok:
                                            unique_tokens.add(tok)
                        except Exception:
                            unique_tokens = set()

                        max_vocab_from_corpus = max(100, len(unique_tokens))
                        # cap vocab to safe upper bound to avoid SentencePiece internal limits
                        safe_cap = 1600
                        final_vocab = min(requested_vocab, max_vocab_from_corpus, safe_cap)
                        if final_vocab < requested_vocab:
                            print(f'Adjusting SentencePiece vocab_size from {requested_vocab} to {final_vocab} based on corpus tokens ({len(unique_tokens)} unique tokens) and safe cap {safe_cap}.')
                        spm.SentencePieceTrainer.Train(f"--input={corpus_path} --model_prefix={model_prefix} --vocab_size={final_vocab} --model_type=unigram --character_coverage=0.9995")
                        print(f'SentencePiece model trained: {model_prefix}.model ({final_vocab} vocab)')

            # 4. Glove word embedding (branch for SentencePiece)
            if config.tokenizer == 'SentencePiece' and spm is not None:
                # Load trained SentencePiece model and build vocabulary from its pieces
                model_prefix = f'spm_{config.dataset}'
                sp_model_file = model_prefix + '.model'
                try:
                    sp_proc = spm.SentencePieceProcessor()
                    sp_proc.Load(sp_model_file)
                except Exception:
                    sp_proc = None

                # Rebuild word_dict from pieces (keep special tokens)
                if sp_proc is not None:
                    pieces = []
                    psz = None
                    try:
                        psz = sp_proc.get_piece_size()
                    except Exception:
                        try:
                            psz = sp_proc.GetPieceSize()
                        except Exception:
                            psz = None
                    if psz is not None:
                        for i in range(psz):
                            p = None
                            try:
                                p = sp_proc.id_to_piece(i)
                            except Exception:
                                try:
                                    p = sp_proc.IdToPiece(i)
                                except Exception:
                                    p = None
                            if p is not None:
                                pieces.append(p)
                    else:
                        # fallback: encode corpus and collect unique pieces
                        pieces_set = set()
                        with open(f'spm_corpus_{config.dataset}.txt', 'r', encoding='utf-8') as cf:
                            for line in cf:
                                pieces_set.update(sp_proc.encode(line.strip(), out_type=str))
                        pieces = sorted(pieces_set)

                    # rebuild word_dict with pieces appended after special tokens
                    word_dict = {'<PAD>': 0, '<UNK>': 1, '<NUM>': 2}
                    for p in pieces:
                        if p not in word_dict:
                            word_dict[p] = len(word_dict)

                    # save updated vocabulary_file
                    with open(vocabulary_file, 'w', encoding='utf-8') as vocabulary_f:
                        json.dump(word_dict, vocabulary_f, ensure_ascii=False)

                # Build embedding matrix using GloVe when possible; for pieces, map by stripping ▁
                if config.word_embedding_dim == 300:
                    glove = GloVe(name='840B', dim=300, cache='../glove', max_vectors=10000000000)
                else:
                    glove = GloVe(name='6B', dim=config.word_embedding_dim, cache='../glove', max_vectors=10000000000)
                glove_stoi = glove.stoi
                glove_vectors = glove.vectors
                glove_mean_vector = torch.mean(glove_vectors, dim=0, keepdim=False)
                word_embedding_vectors = torch.zeros([len(word_dict), config.word_embedding_dim])
                for piece, index in word_dict.items():
                    if index == 0:
                        continue
                    mapped = None
                    # try direct match
                    if piece in glove_stoi:
                        mapped = glove_vectors[glove_stoi[piece]]
                    else:
                        # strip leading underline used by SentencePiece for word-start
                        stripped = piece.lstrip('▁')
                        if stripped in glove_stoi:
                            mapped = glove_vectors[glove_stoi[stripped]]
                    if mapped is not None:
                        word_embedding_vectors[index, :] = mapped
                    else:
                        rv = torch.zeros(config.word_embedding_dim)
                        rv.normal_(mean=0, std=0.1)
                        word_embedding_vectors[index, :] = rv + glove_mean_vector
                with open(word_embedding_file, 'wb') as word_embedding_f:
                    pickle.dump(word_embedding_vectors, word_embedding_f)
            else:
                # Default behavior: word-based vocabulary -> GloVe mapping
                if config.word_embedding_dim == 300:
                    glove = GloVe(name='840B', dim=300, cache='../glove', max_vectors=10000000000)
                else:
                    glove = GloVe(name='6B', dim=config.word_embedding_dim, cache='../glove', max_vectors=10000000000)
                glove_stoi = glove.stoi
                glove_vectors = glove.vectors
                glove_mean_vector = torch.mean(glove_vectors, dim=0, keepdim=False)
                word_embedding_vectors = torch.zeros([len(word_dict), config.word_embedding_dim])
                for word in word_dict:
                    index = word_dict[word]
                    if index != 0:
                        if word in glove_stoi:
                            word_embedding_vectors[index, :] = glove_vectors[glove_stoi[word]]
                        else:
                            random_vector = torch.zeros(config.word_embedding_dim)
                            random_vector.normal_(mean=0, std=0.1)
                            word_embedding_vectors[index, :] = random_vector + glove_mean_vector
                with open(word_embedding_file, 'wb') as word_embedding_f:
                    pickle.dump(word_embedding_vectors, word_embedding_f)

            # 5. user history graph
            category_num = len(category_dict)
            graph_size = config.max_history_num + category_num # graph size of |V_{n}|+|V_{p}|
            prefix_mode = ['train', 'dev', 'test']
            user_history_graph_data = {}
            user_history_orders = {}
            for prefix_index, prefix in enumerate([config.train_root, config.dev_root, config.test_root]):
                mode = prefix_mode[prefix_index]

                users = Command_Corpus._load_users(os.path.join(prefix, 'users.tsv'))
                user_history_items = []
                for user in users:
                    uid = str(user.get("userId", "")).strip()
                    history_list = Command_Corpus._history_from_str(user.get("history", ""))
                    if uid:
                        user_history_items.append((uid, history_list))
                user_history_orders[mode] = [uid for uid, _ in user_history_items]
                user_history_num = len(user_history_items)

                user_history_graph = np.zeros([user_history_num, graph_size, graph_size], dtype=np.float32)
                user_history_category_mask = np.zeros([user_history_num, category_num + 1], dtype=bool)
                user_history_category_indices = np.zeros([user_history_num, config.max_history_num], dtype=np.int64)
                
                for line_index, (user_id, history_list) in enumerate(user_history_items):

                    if not isinstance(history_list, list):
                        continue

                    if config.no_self_connection:
                        history_graph = np.zeros([graph_size, graph_size], dtype=np.float32)
                    else:
                        history_graph = np.identity(graph_size, dtype=np.float32)
                    history_category_mask = np.zeros(category_num + 1, dtype=bool) # extra one category index for padding news
                    history_category_indices = np.full([config.max_history_num], category_num, dtype=np.int64)
                    if history_list and len(history_list) > 0:
                        history_report_ID = history_list
                        offset = max(0, len(history_report_ID) - config.max_history_num)
                        history_report_num = min(len(history_report_ID), config.max_history_num)
                        for i in range(history_report_num):
                            rid = history_report_ID[i + offset]
                            rid = str(rid).strip() if rid is not None else ""
                            if rid == "" or rid not in report_category_dict:
                                continue
                            category_index = report_category_dict[rid]
                            history_category_mask[category_index] = 1
                            history_category_indices[i] = category_index
                            history_graph[i, config.max_history_num + category_index] = 1 # edge of E_{p}^{1} in inter-cluster graph G2
                            history_graph[config.max_history_num + category_index, i] = 1 # edge of E_{p}^{1} in inter-cluster graph G2
                            for j in range(i + 1, history_report_num):
                                rid2 = history_report_ID[j + offset]
                                rid2 = str(rid2).strip() if rid2 is not None else ""
                                if rid2 == "" or rid2 not in report_category_dict:
                                    continue
                                _category_index = report_category_dict[rid2]
                                if category_index == _category_index:
                                    history_graph[i, j] = 1 # edge of E_{n} in intra-cluster graph G1
                                    history_graph[j, i] = 1 # edge of E_{n} in intra-cluster graph G1
                                else:
                                    history_graph[config.max_history_num + category_index, config.max_history_num + _category_index] = 1 # edge of E_{p}^{2} in inter-cluster graph G2
                                    history_graph[config.max_history_num + _category_index, config.max_history_num + category_index] = 1 # edge of E_{p}^{2} in inter-cluster graph G2
                        if not config.no_adjacent_normalization:
                            deg = history_graph.sum(axis=1, keepdims=False)
                            deg[deg == 0] = 1
                            if config.gcn_normalization_type == 'asymmetric':
                                # Asymmetric adjacent matrix normalization: D^{-1}A
                                D_inv = np.zeros([graph_size, graph_size], dtype=np.float32)
                                np.fill_diagonal(D_inv, 1 / deg)
                                history_graph = np.matmul(D_inv, history_graph)
                            else:
                                # Symmetric adjacent matrix normalization: D^{-\frac{1}{2}}AD^{-\frac{1}{2}}
                                D_inv_sqrt = np.zeros([graph_size, graph_size], dtype=np.float32)
                                np.fill_diagonal(D_inv_sqrt, np.sqrt(1 / deg))
                                history_graph = np.matmul(np.matmul(D_inv_sqrt, history_graph), D_inv_sqrt)
                    user_history_graph[line_index] = history_graph
                    user_history_category_mask[line_index] = history_category_mask
                    user_history_category_indices[line_index] = history_category_indices
                user_history_graph_data[mode + '_user_history_graph'] = user_history_graph
                user_history_graph_data[mode + '_user_history_category_mask'] = user_history_category_mask
                user_history_graph_data[mode + '_user_history_category_indices'] = user_history_category_indices
            user_history_graph_data['user_history_orders'] = user_history_orders
            with open(user_history_graph_file, 'wb') as user_history_graph_f:
                pickle.dump(user_history_graph_data, user_history_graph_f)

    def __init__(self, config: Config):
        # preprocess data
        Command_Corpus.preprocess(config)
        with open('user_ID-%s.json' % config.dataset, 'r', encoding='utf-8') as user_ID_f:
            self.user_ID_dict = json.load(user_ID_f)
            config.user_num = len(self.user_ID_dict)
        with open('report_ID-%s.json' % config.dataset, 'r', encoding='utf-8') as report_ID_f:
            self.report_ID_dict = json.load(report_ID_f)
            self.report_num = len(self.report_ID_dict)
        with open('category-%s.json' % config.dataset, 'r', encoding='utf-8') as category_f:
            self.category_dict = json.load(category_f)
            config.category_num = len(self.category_dict)
        with open('vocabulary-' + str(config.word_threshold) + '-' + config.tokenizer + '-' + str(config.max_title_length) + '-' + str(config.max_content_length) + '-' + str(config.max_time_length) + '-' + config.dataset + '.json', 'r', encoding='utf-8') as vocabulary_f:
            self.word_dict = json.load(vocabulary_f)
            config.vocabulary_size = len(self.word_dict)
        
        with open('department-%s.json' % config.dataset, 'r', encoding='utf-8') as f:
            self.department_dict = json.load(f)
            config.department_num = len(self.department_dict)
        with open('position-%s.json' % config.dataset, 'r', encoding='utf-8') as f:
            self.position_dict = json.load(f)
            config.position_num = len(self.position_dict)
        with open('rank-%s.json' % config.dataset, 'r', encoding='utf-8') as f:
            self.rank_dict = json.load(f)
            config.rank_num = len(self.rank_dict)
        with open('unit-%s.json' % config.dataset, 'r', encoding='utf-8') as f:
            self.unit_dict = json.load(f)
            config.unit_num = len(self.unit_dict)
        
        with open('user_history_graph-' + str(config.max_history_num) + ('' if config.no_self_connection else '-self') + ('' if config.no_adjacent_normalization else '-normalize-' + config.gcn_normalization_type) + '-' + config.dataset + '.pkl', 'rb') as user_history_graph_f:
            user_history_data = pickle.load(user_history_graph_f)
            self.train_user_history_graph = user_history_data['train_user_history_graph']
            self.train_user_history_category_mask = user_history_data['train_user_history_category_mask']
            self.train_user_history_category_indices = user_history_data['train_user_history_category_indices']
            self.dev_user_history_graph = user_history_data['dev_user_history_graph']
            self.dev_user_history_category_mask = user_history_data['dev_user_history_category_mask']
            self.dev_user_history_category_indices = user_history_data['dev_user_history_category_indices']
            self.test_user_history_graph = user_history_data['test_user_history_graph']
            self.test_user_history_category_mask = user_history_data['test_user_history_category_mask']
            self.test_user_history_category_indices = user_history_data['test_user_history_category_indices']

        # meta data
        self.negative_sample_num = config.negative_sample_num                                           # negative sample number for training
        self.max_history_num = config.max_history_num                                                   # max history number for each training user
        self.max_title_length = config.max_title_length                                                 # max title length for each news text
        self.max_content_length = config.max_content_length                                           # max content length for each news text
        self.max_time_length = config.max_time_length
        
        self.report_category = np.zeros([self.report_num], dtype=np.int32)                                  # [report_num]
        self.report_title_text = np.zeros([self.report_num, self.max_title_length], dtype=np.int32)         # [report_num, max_title_length]
        self.report_title_mask = np.zeros([self.report_num, self.max_title_length], dtype=bool)             # [report_num, max_title_length]
        self.report_content_text = np.zeros([self.report_num, self.max_content_length], dtype=np.int32)   # [report_num, max_content_length]
        self.report_content_mask = np.zeros([self.report_num, self.max_content_length], dtype=bool)       # [report_num, max_content_length]
        self.report_time_text = np.zeros([self.report_num, self.max_time_length], dtype=np.int32)
        self.report_time_mask = np.zeros([self.report_num, self.max_time_length], dtype=bool)
        self.report_valid_until = np.zeros([self.report_num], dtype=np.int32)
        self.report_security_level = np.zeros([self.report_num], dtype=np.int32)
        
        self.train_userDataset = []                                                                       # [user_ID, [history], [history_mask], click impression, [non-click impressions], behavior_index]
        self.dev_userDataset = []                                                                         # [user_ID, [history], [history_mask], candidate_news_ID, behavior_index]
        self.dev_indices = []                                                                            # index for dev
        self.test_userDataset = []                                                                        # [user_ID, [history], [history_mask], candidate_news_ID, behavior_index]
        self.test_indices = []
        
        self.user_department = np.zeros([config.user_num], dtype=np.int32)
        self.user_position  = np.zeros([config.user_num], dtype=np.int32)
        self.user_rank      = np.zeros([config.user_num], dtype=np.int32)
        self.user_unit      = np.zeros([config.user_num], dtype=np.int32)
        
        for prefix in [config.train_root, config.dev_root, config.test_root]:
            for user in Command_Corpus._load_users(os.path.join(prefix, 'users.tsv')):
                user_id = user.get("userId", None)
                if user_id is None:
                    continue
                user_id = str(user_id).strip()
                if user_id == "" or user_id not in self.user_ID_dict:
                    continue

                uidx = self.user_ID_dict[user_id]

                dept = str(user.get("department", "<UNK>")).strip() or "<UNK>"
                pos  = str(user.get("position", "<UNK>")).strip()  or "<UNK>"
                rnk  = str(user.get("rank", "<UNK>")).strip()      or "<UNK>"
                unt  = str(user.get("unit", "<UNK>")).strip()      or "<UNK>"

                self.user_department[uidx] = self.department_dict.get(dept, 0)
                self.user_position[uidx]   = self.position_dict.get(pos, 0)
                self.user_rank[uidx]       = self.rank_dict.get(rnk, 0)
                self.user_unit[uidx]       = self.unit_dict.get(unt, 0)

        self.title_word_num = 0
        self.content_word_num = 0
        self.time_word_num = 0

        # generate news meta data
        report_ID_set  = set(['<PAD>'])
        report_items = []
        for prefix in [config.train_root, config.dev_root, config.test_root]:
            reports = Command_Corpus._load_commands(os.path.join(prefix, 'commands.tsv'))
            for report in reports:
                report_ID = report.get('dataId', None)
                if report_ID is None:
                    continue
                report_ID = str(report_ID).strip()
                if report_ID == "":
                    continue

                if report_ID not in report_ID_set:
                    report_items.append(report)
                    report_ID_set.add(report_ID)

        assert self.report_num == len(report_ID_set), 'report num mismatch %d v.s. %d' % (self.report_num, len(report_ID_set))

        UNK_IDX = self.word_dict.get('<UNK>', 1)
        NUM_IDX = self.word_dict.get('<NUM>', 2)

        # Prepare SentencePiece processor if available and requested
        sp_proc = None
        if config.tokenizer == 'SentencePiece' and spm is not None:
            sp_model_file = f'spm_{config.dataset}.model'
            if os.path.exists(sp_model_file):
                try:
                    sp_proc = spm.SentencePieceProcessor()
                    sp_proc.Load(sp_model_file)
                except Exception:
                    sp_proc = None

        for report in report_items:
            report_ID = report.get('dataId', None)
            if report_ID is None:
                continue
            report_ID = str(report_ID).strip()
            if report_ID == "":
                continue
            if report_ID not in self.report_ID_dict:
                continue

            category = report.get('category', 'UNK')
            category = str(category).strip() if category is not None else 'UNK'
            if category == "":
                category = 'UNK'

            title = report.get('title', '')
            content = report.get('body', '')
            time_str = report.get('reportTime', '')

            title = "" if title is None else str(title)
            content = "" if content is None else str(content)
            time_str = "" if time_str is None else str(time_str)
            valid_until = report.get('validUntil', None)
            security_lv = report.get('securityLevel', None)

            index = self.report_ID_dict[report_ID]
            self.report_category[index] = self.category_dict.get(category, 0)
            
            if valid_until is not None:
                try:
                    self.report_valid_until[index] = int(valid_until)
                except Exception:
                    pass
            if security_lv is not None:
                try:
                    self.report_security_level[index] = int(security_lv)
                except Exception:
                    pass
            # title
            if sp_proc is not None:
                words = sp_proc.encode(title.lower(), out_type=str)
            else:
                words = word_tokenize(title.lower())
            for j, word in enumerate(words):
                if j >= self.max_title_length:
                    break
                if is_number(word):
                    self.report_title_text[index][j] = NUM_IDX
                else:
                    self.report_title_text[index][j] = self.word_dict.get(word, UNK_IDX)
                self.report_title_mask[index][j] = True
            self.title_word_num += len(words)
            # content
            if sp_proc is not None:
                words = sp_proc.encode(content.lower(), out_type=str)
            else:
                words = word_tokenize(content.lower())
            for j, word in enumerate(words):
                if j >= self.max_content_length:
                    break
                if is_number(word):
                    self.report_content_text[index][j] = NUM_IDX
                else:
                    self.report_content_text[index][j] = self.word_dict.get(word, UNK_IDX)
                self.report_content_mask[index][j] = True
            self.content_word_num += len(words)
            # time
            if sp_proc is not None:
                words = sp_proc.encode(time_str.lower(), out_type=str)
            else:
                words = word_tokenize(time_str.lower())
            for j, word in enumerate(words):
                if j >= self.max_time_length:
                    break
                if is_number(word):
                    self.report_time_text[index][j] = NUM_IDX
                else:
                    self.report_time_text[index][j] = self.word_dict.get(word, UNK_IDX)
                self.report_time_mask[index][j] = True
            self.time_word_num += len(words)

        self.report_title_mask[0][0] = True    # for <PAD> report
        self.report_content_mask[0][0] = True  # for <PAD> report
        self.report_time_mask[0][0] = True     # for <PAD> report

        # generate userDataset meta data
        train_history = {u.get("userId", ""): Command_Corpus._history_from_str(u.get("history", "")) for u in Command_Corpus._load_users(os.path.join(config.train_root, 'users.tsv'))}
        dev_history = {u.get("userId", ""): Command_Corpus._history_from_str(u.get("history", "")) for u in Command_Corpus._load_users(os.path.join(config.dev_root, 'users.tsv'))}
        test_history = {u.get("userId", ""): Command_Corpus._history_from_str(u.get("history", "")) for u in Command_Corpus._load_users(os.path.join(config.test_root, 'users.tsv'))}

        def build_samples(history_dict, dataset_store, indices_store=None, impressions_list=None):
            """USER 추천 방식으로 샘플 생성
            history_dict: 각 user의 command 히스토리 (user → [commands])
            impressions_list: behaviors.tsv에서 로드한 command별 user 추천 정보
            """
            sample_idx = 0

            if impressions_list is not None:
                # dev/test: behaviors.tsv로부터 command별 user 추천 정보 사용
                for impression_id, behavior in enumerate(impressions_list, start=1):
                    cmd_id = behavior.get("CommandID", "").strip()
                    cmd_idx = self.report_ID_dict.get(cmd_id, 0)
                    if cmd_idx == 0:
                        continue

                    impressions_str = behavior.get("Impressions", "")
                    if not impressions_str:
                        continue

                    impressions = impressions_str.split()
                    
                    for pos, impression in enumerate(impressions):
                        impression = impression.strip()
                        if not impression:
                            continue

                        parts = impression.rsplit('-', 1)
                        if len(parts) != 2:
                            continue
                        
                        user_id, label = parts[0], parts[1]
                        user_id = str(user_id).strip() if user_id else ""
                        user_idx = self.user_ID_dict.get(user_id, 0)
                        if user_idx == 0 or user_id not in history_dict:
                            continue

                        label_int = 1 if label == '1' else 0
                        
                        # User의 히스토리 가져오기
                        user_history_list = history_dict.get(user_id, [])
                        mapped_history = []
                        for rid in user_history_list:
                            rid = "" if rid is None else str(rid).strip()
                            mapped_history.append(self.report_ID_dict.get(rid, 0))

                        # history padding
                        hist_len = len(mapped_history)
                        if hist_len > self.max_history_num:
                            user_history = mapped_history[-self.max_history_num:]
                            hist_len = self.max_history_num
                        else:
                            user_history = mapped_history + [0] * (self.max_history_num - hist_len)

                        user_history_mask = np.zeros([self.max_history_num], dtype=bool)
                        user_history_mask[:hist_len] = True

                        # dataset 저장
                        dataset_store.append([user_idx, user_history[:], user_history_mask.copy(),
                                            cmd_idx, [], sample_idx])

                        # indices_store에 label 정보 저장
                        if indices_store is not None:
                            indices_store[sample_idx] = (impression_id, pos, label_int)

                        sample_idx += 1
            else:
                # training: history progression 방식으로 샘플 생성
                for user_id, history_list in history_dict.items():
                    user_id = "" if user_id is None else str(user_id).strip()
                    if user_id == "":
                        continue
                    user_idx = self.user_ID_dict.get(user_id, 0)

                    if not isinstance(history_list, list) or len(history_list) < 1:
                        continue

                    mapped_history = []
                    for rid in history_list:
                        rid = "" if rid is None else str(rid).strip()
                        mapped_history.append(self.report_ID_dict.get(rid, 0))

                    # history padding
                    hist_len = len(mapped_history)
                    if hist_len > self.max_history_num:
                        user_history = mapped_history[-self.max_history_num:]
                        hist_len = self.max_history_num
                    else:
                        user_history = mapped_history + [0] * (self.max_history_num - hist_len)

                    user_history_mask = np.zeros([self.max_history_num], dtype=bool)
                    user_history_mask[:hist_len] = True

                    # history progression: user가 읽은 각 command를 학습 샘플로 사용
                    for t in range(1, len(mapped_history)):
                        full_history = mapped_history[:t]
                        if len(full_history) > self.max_history_num:
                            full_history = full_history[-self.max_history_num:]

                        hist_len = len(full_history)
                        padding_num = self.max_history_num - hist_len
                        user_history = full_history + [0] * padding_num

                        user_history_mask = np.zeros([self.max_history_num], dtype=bool)
                        user_history_mask[:hist_len] = True

                        pos_command = mapped_history[t]  # 다음에 읽을 command (양성 샘플)
                        if pos_command == 0:
                            continue

                        neg_commands = []

                        dataset_store.append([user_idx, user_history, user_history_mask.copy(), pos_command, neg_commands, sample_idx])
                        if indices_store is not None:
                            indices_store.append(sample_idx)
                        sample_idx += 1
        
        # Load behaviors for dev/test (USER 추천 방식)
        dev_behaviors = Command_Corpus._load_behaviors(os.path.join(config.dev_root, 'behaviors.tsv'))
        test_behaviors = Command_Corpus._load_behaviors(os.path.join(config.test_root, 'behaviors.tsv'))
        
        build_samples(train_history, self.train_userDataset)
        self.dev_indices = {}  # dict: sample_idx -> (impression_id, pos, label)
        build_samples(dev_history, self.dev_userDataset, self.dev_indices, dev_behaviors)
        self.test_indices = {}  # dict: sample_idx -> (impression_id, pos, label)
        build_samples(test_history, self.test_userDataset, self.test_indices, test_behaviors)

        self.train_useridx_to_graphrow = Command_Corpus._build_useridx_to_graphrow(
            user_history_data.get('user_history_orders', {}).get('train', []), self.user_ID_dict
        )
        self.dev_useridx_to_graphrow = Command_Corpus._build_useridx_to_graphrow(
            user_history_data.get('user_history_orders', {}).get('dev', []), self.user_ID_dict
        )
        self.test_useridx_to_graphrow = Command_Corpus._build_useridx_to_graphrow(
            user_history_data.get('user_history_orders', {}).get('test', []), self.user_ID_dict
        )