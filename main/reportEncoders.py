import pickle
# from config import Config
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence
from layers import Conv1D, Conv2D_Pool, MultiHeadAttention, Attention, ScaledDotProduct_CandidateAttention, CandidateAttention


class ReportEncoder(nn.Module):
    #def __init__(self, config: Config):
    def __init__(self, config):
        super(ReportEncoder, self).__init__()
        
        self.word_embedding_dim = config.word_embedding_dim
        # 단어 임베딩 로드 (GloVe 기반)
        self.word_embedding = nn.Embedding(num_embeddings=config.vocabulary_size, embedding_dim=self.word_embedding_dim)
        with open('word_embedding-' + str(config.word_threshold) + '-' + str(config.word_embedding_dim) + '-' + config.tokenizer + '-' + str(config.max_title_length) + '-' + str(config.max_content_length) + '-' + str(config.max_time_length) + '-' + config.dataset + '.pkl', 'rb') as word_embedding_f:
            self.word_embedding.weight.data.copy_(pickle.load(word_embedding_f))
        # 카테고리 임베딩
        self.category_embedding  = nn.Embedding(num_embeddings=config.category_num + 1, embedding_dim=config.category_embedding_dim)
        self.dropout = nn.Dropout(p=config.dropout_rate, inplace=True)
        self.dropout_ = nn.Dropout(p=config.dropout_rate, inplace=False)
        self.auxiliary_loss = None
        '''
        self.word_embedding_dim = 300
        self.word_embedding = nn.Embedding(num_embeddings=config.vocabulary_size, embedding_dim=self.word_embedding_dim)
        with open('word_embedding-' + str(3) + '-' + str(300)  + '-' + str(200) + '-' + str(20) + '-' + 'command' + '.pkl', 'rb') as word_embedding_f:
            self.word_embedding.weight.data.copy_(pickle.load(word_embedding_f))
        self.category_embedding = nn.Embedding(num_embeddings=20, embedding_dim=50)
        self.dropout = nn.Dropout(p=0.2, inplace=True)
        self.dropout_ = nn.Dropout(p=0.2, inplace=False)
        self.auxiliary_loss = None
        '''

    def initialize(self):
        nn.init.uniform_(self.category_embedding.weight, -0.1, 0.1)

    # Input
    # title_text          : [batch_size, report_num, max_title_length] 제목
    # title_mask          : [batch_size, report_num, max_title_length]
    # content_text        : [batch_size, report_num, max_content_length] 본문
    # content_mask        : [batch_size, report_num, max_content_length]
    # time_text           : [batch_size, report_num, max_time_length] 시간 정보
    # time_mask           : [batch_size, report_num, max_time_length]
    # category            : [batch_size, report_num] 카테고리
    # user_embedding      : [batch_size, user_embedding_dim] (옵션) 사용자 임베딩
    # Output
    # report_representation : [batch_size, report_num, report_embedding_dim] 명령 임베딩
    def forward(self, title, content_text, content_mask, time_text, time_mask, user_embedding):
        raise Exception('Function forward must be implemented at sub-class')

    # Input
    # report_representation : [batch_size, report_num, unfused_report_embedding_dim]
    # category                 : [batch_size, report_num]
    # Output
    # report_representation : [batch_size, report_num, report_embedding_dim]
    def feature_fusion(self, report_representation, category):
        category_representation = self.category_embedding(category)                                                      # [batch_size, report_num, category_embedding_dim]
        report_representation = torch.cat([report_representation, self.dropout(category_representation)], dim=2)   # [batch_size, report_num, report_embedding_dim]
        return report_representation


class NAML(ReportEncoder):
    #def __init__(self, config: Config):
    def __init__(self, config):
        super(NAML, self).__init__(config)
        '''
        self.max_time_length = 20
        self.max_content_length = 200
        self.cnn_kernel_num = 64
        self.report_embedding_dim = 64
        self.time_conv = Conv1D('naive', 300, 64, 3)
        self.content_conv = Conv1D('naive', 300, 64, 3)
        self.time_attention = Attention(64, 64)
        self.content_attention = Attention(64, 64)
        self.category_affine = nn.Linear(50, 64, bias=True)
        self.affine1 = nn.Linear(64, 64, bias=True)
        self.affine2 = nn.Linear(64, 1, bias=False)

        '''
        self.max_time_length = config.max_time_length
        self.max_content_length = config.max_content_length
        self.cnn_kernel_num = config.cnn_kernel_num
        self.report_embedding_dim = config.cnn_kernel_num
        self.title_conv = Conv1D(config.cnn_method, config.word_embedding_dim, config.cnn_kernel_num, config.cnn_window_size)
        self.time_conv = Conv1D(config.cnn_method, config.word_embedding_dim, config.cnn_kernel_num, config.cnn_window_size)
        self.content_conv = Conv1D(config.cnn_method, config.word_embedding_dim, config.cnn_kernel_num, config.cnn_window_size)
        self.title_attention = Attention(config.cnn_kernel_num, config.attention_dim)
        self.time_attention = Attention(config.cnn_kernel_num, config.attention_dim)
        self.content_attention = Attention(config.cnn_kernel_num, config.attention_dim)
        self.category_affine = nn.Linear(config.category_embedding_dim, config.cnn_kernel_num, bias=True)
        self.affine1 = nn.Linear(config.cnn_kernel_num, config.attention_dim, bias=True)
        self.affine2 = nn.Linear(config.attention_dim, 1, bias=False)
        

    def initialize(self):
        super().initialize()
        self.time_attention.initialize()
        self.content_attention.initialize()
        nn.init.xavier_uniform_(self.category_affine.weight)
        nn.init.zeros_(self.category_affine.bias)
        nn.init.xavier_uniform_(self.affine1.weight)
        nn.init.zeros_(self.affine1.bias)
        nn.init.xavier_uniform_(self.affine2.weight)

    def forward(self, title_text, title_mask, content_text, content_mask, time_text, time_mask, category, user_embedding):
        batch_size = time_text.size(0)
        report_num = time_text.size(1)
        batch_report_num = batch_size * report_num
        # 1. word embedding
        title_emb = self.dropout(self.word_embedding(title_text))
        time_emb  = self.dropout(self.word_embedding(time_text))
        content_emb = self.dropout(self.word_embedding(content_text))

        # 실제 길이 기반으로 reshape
        title_len = title_emb.size(-2)   # ← 중요
        time_len  = time_emb.size(-2)
        content_len = content_emb.size(-2)

        title_w = title_emb.view(batch_report_num, title_len, self.word_embedding_dim)
        time_w  = time_emb.view(batch_report_num, time_len,  self.word_embedding_dim)
        content_w = content_emb.view(batch_report_num, content_len, self.word_embedding_dim)

        # 2. CNN encoding
        title_c = self.dropout_(self.title_conv(title_w.permute(0, 2, 1)).permute(0, 2, 1))
        time_c = self.dropout_(self.time_conv(time_w.permute(0, 2, 1)).permute(0, 2, 1))                                                        # [batch_size * report_num, max_title_length, cnn_kernel_num]
        content_c = self.dropout_(self.content_conv(content_w.permute(0, 2, 1)).permute(0, 2, 1))                                               # [batch_size * report_num, max_content_length, cnn_kernel_num]

        # 3. attention layer
        title_representation = self.title_attention(title_c).view([batch_size, report_num, self.cnn_kernel_num])
        time_representation = self.time_attention(time_c).view([batch_size, report_num, self.cnn_kernel_num])                                   # [batch_size, report_num, cnn_kernel_num]
        content_representation = self.content_attention(content_c).view([batch_size, report_num, self.cnn_kernel_num])                          # [batch_size, report_num, cnn_kernel_num]
        #print(f"title_representation shape: {title_representation.shape}")
        #print(f"time_representation shape: {time_representation.shape}")
        #print(f"content_representation shape: {content_representation.shape}")


        # 4. category encoding
        category_representation = F.relu(self.category_affine(self.category_embedding(category)), inplace=True)                                             # [batch_size, report_num, cnn_kernel_num]
        #print(f"category_representation shape: {category_representation.shape}")

       
        # 5. multi-view attention
        feature = torch.stack([title_representation, time_representation, content_representation, category_representation], dim=2)                                       # [batch_size, report_num, 3, cnn_kernel_num]
        alpha = F.softmax(self.affine2(torch.tanh(self.affine1(feature))), dim=2)                                                               # [batch_size, report_num, 3, 1]
        report_representation = (feature * alpha).sum(dim=2, keepdim=False)                                                                     # [batch_size, report_num, cnn_kernel_num]
        return report_representation

