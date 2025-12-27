import math
from config import Config
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence
from layers import MultiHeadAttention, Attention, ScaledDotProduct_CandidateAttention, CandidateAttention, GCN
from reportEncoders import ReportEncoder
from torch_scatter import scatter_sum, scatter_softmax # need to be installed by following `https://pytorch-scatter.readthedocs.io/en/latest`


class UserEncoder(nn.Module):
    def __init__(self, report_encoder: ReportEncoder, config: Config):
        super(UserEncoder, self).__init__()
        self.report_embedding_dim = report_encoder.report_embedding_dim
        self.position_embedding = nn.Embedding(num_embeddings=config.position_num, embedding_dim=config.position_embedding_dim)
        self.report_encoder = report_encoder
        self.device = torch.device('cuda')
        self.dropout = nn.Dropout(p=config.dropout_rate, inplace=True)
        self.dropout_ = nn.Dropout(p=config.dropout_rate, inplace=False)
        self.auxiliary_loss = None

    def initialize(self):
        nn.init.uniform_(self.position_embedding.weight, -0.1, 0.1)

    # Input (각 배치의 사용자 정보)
    # user_dept                     : [batch_size] 사용자 부서
    # user_pos                      : [batch_size] 사용자 직급 (position)
    # user_rank                     : [batch_size] 사용자 계급
    # user_unit                     : [batch_size] 사용자 부대/팀
    # user_title_text               : [batch_size, max_history_num, max_title_length] 사용자 히스토리 명령들의 제목
    # user_title_mask               : [batch_size, max_history_num, max_title_length]
    # user_content_text             : [batch_size, max_history_num, max_content_length] 사용자 히스토리 명령들의 본문
    # user_content_mask             : [batch_size, max_history_num, max_content_length]
    # user_time_text                : [batch_size, max_history_num, max_time_length] 사용자 히스토리 명령들의 시간
    # user_time_mask                : [batch_size, max_history_num, max_time_length]
    # user_history_category         : [batch_size, max_history_num] 히스토리 명령들의 카테고리
    # user_history_mask             : [batch_size, max_history_num] 실제 히스토리 항목 마스크
    # user_history_graph            : [batch_size, max_history_num, max_history_num] 히스토리 명령 간의 그래프
    # user_history_category_mask    : [batch_size, category_num] 사용자가 읽은 카테고리
    # user_history_category_indices : [batch_size, max_history_num] 히스토리 명령의 카테고리 인덱스
    # user_embedding                : [batch_size, user_embedding_dim] (옵션) 사용자 ID 임베딩
    # candidate_news_representation : [batch_size, candidate_num, report_embedding_dim] 후보 명령들의 임베딩
    # Output
    # user_representation           : [batch_size, candidate_num, report_embedding_dim] 각 후보 명령에 대한 사용자 벡터
    def forward(self, user_dept, user_pos, user_rank, user_unit, user_title_text, user_title_mask, user_content_text, user_content_mask, user_time_text, user_time_mask, user_history_category, \
                user_history_mask, user_history_graph, user_history_category_mask, user_history_category_indices, user_embedding, candidate_news_representation):
        raise Exception('Function forward must be implemented at sub-class')


class ATT(UserEncoder):
    def __init__(self, report_encoder: ReportEncoder, config: Config):
        super(ATT, self).__init__(report_encoder, config)
        self.report_attention = Attention(self.report_embedding_dim, config.attention_dim)
        self.position_affine = nn.Linear(config.position_embedding_dim, config.cnn_kernel_num, bias=True)
        self.fusion = nn.Linear(self.report_embedding_dim + config.cnn_kernel_num, self.report_embedding_dim, bias=True)

    def initialize(self):
        super().initialize()
        self.report_attention.initialize()
        nn.init.xavier_uniform_(self.position_affine.weight)
        nn.init.zeros_(self.position_affine.bias)
        nn.init.xavier_uniform_(self.fusion.weight)
        nn.init.zeros_(self.fusion.bias)

    def forward(self, user_dept, user_pos, user_rank, user_unit, user_title_text, user_title_mask, user_content_text, user_content_mask, user_time_text, user_time_mask, user_history_category, \
                user_history_mask, user_history_graph, user_history_category_mask, user_history_category_indices, user_embedding, candidate_report_representation):
        report_num = candidate_report_representation.size(1)
        history_embedding = self.report_encoder(user_title_text, user_title_mask, \
                                              user_content_text, user_content_mask, user_time_text, user_time_mask, user_history_category, user_embedding)            # [batch_size, max_history_num, embedding_dim]
        
        # Step 2: Attention으로 히스토리 벡터 생성 (히스토리 명령들의 가중치 합)
        history_vector = self.report_attention(history_embedding)  # [batch_size, embedding_dim]

        # Step 3: 사용자 속성 인코딩 (직급 정보 활용)
        position_representation = F.relu(self.position_affine(self.position_embedding(user_pos)), inplace=True)  # [batch_size, cnn_kernel_num]

        # Step 4: 히스토리 벡터와 속성 벡터 결합
        fusion_input = torch.cat([history_vector, position_representation], dim=-1)
        fused_user_vector = torch.tanh(self.fusion(fusion_input))  # [batch_size, embedding_dim]

        # Step 5: 각 후보 명령에 대해 사용자 벡터 확장
        # 동일한 사용자 벡터를 모든 후보 명령에 대해 반복
        user_representation = fused_user_vector.unsqueeze(dim=1).expand(-1, report_num, -1)  # [batch_size, 1+neg_num, embedding_dim]
        
        return user_representation
