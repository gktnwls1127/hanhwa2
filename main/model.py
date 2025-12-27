from config import Config
import torch
import torch.nn as nn
import torch.nn.functional as F
import reportEncoders
import userEncoders


class Model(nn.Module):
    def __init__(self, config: Config):
        super(Model, self).__init__()
        # Report Encoder: 명령 인코딩
        if config.report_encoder == 'NAML':
            self.report_encoder = reportEncoders.NAML(config)
        else:
            raise Exception(config.report_encoder + 'is not implemented')

        # User Encoder: 사용자 인코딩 (히스토리 + 속성 정보)
        if config.user_encoder == 'ATT':
            self.user_encoder = userEncoders.ATT(self.report_encoder, config)
        else:
            raise Exception(config.user_encoder + 'is not implemented')
        
        self.report_embedding_dim = self.report_encoder.report_embedding_dim

        self.use_user_embedding = False
        self.model_name = config.report_encoder + '-' + config.user_encoder
        self.dropout = nn.Dropout(p=config.dropout_rate)

        self.user_embedding = None
        
        self.click_predictor = config.click_predictor
        if self.click_predictor == 'mlp':
            self.mlp = nn.Linear(in_features=self.report_embedding_dim * 2, out_features=self.report_embedding_dim // 2, bias=True)
            self.out = nn.Linear(in_features=self.report_embedding_dim // 2, out_features=1, bias=True)


    def initialize(self):
        self.report_encoder.initialize()
        self.user_encoder.initialize()

        if self.click_predictor == 'mlp':
            nn.init.xavier_uniform_(self.mlp.weight, gain=nn.init.calculate_gain('relu'))
            nn.init.zeros_(self.mlp.bias)

    def forward(self, user_ID, user_dept, user_pos, user_rank, user_unit, user_title_text, user_content_text, user_time_text, user_history_mask, user_history_graph, user_history_category_mask, user_history_category_indices, user_embedding, 
        report_title_text, report_title_mask, report_content_text, report_content_mask, report_time_text, report_time_mask, report_category, user_history_category, sample_idx):
        user_title_mask = user_title_text.ne(0)
        user_content_mask = user_content_text.ne(0)
        user_time_mask = user_time_text.ne(0)
        
        # 사용자 임베딩 (사용자 히스토리 기반)
        user_embedding = self.dropout(self.user_embedding(user_ID)) if self.use_user_embedding else None
        
        # 명령 인코딩 (제목, 본문, 시간, 카테고리)
        report_representation = self.report_encoder(report_title_text, report_title_mask, report_content_text, report_content_mask, report_time_text, report_time_mask, report_category, user_embedding) # [batch_size, 1 + negative_sample_num, report_embedding_dim]
        
        # 사용자 인코딩 (사용자 히스토리 + 속성)
        # report_representation를 후보 명령 정보로 제공하여 사용자 벡터 생성
        user_representation = self.user_encoder(user_dept, user_pos, user_rank, user_unit, user_title_text, user_title_mask, user_content_text, user_content_mask, user_time_text, user_time_mask, user_history_category, \
                user_history_mask, user_history_graph, user_history_category_mask, user_history_category_indices, user_embedding, report_representation)  # [batch_size, 1 + negative_sample_num, report_embedding_dim]
        
        # USER-COMMAND 호환성 점수 계산 (사용자가 해당 명령을 읽을 확률)
        if self.click_predictor == 'dot_product':
            # Dot product: user_vec · command_vec
            logits = (user_representation * report_representation).sum(dim=2) # [batch_size, 1+negative_sample_num]
        elif self.click_predictor == 'mlp':
            # MLP: concatenate user_vec와 command_vec → MLP → 점수
            context = self.dropout(F.relu(self.mlp(torch.cat([user_representation, report_representation], dim=2)), inplace=True))
            logits = self.out(context).squeeze(dim=2)  # [batch_size, 1+negative_sample_num]
        
        return logits
