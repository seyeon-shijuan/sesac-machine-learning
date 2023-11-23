import torch
import torch.nn.functional as F
import torch.nn as nn


class Attention(nn.Module):
    def __init__(self, query_dim, key_dim, value_dim):
        super(Attention, self).__init__()
        self.query_dim = query_dim
        self.key_dim = key_dim
        self.value_dim = value_dim

        # 가중치 행렬 정의
        self.W_q = nn.Linear(query_dim, query_dim, bias=False)
        self.W_k = nn.Linear(key_dim, query_dim, bias=False)
        self.W_v = nn.Linear(value_dim, value_dim, bias=False)

    def forward(self, query, key, value):
        # 가중치 행렬 적용
        Q = self.W_q(query)
        K = self.W_k(key)
        V = self.W_v(value)

        # 쿼리와 키의 유사도 계산
        scores = torch.matmul(Q, K.transpose(-2, -1))

        # 소프트맥스 함수를 사용하여 가중치 계산
        attention_weights = F.softmax(scores, dim=-1)

        # 가중 평균을 계산하여 어텐션 값 얻기
        attention_output = torch.matmul(attention_weights, V)

        return attention_output

# 임의의 입력 데이터 생성
query = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32)
key = torch.tensor([[4.0, 5.0, 6.0]], dtype=torch.float32)
value = torch.tensor([[7.0, 8.0, 9.0]], dtype=torch.float32)

# 어텐션 적용
attention_layer = Attention(query_dim=query.size(-1), key_dim=key.size(-1), value_dim=value.size(-1))
result = attention_layer(query, key, value)

print("Query:", query)
print("Key:", key)
print("Value:", value)
print("Attention Result:", result)
