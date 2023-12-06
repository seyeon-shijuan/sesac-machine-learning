import torch
import torch.nn as nn


class DilatedConvolutionalTransformer(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dilation_rates, heads):
        super(DilatedConvolutionalTransformer, self).__init__()

        # Dilated Convolutional Layers
        self.conv_layers = nn.ModuleList([
            nn.Conv1d(in_channels=input_dim, out_channels=hidden_dim, kernel_size=3, dilation=dilation_rate)
            for dilation_rate in dilation_rates
        ])

        # Transformer Layers
        self.transformer_layers = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=heads)
        self.transformer = nn.TransformerEncoder(self.transformer_layers, num_layers=num_layers)

        # Output Layer
        self.output_layer = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # Dilated Convolutional Layers
        for conv_layer in self.conv_layers:
            x = conv_layer(x)

        # Permute for Transformer input
        x = x.permute(2, 0, 1)

        # Transformer Layers
        x = self.transformer(x)

        # Permute back to original shape
        x = x.permute(1, 2, 0)

        # Output Layer
        x = self.output_layer(x)

        return x

# 모델 인스턴스 생성
input_dim = 5  # 예시: start, low, high, end, TA metrics
hidden_dim = 64
num_layers = 3
dilation_rates = [1, 2, 4]  # 예시의 dilation rates
heads = 4

model = DilatedConvolutionalTransformer(input_dim, hidden_dim, num_layers, dilation_rates, heads)

# 예시 입력 데이터 생성
batch_size = 32
sequence_length = 50
input_data = torch.randn((batch_size, input_dim, sequence_length))

# 모델에 입력 전달
output = model(input_data)
print("Output shape:", output.shape)