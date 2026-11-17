import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict, Any


class ConvBlock(nn.Module):
    """Khối tích chập tiêu chuẩn có BatchNorm, LeakyReLU và Dropout."""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1, padding: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [B, C_in, H, W]
        out = self.act(self.bn(self.conv(x)))
        # out shape: [B, C_out, H, W]
        return out


class CRNNHandwritingRecognizer(nn.Module):
    """
    Kiến trúc CRNN (CNN + BiLSTM + CTC) chuyên nhận diện từ ngữ & dòng chữ viết tay liền nét.
    Tuân thủ nghiêm ngặt protocol 'ai-model-architect':
      - Không hardcode thiết bị (.cuda())
      - Ghi chú Tensor Shape đầy đủ trên từng bước biến đổi
      - Log_softmax cho CTC Loss ở tầng đầu ra
    """

    DEFAULT_CHARSET = (
        "0123456789"
        "abcdefghijklmnopqrstuvwxyz"
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        " -_.,'\""
    )

    def __init__(
        self,
        img_h: int = 32,
        in_channels: int = 1,
        num_classes: int = len(DEFAULT_CHARSET) + 1,  # +1 cho CTC Blank Token (chỉ số 0)
        hidden_size: int = 256,
        num_lstm_layers: int = 2,
        dropout: float = 0.2
    ):
        super().__init__()
        self.img_h = img_h
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.hidden_size = hidden_size

        # ── 1. TẦNG CNN FEATURE EXTRACTOR ──
        # Mục tiêu: Giảm chiều cao từ 32 xuống 1 (H: 32 -> 16 -> 8 -> 4 -> 2 -> 1)
        # Đồng thời bảo tồn độ phân giải chiều ngang W theo trục thời gian nét chữ
        self.cnn = nn.Sequential(
            # Stage 1: [B, 1, 32, W] -> [B, 64, 16, W/2]
            ConvBlock(in_channels, 64),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Stage 2: [B, 64, 16, W/2] -> [B, 128, 8, W/4]
            ConvBlock(64, 128),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Stage 3: [B, 128, 8, W/4] -> [B, 256, 4, W/4] (pooling không đối xứng W giữ nguyên)
            ConvBlock(128, 256),
            ConvBlock(256, 256),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),

            # Stage 4: [B, 256, 4, W/4] -> [B, 512, 2, W/4]
            ConvBlock(256, 512),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),

            # Stage 5: [B, 512, 2, W/4] -> [B, 512, 1, W/4]
            ConvBlock(512, 512, kernel_size=2, stride=1, padding=0),
            nn.Dropout2d(p=dropout)
        )

        # ── 2. TẦNG BI-DIRECTIONAL LSTM SEQUENCE MODELING ──
        # Sau CNN, kích thước mỗi lát cắt thời gian là 512
        self.rnn = nn.LSTM(
            input_size=512,
            hidden_size=hidden_size,
            num_layers=num_lstm_layers,
            bidirectional=True,
            dropout=dropout if num_lstm_layers > 1 else 0.0,
            batch_first=False  # PyTorch CTC Loss mặc định ưu tiên time-first: [T, B, C]
        )

        # ── 3. TẦNG PHÂN PHỐI XÁC SUẤT CTC (PROJECTION HEAD) ──
        # BiLSTM 2 chiều nên output dimension = hidden_size * 2
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass của CRNN:
          x: [Batch, Channel=1, Height=32, Width]
          Returns:
            log_probs: [Time_Steps (W/4 - 1), Batch, Num_Classes]
        """
        # Kiểm tra shape đầu vào
        b, c, h, w = x.shape
        assert h == self.img_h, f"Chiều cao ảnh phải là {self.img_h}, nhận được: {h}"
        assert c == self.in_channels, f"Số kênh ảnh phải là {self.in_channels}, nhận được: {c}"

        # 1. CNN Feature Extraction
        # conv_feat: [B, 512, 1, W']
        conv_feat = self.cnn(x)
        assert conv_feat.size(2) == 1, f"Chiều cao sau CNN phải bằng 1, nhận được {conv_feat.size(2)}"

        # 2. Map-to-Sequence (Chuyển đổi sang chuỗi thời gian)
        # [B, 512, 1, W'] -> [B, 512, W'] -> [W', B, 512] (T, B, D)
        seq_feat = conv_feat.squeeze(2).permute(2, 0, 1)
        t, b_seq, d = seq_feat.shape
        assert b_seq == b, "Batch size bị lệch trong quá trình permute sequence!"

        # 3. BiLSTM Contextual Sequence Modeling
        # rnn_out: [T, B, hidden_size * 2]
        rnn_out, _ = self.rnn(seq_feat)

        # 4. Projection to Character Class Logits
        # logits: [T, B, num_classes]
        logits = self.fc(rnn_out)

        # 5. Log Softmax cho CTC Loss (an toàn số học, không tràn số)
        log_probs = F.log_softmax(logits, dim=2)
        assert not torch.isnan(log_probs).any(), "Phát hiện giá trị NaN trong log_probs!"

        return log_probs


class CTCLabelConverter:
    """Bộ mã hóa và giải mã chuỗi ký tự theo chuẩn CTC Loss (Blank token = 0)."""

    def __init__(self, charset: str = CRNNHandwritingRecognizer.DEFAULT_CHARSET):
        self.charset = charset
        # Ký tự Blank token luôn ở chỉ số 0
        self.char2idx = {char: idx + 1 for idx, char in enumerate(charset)}
        self.idx2char = {idx + 1: char for idx, char in enumerate(charset)}
        self.idx2char[0] = ""  # CTC Blank

    def encode(self, texts: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Mã hóa danh sách các từ thành Tensor 1D phẳng cùng độ dài targets:
          targets: [sum(target_lengths)]
          target_lengths: [Batch]
        """
        encoded_list = []
        lengths = []
        for text in texts:
            encoded_text = [self.char2idx[ch] for ch in text if ch in self.char2idx]
            encoded_list.extend(encoded_text)
            lengths.append(len(encoded_text))

        targets = torch.tensor(encoded_list, dtype=torch.long)
        target_lengths = torch.tensor(lengths, dtype=torch.long)
        return targets, target_lengths

    def decode_greedy(self, log_probs: torch.Tensor) -> List[str]:
        """
        Giải mã Greedy Decoder tốt nhất từ log_probs:
          log_probs: [T, B, Num_Classes]
          Trả về: List các chuỗi văn bản nhận diện được
        """
        # Lấy nhãn có xác suất lớn nhất tại mỗi bước thời gian: [T, B]
        best_indices = torch.argmax(log_probs, dim=2).transpose(0, 1)  # [B, T]
        decoded_texts = []

        for b_idx in range(best_indices.size(0)):
            seq = best_indices[b_idx].tolist()
            chars = []
            prev_token = None
            for token in seq:
                # Bỏ qua token lặp liên tiếp và bỏ qua CTC blank token (0)
                if token != 0 and token != prev_token:
                    chars.append(self.idx2char.get(token, ""))
                prev_token = token
            decoded_texts.append("".join(chars))

        return decoded_texts
