import torch
from torch import nn
from torchvision import models


class PageSequenceClassifier(nn.Module):
    MAX_SEQUENCE_LENGTH = 200

    def __init__(self, num_classes, hidden_dim):
        super().__init__()
        edge_resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        spread_resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        edge_conv1 = edge_resnet.conv1
        edge_resnet.conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        with torch.no_grad():
            edge_resnet.conv1.weight[:] = edge_conv1.weight.mean(dim=1, keepdim=True)

        spread_conv1 = spread_resnet.conv1
        spread_resnet.conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        with torch.no_grad():
            spread_resnet.conv1.weight[:] = spread_conv1.weight.mean(
                dim=1, keepdim=True
            )

        self.edge_encoder = nn.Sequential(*list(edge_resnet.children())[:-1])
        self.spread_encoder = nn.Sequential(*list(spread_resnet.children())[:-1])

        # 512 (Left Edge) + 512 (Right Edge) + 512 (Stitched Spread) + 1 (Aspect Ratio)
        self.feature_projection = nn.Linear(512 + 512 + 512 + 1, hidden_dim)
        self.projection_dropout = nn.Dropout(p=0.3)

        self.position_embedding = nn.Embedding(self.MAX_SEQUENCE_LENGTH, hidden_dim)

        self.sequence_processor = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim // 2,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.5,
        )

        self.dropout = nn.Dropout(p=0.6)
        self.classifier = nn.Linear(hidden_dim, num_classes)

        # log(ratio) + is_wide binary flag
        self.ratio_head = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes),
        )

        # is_wide=1 fires class 2
        with torch.no_grad():
            self.ratio_head[0].weight.zero_()
            self.ratio_head[0].bias.zero_()
            self.ratio_head[0].weight[0, 1] = 5.0
            self.ratio_head[2].weight.zero_()
            self.ratio_head[2].bias.zero_()
            self.ratio_head[2].weight[2, 0] = 5.0

        self.transition_scores = nn.Parameter(torch.zeros(num_classes, num_classes))
        with torch.no_grad():
            self.transition_scores[0, 0] = -2.0  # right -> right = bad
            self.transition_scores[1, 1] = 0  # left -> left = neutral
            self.transition_scores[0, 1] = 1.0  # right -> left = good
            self.transition_scores[1, 0] = 0.5  # left -> right = good
            self.transition_scores[0, 2] = -1.0  # right -> double = bad
            self.transition_scores[1, 2] = 0.5  # left -> double = good

    def forward(self, left_edge_x, right_edge_x, spread_x, ratios, lengths):
        batch_size, max_seq_len, c, h, w = left_edge_x.size()

        left_flat = left_edge_x.view(batch_size * max_seq_len, c, h, w)
        left_feats = self.edge_encoder(left_flat).view(batch_size, max_seq_len, -1)

        right_flat = right_edge_x.view(batch_size * max_seq_len, c, h, w)
        right_feats = self.edge_encoder(right_flat).view(batch_size, max_seq_len, -1)

        batch_size, max_seq_len, c, h, w = spread_x.size()
        spread_flat = spread_x.view(batch_size * max_seq_len, c, h, w)
        spread_feats = self.spread_encoder(spread_flat).view(
            batch_size, max_seq_len, -1
        )

        ratios_unsqueezed = ratios.unsqueeze(-1)
        combined_raw = torch.cat(
            [left_feats, right_feats, spread_feats, ratios_unsqueezed], dim=-1
        )

        projected = self.feature_projection(combined_raw)
        projected = self.projection_dropout(projected)

        positions = torch.arange(max_seq_len, device=projected.device)
        projected = projected + self.position_embedding(positions).unsqueeze(0)

        lengths_cpu = lengths.cpu()
        packed_input = nn.utils.rnn.pack_padded_sequence(
            projected, lengths_cpu, batch_first=True, enforce_sorted=False
        )
        packed_output, _ = self.sequence_processor(packed_input)
        seq_features, _ = nn.utils.rnn.pad_packed_sequence(
            packed_output, batch_first=True, total_length=max_seq_len
        )

        seq_features = self.dropout(seq_features)
        seq_logits = self.classifier(seq_features)

        # log(ratio) + is_wide flag for double signal
        log_ratio = torch.log(ratios_unsqueezed.clamp(min=0.1))  # [Batch, Seq, 1]
        is_wide = (ratios_unsqueezed > 1.2).float()  # [Batch, Seq, 1]
        ratio_features = torch.cat([log_ratio, is_wide], dim=-1)  # [Batch, Seq, 2]
        ratio_logits = self.ratio_head(ratio_features)  # [Batch, Seq, num_classes]
        logits = seq_logits + ratio_logits

        # bias each position based on previous position's prediction
        prev_probs = torch.softmax(logits[:, :-1, :], dim=-1)  # [Batch, Seq-1, 3]
        transition_bias = torch.matmul(
            prev_probs, self.transition_scores
        )  # [Batch, Seq-1, 3]
        zero_pad = torch.zeros(batch_size, 1, logits.size(-1), device=logits.device)
        transition_bias = torch.cat(
            [zero_pad, transition_bias], dim=1
        )  # [Batch, Seq, 3]
        logits = logits + transition_bias

        return logits
