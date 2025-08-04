import torch
import torch.nn as nn
import torch.nn.functional as F

def _replace_fc(model, output_dim):
    d = model.fc.in_features
    model.fc = torch.nn.Linear(d, output_dim)
    return model


class MultiTaskHead(torch.nn.Module):
    def __init__(self, n_features, n_classes_list):
        super(MultiTaskHead, self).__init__()
        self.fc_list = [
            torch.nn.Linear(n_features, n_classes).cuda() for n_classes in n_classes_list
        ]

    def forward(self, x):
        outputs = []
        for head in self.fc_list:
            out = head(x)
            outputs.append(out)
        return outputs



class MILAttention(nn.Module):
    def __init__(self, feature_dim, attention_dim=128, temperature=10):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(feature_dim, attention_dim),
            nn.Tanh(),
            nn.Linear(attention_dim, 1),
        )
        self.temperature = temperature

    def forward(self, features):  # features: [B, N, D]
        A = self.attention(features)  # [B, N, 1]
        A = torch.softmax(A / self.temperature, dim=1)  # softmax over N
        bag_features = torch.sum(A * features, dim=1)  # [B, D]
        return bag_features, A.squeeze(-1)  # A: [B, N]


class MILCBMWrapperModel(nn.Module):
    def __init__(self, num_classes, input_dim=2048, clip_dim=512, feature_dim=512):
        super().__init__()

        self.attention = MILAttention(feature_dim)

        self.concept_head = nn.Linear(input_dim, clip_dim)  
        self.tmp_head = nn.Linear(clip_dim, input_dim)  

        self.classifier = nn.Linear(feature_dim, num_classes)  # MIL classifier
        # self.alpha = nn.Parameter(torch.tensor(0.01))  # weight for MIL vs image prediction

    def forward(self, bag_emd):

        concept_logits = self.concept_head(bag_emd)
        concept_logits = F.normalize(concept_logits, dim=-1)
        
        bag_features, attn_weights = self.attention(concept_logits)

        bag_out_features = self.tmp_head(bag_features)

        mil_logits = self.classifier(bag_features)  # [B, num_classes]

        return mil_logits, attn_weights, concept_logits, bag_out_features