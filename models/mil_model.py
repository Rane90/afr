import torch
import torch.nn as nn
import torch.nn.functional as F


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


class MILAttentionModel(nn.Module):
    def __init__(
        self,
        feature_extractor: nn.Module,
        feature_dim: int,
        num_classes: int = 200,
        integrate_original: bool = False,
    ):
        super().__init__()
        self.feature_extractor = feature_extractor  # Loaded from Stage 1
        self.feature_dim = feature_dim

        self.concept_head = nn.Linear(self.feature_dim, 512)
        self.attention = MILAttention(self.feature_dim)
        self.classifier = nn.Linear(self.feature_dim, num_classes)

        self.integrate_original = integrate_original

    def forward(self, bag, original_img=None, warmup=False):
        """
        Args:
            bag: [B, N, C, H, W] - bag of masked images
            original_img: [B, C, H, W] - original image (optional)
            warmup: If True, just forward original_img through classifier (no MIL)
        Returns:
            logits: [B, num_classes]
            attn_weights: [B, N]
            concept_logits: [B, N, 512]
            features: [B, N, D]
        """
        if warmup:
            assert original_img is not None
            features = self.feature_extractor(original_img).squeeze()  # [B, D]
            logits = self.classifier(features)
            return logits

        B, N, C, H, W = bag.shape

        if self.integrate_original and original_img is not None:
            original_img = original_img.unsqueeze(1)  # [B, 1, C, H, W]
            bag = torch.cat([original_img, bag], dim=1)  # [B, N+1, C, H, W]
            N += 1

        bag = bag.view(B * N, C, H, W)  # flatten bag
        features = self.feature_extractor(bag).view(B, N, -1)  # [B, N, D]

        concept_logits = self.concept_head(features)  # [B, N, 512]
        concept_logits = F.normalize(concept_logits, dim=-1)

        bag_reprs, attn_weights = self.attention(features)  # [B, D], [B, N]
        logits = self.classifier(bag_reprs)  # [B, num_classes]

        return logits, attn_weights, concept_logits, features
    
    def get_embeddings(self, bag, original_img=None):
        """
        Returns the final MIL bag embedding (before classification).
        """
        if self.integrate_original and original_img is not None:
            original_img = original_img.unsqueeze(1)
            bag = torch.cat([original_img, bag], dim=1)

        B, N, C, H, W = bag.shape
        bag = bag.view(B * N, C, H, W)
        features = self.feature_extractor(bag).view(B, N, -1)
        bag_reprs, _ = self.attention(features)  # [B, D]
        return bag_reprs

    def get_instance_embeddings(self, bag):
        B, N, C, H, W = bag.shape
        bag = bag.view(B * N, C, H, W)
        features = self.feature_extractor(bag).view(B, N, -1)
        return features
