import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from torch_geometric.nn import GATConv, GraphConv
from torch.utils.data import Dataset


def set_random_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class CompositePolyLoss(nn.Module):
    def __init__(self, poly_alpha=1.2, poly_gamma=1.2, ce_weight=None, reduction="mean"):
        super().__init__()
        self.poly_alpha = poly_alpha
        self.poly_gamma = poly_gamma
        self.ce = nn.CrossEntropyLoss(weight=ce_weight, reduction="none")
        self.reduction = reduction

    def forward(self, logits, targets):
        ce_loss = self.ce(logits, targets)
        pt = F.softmax(logits, dim=-1)[range(len(targets)), targets] + 1e-7
        focal = (1 - pt) ** self.poly_gamma
        loss = ce_loss + self.poly_alpha * (1 - pt) * focal
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


class FocalLoss(nn.Module):
    """Proper Focal Loss: FL = -(1-p_t)^gamma * log(p_t).
    Down-weights easy samples (majority class) to focus on hard/tail samples.
    Gradients flow through the focusing factor unlike the old detached re-weighting.
    """
    def __init__(self, gamma=2.0, alpha=None, reduction="mean"):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        if self.alpha is not None:
            if self.alpha.device != focal_loss.device:
                self.alpha = self.alpha.to(focal_loss.device)
            focal_loss = self.alpha[targets] * focal_loss
        if self.reduction == "mean":
            return focal_loss.mean()
        if self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


class ClassBalancedFocalLoss(FocalLoss):
    """Focal Loss + Class-Balanced re-weighting by effective number of samples.
    beta: 0.9999 (close to 1 = stronger re-weighting), 0.99 (milder).
    samples_per_class: list of sample counts per class in training set.
    """
    def __init__(self, samples_per_class, beta=0.9999, gamma=2.0, reduction="mean"):
        effective_num = 1.0 - np.power(beta, np.array(samples_per_class, dtype=np.float64))
        weights = (1.0 - beta) / (effective_num + 1e-8)
        weights = weights / weights.sum() * len(samples_per_class)
        super().__init__(gamma=gamma, alpha=torch.tensor(weights, dtype=torch.float32),
                         reduction=reduction)


class PolyFocalLoss(nn.Module):
    """Poly-Focal Loss: Focal Loss + Poly perturbation term.
    L = -(1-p_t)^gamma * log(p_t) + alpha * (1-p_t)^(gamma+1)
    Combines Focal's hard-sample focusing with PolyLoss's polynomial coefficient tuning.
    Reference: PolyLoss paper (ICLR 2022) applied to Focal Loss base.
    """
    def __init__(self, gamma=2.0, poly_alpha=1.0, reduction="mean"):
        super().__init__()
        self.gamma = gamma
        self.poly_alpha = poly_alpha
        self.reduction = reduction

    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce_loss)  # pt = softmax(logits)[targets]
        # Focal term: -(1-p_t)^gamma * log(p_t)
        focal_term = (1 - pt) ** self.gamma * ce_loss
        # Poly perturbation: alpha * (1-p_t)^(gamma+1)
        poly_term = self.poly_alpha * (1 - pt) ** (self.gamma + 1)
        loss = focal_term + poly_term
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


class MELDDataset(Dataset):
    def __init__(
        self,
        metadata,
        visual_features,
        audio_features,
        text_features,
        label_encoder,
        modalities,
        embed_dims,
        is_training=False,
        augmentation_config=None,
    ):
        self.metadata = metadata.reset_index(drop=True)
        self.visual_features = visual_features
        self.audio_features = audio_features
        self.text_features = text_features
        self.label_encoder = label_encoder
        self.modalities = modalities
        self.embed_dims = embed_dims
        self.is_training = is_training
        self.augmentation_config = augmentation_config or {}

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        sample = self.metadata.iloc[idx]
        key = f"dia{sample['Dialogue_ID']}_utt{sample['Utterance_ID']}"
        features = {}

        if "t" in self.modalities:
            t = np.array(self.text_features.get(key, np.zeros(self.embed_dims["t"])))
            if self.is_training:
                cfg = self.augmentation_config.get("t", {})
                dropout_prob = cfg.get("dropout_prob", 0.2)
                noise_std = cfg.get("noise_std", 0.1)
                shuffle_prob = cfg.get("shuffle_prob", 0.3)
                shuffle_ratio = cfg.get("shuffle_ratio", 0.1)
                scale_prob = cfg.get("scale_prob", 0.4)
                scale_low = cfg.get("scale_low", 0.8)
                scale_high = cfg.get("scale_high", 1.2)
                dropout_mask = np.random.rand(len(t)) > 0.8
                t_aug = t.copy()
                if dropout_prob > 0:
                    dropout_mask = np.random.rand(len(t)) < dropout_prob
                    t_aug[dropout_mask] = 0
                if noise_std > 0:
                    noise = np.random.normal(0, noise_std, size=t.shape)
                    t_aug = t_aug + noise
                if shuffle_ratio > 0 and np.random.rand() < shuffle_prob:
                    shuffle_size = max(1, int(shuffle_ratio * len(t_aug)))
                    shuffle_indices = np.random.choice(len(t_aug), size=shuffle_size, replace=False)
                    shuffled_values = t_aug[shuffle_indices].copy()
                    np.random.shuffle(shuffled_values)
                    t_aug[shuffle_indices] = shuffled_values
                if np.random.rand() < scale_prob:
                    scale = np.random.uniform(scale_low, scale_high)
                    t_aug = t_aug * scale
                features["t"] = torch.tensor(t_aug, dtype=torch.float32)
            else:
                features["t"] = torch.tensor(t, dtype=torch.float32)

        if "a" in self.modalities:
            a = np.array(self.audio_features.get(key, np.zeros(self.embed_dims["a"])))
            if self.is_training:
                cfg = self.augmentation_config.get("a", {})
                noise = np.random.normal(0, cfg.get("noise_std", 0.05), size=a.shape)
                a_aug = a + noise
                features["a"] = torch.tensor(a_aug, dtype=torch.float32)
            else:
                features["a"] = torch.tensor(a, dtype=torch.float32)

        if "v" in self.modalities:
            v = np.array(self.visual_features.get(key, np.zeros(self.embed_dims["v"])))
            if self.is_training:
                cfg = self.augmentation_config.get("v", {})
                scale_low = cfg.get("scale_low", 0.95)
                scale_high = cfg.get("scale_high", 1.05)
                scale = np.random.uniform(scale_low, scale_high)
                v_aug = v * scale
                features["v"] = torch.tensor(v_aug, dtype=torch.float32)
            else:
                features["v"] = torch.tensor(v, dtype=torch.float32)

        features["_dialogue_id"] = int(sample["Dialogue_ID"])
        features["_utterance_id"] = int(sample["Utterance_ID"])
        label = self.label_encoder.transform([sample["Emotion"]])[0]
        return features, torch.tensor(label, dtype=torch.long)


META_KEYS = {"_dialogue_id", "_utterance_id"}

def custom_collate(batch):
    feats, labels = zip(*batch)
    batch_feats = {}
    for mod in feats[0].keys():
        if mod in META_KEYS:
            batch_feats[mod] = torch.tensor([f[mod] for f in feats])
        else:
            batch_feats[mod] = torch.stack([f[mod] for f in feats])
    return batch_feats, torch.stack(labels)


class MELDDialogueDataset(Dataset):
    """DFGCN-style dialogue-level dataset. Returns complete dialogues with speaker-temporal adjacency.

    Each dialogue has all utterances with 6-type relation adjacency (1=self, 2=intra-future,
    3=intra-past, 4=inter-future, 5=inter-past, 0=padding).
    """

    def __init__(self, metadata, visual_features, audio_features, text_features,
                 label_encoder, modalities, embed_dims, is_training=False, augmentation_config=None):
        self.visual_features = visual_features
        self.audio_features = audio_features
        self.text_features = text_features
        self.label_encoder = label_encoder
        self.modalities = modalities
        self.embed_dims = embed_dims
        self.is_training = is_training
        self.augmentation_config = augmentation_config or {}

        # Group by Dialogue_ID, sort by Utterance_ID within each dialogue
        self.dialogues = []
        for dia_id, dia_group in metadata.groupby('Dialogue_ID'):
            dia_group = dia_group.sort_values('Utterance_ID')
            utterances = []
            for _, row in dia_group.iterrows():
                key = f"dia{row['Dialogue_ID']}_utt{row['Utterance_ID']}"
                utterances.append({
                    'key': key,
                    'label': label_encoder.transform([row['Emotion']])[0],
                    'speaker': row['Speaker'],
                })
            if len(utterances) >= 2:
                self.dialogues.append(utterances)

    def __len__(self):
        return len(self.dialogues)

    @staticmethod
    def _build_speaker_adj(speakers):
        """Build 6-type relation adjacency (DFGCN-style).

        1=self-loop, 2=intra-future, 3=intra-past, 4=inter-future, 5=inter-past, 0=padding.
        """
        N = len(speakers)
        adj = torch.zeros(N, N, dtype=torch.long)
        for i in range(N):
            for j in range(N):
                if i == j:
                    adj[i, j] = 1
                elif speakers[i] == speakers[j]:
                    adj[i, j] = 2 if i < j else 3
                else:
                    adj[i, j] = 4 if i < j else 5
        return adj

    def __getitem__(self, idx):
        utterances = self.dialogues[idx]
        N = len(utterances)

        features = {}
        for m in self.modalities:
            feat_key = {'a': 'audio', 'v': 'visual', 't': 'text'}[m]
            feat_dict = getattr(self, f"{feat_key}_features")
            feats_list = []
            for utt in utterances:
                feat = np.array(feat_dict.get(utt['key'],
                                np.zeros(self.embed_dims[m])))
                feats_list.append(torch.tensor(feat, dtype=torch.float32))
            features[m] = torch.stack(feats_list)  # (N, D_m)

        labels = torch.tensor([utt['label'] for utt in utterances], dtype=torch.long)  # (N,)
        adj = self._build_speaker_adj([utt['speaker'] for utt in utterances])  # (N, N)
        mask = torch.ones(N, dtype=torch.bool)  # (N,)

        return features, labels, adj, mask


def dialogue_collate(batch):
    """Pad variable-length dialogues to max len in the batch."""
    features_list, labels_list, adj_list, mask_list = zip(*batch)

    max_len = max(m.shape[0] for m in mask_list)
    B = len(batch)

    batch_feats = {}
    for mod in features_list[0].keys():
        D = features_list[0][mod].shape[-1]
        padded = torch.zeros(B, max_len, D)
        for b_idx, feats in enumerate(features_list):
            N = feats[mod].shape[0]
            padded[b_idx, :N] = feats[mod]
        batch_feats[mod] = padded

    batch_labels = torch.full((B, max_len), -1, dtype=torch.long)
    for b_idx, lab in enumerate(labels_list):
        N = lab.shape[0]
        batch_labels[b_idx, :N] = lab

    batch_adj = torch.zeros(B, max_len, max_len, dtype=torch.long)
    for b_idx, adj in enumerate(adj_list):
        N = adj.shape[0]
        batch_adj[b_idx, :N, :N] = adj

    batch_mask = torch.zeros(B, max_len, dtype=torch.bool)
    for b_idx, mask in enumerate(mask_list):
        N = mask.shape[0]
        batch_mask[b_idx, :N] = True

    batch_feats["_adj"] = batch_adj
    batch_feats["_mask"] = batch_mask
    return batch_feats, batch_labels


def _build_dialogue_aware_edges(x, dialogue_ids, utterance_ids, k, tw):
    """Batch-level cosine k-NN with dialogue-aware temporal edges.

    Uses full-batch k-NN for rich graph structure (same as old random graph),
    but replaces fake batch-index temporal edges with real dialogue-consecutive
    temporal edges based on dialogue_id + utterance_id.
    """
    device = x.device
    B = x.size(0)
    x_norm = x / (x.norm(dim=1, keepdim=True) + 1e-8)
    sim = torch.mm(x_norm, x_norm.t())

    # Batch-level k-NN (rich cross-dialogue structure)
    k_eff = min(k, B - 1)
    _, idx = torch.topk(sim, k=k_eff + 1, dim=-1)

    # Build per-utterance dialogue position map
    dia_groups = {}
    for i in range(B):
        did = dialogue_ids[i].item() if isinstance(dialogue_ids[i], torch.Tensor) else int(dialogue_ids[i])
        dia_groups.setdefault(did, []).append(i)

    utt_dia = {}   # batch_idx -> dialogue_id
    utt_pos = {}   # batch_idx -> position within dialogue
    for did, indices in dia_groups.items():
        idx_list = sorted(indices, key=lambda i: utterance_ids[i].item()
                          if isinstance(utterance_ids[i], torch.Tensor) else int(utterance_ids[i]))
        for pos, i in enumerate(idx_list):
            utt_dia[i] = did
            utt_pos[i] = pos

    ei, ew = [], []
    for i in range(B):
        for j in idx[i][1:]:  # skip self
            jj = j.item()
            w = sim[i, jj].item()
            # Temporal boost for consecutive utterances in the same dialogue
            if utt_dia.get(i) == utt_dia.get(jj) and abs(utt_pos.get(i, -1) - utt_pos.get(jj, -1)) == 1:
                w += tw
            ei.append([i, jj])
            ew.append(w)

    # Force-add bidirectional temporal edges for consecutive same-dialogue pairs
    for indices in dia_groups.values():
        idx_list = sorted(indices, key=lambda i: utterance_ids[i].item()
                          if isinstance(utterance_ids[i], torch.Tensor) else int(utterance_ids[i]))
        for p in range(len(idx_list) - 1):
            a, b = idx_list[p], idx_list[p + 1]
            if [a, b] not in ei:
                ei += [[a, b], [b, a]]
                ew += [tw, tw]

    if len(ei) == 0:
        return torch.zeros(2, 0, dtype=torch.long, device=device), torch.zeros(0, device=device)
    return torch.tensor(ei).t().contiguous().to(device), torch.tensor(ew).to(device)


def _adj_to_edge_index(adj):
    """Convert (N,N) speaker-temporal adjacency (values 1-5) to edge_index/edge_weight.

    Relation weights: self=1.0, intra=0.7, inter=0.4.
    """
    N = adj.shape[0]
    ei_list, ew_list = [], []
    weight_map = {1: 1.0, 2: 0.7, 3: 0.7, 4: 0.4, 5: 0.4}
    for i in range(N):
        for j in range(N):
            r = adj[i, j].item()
            if r > 0:
                ei_list.append([i, j])
                ew_list.append(weight_map[r])
    if len(ei_list) == 0:
        return torch.zeros(2, 0, dtype=torch.long), torch.zeros(0)
    return torch.tensor(ei_list).t().contiguous(), torch.tensor(ew_list)


def _adj_to_edge_index_with_type(adj):
    """Convert (N,N) speaker-temporal adjacency to (edge_index, edge_type).

    Returns edge_type in 1..5 for gradient-flow indexing via learnable weights.
    """
    N = adj.shape[0]
    ei_list, et_list = [], []
    for i in range(N):
        for j in range(N):
            r = adj[i, j].item()
            if r > 0:
                ei_list.append([i, j])
                et_list.append(r)  # 1-5
    if len(ei_list) == 0:
        return torch.zeros(2, 0, dtype=torch.long), torch.zeros(0, dtype=torch.long)
    return (torch.tensor(ei_list).t().contiguous(),
            torch.tensor(et_list, dtype=torch.long))


class TextGraphEncoder(nn.Module):
    def __init__(self, embed_dim, num_classes, graph_k=8, temporal_weight=1.0, dropout=0.15,
                 learnable_relations=False):
        super().__init__()
        if embed_dim == 768:
            hidden_dim = 192
            heads = 4
        else:
            hidden_dim = 256
            heads = 4

        self.gat1 = GATConv(embed_dim, hidden_dim, heads=heads, concat=True)
        self.gat2 = GATConv(hidden_dim * heads, hidden_dim, heads=heads, concat=True)
        self.gat3 = GATConv(hidden_dim * heads, hidden_dim, heads=heads, concat=False)
        self.gat4 = GATConv(hidden_dim, hidden_dim, heads=heads, concat=False)
        self.fc = nn.Linear(hidden_dim, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.res1 = nn.Linear(embed_dim, hidden_dim * heads)
        self.res2 = nn.Linear(hidden_dim * heads, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim * heads, eps=1e-5)
        self.ln2 = nn.LayerNorm(hidden_dim * heads, eps=1e-5)
        self.ln3 = nn.LayerNorm(hidden_dim, eps=1e-5)
        self.ln4 = nn.LayerNorm(hidden_dim, eps=1e-5)
        self.bn = nn.BatchNorm1d(embed_dim, momentum=0.1)
        self.k = graph_k
        self.tw = temporal_weight
        self.hidden_dim = hidden_dim
        self.heads = heads

        self.feature_proj = nn.Sequential(
            nn.Linear(hidden_dim, embed_dim),
            nn.LayerNorm(embed_dim, eps=1e-5),
            nn.GELU(),
        )

        self.learnable_relations = learnable_relations
        if learnable_relations:
            self.relation_weights = nn.Parameter(torch.tensor([0.0, 1.0, 0.7, 0.7, 0.4, 0.4]))
        else:
            self.register_buffer('_static_weights', torch.tensor([0.0, 1.0, 0.7, 0.7, 0.4, 0.4]))

    def _forward_gat(self, x, edge_index, edge_weight):
        """Shared GAT processing. Caller must apply BatchNorm first."""
        r = self.res1(x)
        x = self.gat1(x, edge_index, edge_weight)
        x = self.ln1(self.dropout(self.relu(x)) + r)
        r = x
        x = self.gat2(x, edge_index, edge_weight)
        x = self.ln2(self.dropout(self.relu(x)) + r)
        r = self.res2(x)
        x = self.gat3(x, edge_index, edge_weight)
        x = self.ln3(self.dropout(self.relu(x)) + r)
        r = x
        x = self.gat4(x, edge_index, edge_weight)
        x = self.ln4(self.dropout(self.relu(x)) + r)
        x = self.dropout(x)
        return x

    def _dialogue_forward(self, x, adj, mask, apply_fc=True, apply_feature_proj=False):
        """Process dialogue-level input: (B, max_N, D), (B, max_N, max_N), (B, max_N)."""
        B, max_N = x.shape[0], x.shape[1]
        outputs = []
        for b in range(B):
            N = mask[b].sum().item()
            x_b = x[b, :N]  # (N, D)
            if N <= 1:
                feat = self.bn(x_b)  # Apply BN even without GAT
            else:
                adj_b = adj[b, :N, :N]
                edge_index, edge_type = _adj_to_edge_index_with_type(adj_b)
                edge_index = edge_index.to(x.device)
                edge_type = edge_type.to(x.device)
                if self.learnable_relations:
                    edge_weight = self.relation_weights[edge_type]
                else:
                    edge_weight = self._static_weights[edge_type]
                feat = self._forward_gat(self.bn(x_b), edge_index, edge_weight)
            if apply_feature_proj:
                feat = self.feature_proj(feat)
            if apply_fc:
                feat = self.fc(feat)
            out_dim = feat.shape[-1]
            padded = torch.zeros(max_N, out_dim, device=x.device)
            padded[:N] = feat
            outputs.append(padded.unsqueeze(0))
        return torch.cat(outputs, dim=0)  # (B, max_N, out_dim)

    def forward(self, x, adj=None, mask=None, dialogue_ids=None, utterance_ids=None):
        if x.dim() == 3 and adj is not None and mask is not None:
            return self._dialogue_forward(x, adj, mask, apply_fc=True, apply_feature_proj=False)
        # Old utterance-level forward (backward compat)
        if self.training:
            x = x * (torch.rand_like(x) > 0.1).float()
        x = self.bn(x)
        edge_index, edge_weight = self.create_similarity_edge_index(x, dialogue_ids, utterance_ids)
        x = self._forward_gat(x, edge_index, edge_weight)
        return self.fc(x)

    def create_similarity_edge_index(self, x, dialogue_ids=None, utterance_ids=None):
        if dialogue_ids is not None and utterance_ids is not None:
            return _build_dialogue_aware_edges(x, dialogue_ids, utterance_ids, self.k, self.tw)
        x_norm = x / (x.norm(dim=1, keepdim=True) + 1e-8)
        sim = torch.mm(x_norm, x_norm.t())
        k = min(self.k, x.size(0) - 1)
        _, idx = torch.topk(sim, k=k + 1, dim=-1)
        ei, ew = [], []
        for i in range(x.size(0)):
            for j in idx[i][1:]:
                w = sim[i, j].item()
                if abs(j - i) == 1:
                    w += self.tw
                ei.append([i, j.item()])
                ew.append(w)
        for i in range(x.size(0) - 1):
            if [i, i + 1] not in ei:
                ei += [[i, i + 1], [i + 1, i]]
                ew += [self.tw, self.tw]
        ei = torch.tensor(ei).t().contiguous()
        ew = torch.tensor(ew)
        return ei.to(x.device), ew.to(x.device)

    def extract_features(self, x, adj=None, mask=None, dialogue_ids=None, utterance_ids=None):
        if x.dim() == 3 and adj is not None and mask is not None:
            return self._dialogue_forward(x, adj, mask, apply_fc=False, apply_feature_proj=True)
        x = self.bn(x)
        edge_index, edge_weight = self.create_similarity_edge_index(x, dialogue_ids, utterance_ids)
        feat = self._forward_gat(x, edge_index, edge_weight)
        return self.feature_proj(feat)


class AudioGraphEncoder(nn.Module):
    def __init__(self, embed_dim, num_classes, graph_k=8, temporal_weight=1.0, dropout=0.4,
                 learnable_relations=False):
        super().__init__()
        self.conv1 = GraphConv(embed_dim, 256)
        self.conv2 = GraphConv(256, 256)
        self.conv3 = GraphConv(256, 256)
        self.fc = nn.Linear(256, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.res = nn.Linear(embed_dim, 256)
        self.ln1 = nn.LayerNorm(256)
        self.ln2 = nn.LayerNorm(256)
        self.ln3 = nn.LayerNorm(256)
        self.bn = nn.BatchNorm1d(embed_dim)
        self.k = graph_k
        self.tw = temporal_weight

        self.feature_proj = nn.Sequential(
            nn.Linear(256, embed_dim),
            nn.LayerNorm(embed_dim, eps=1e-5),
            nn.GELU(),
        )

        self.learnable_relations = learnable_relations
        if learnable_relations:
            self.relation_weights = nn.Parameter(torch.tensor([0.0, 1.0, 0.7, 0.7, 0.4, 0.4]))
        else:
            self.register_buffer('_static_weights', torch.tensor([0.0, 1.0, 0.7, 0.7, 0.4, 0.4]))

    def _forward_graphconv(self, x, edge_index, edge_weight):
        """Shared GraphConv processing. Caller must apply BatchNorm first."""
        r = self.res(x)
        x = self.conv1(x, edge_index, edge_weight)
        x = self.ln1(self.dropout(self.relu(x)) + r)
        r = x
        x = self.conv2(x, edge_index, edge_weight)
        x = self.ln2(self.dropout(self.relu(x)) + r)
        r = x
        x = self.conv3(x, edge_index, edge_weight)
        x = self.ln3(self.dropout(self.relu(x)) + r)
        x = self.dropout(x)
        return x

    def _dialogue_forward(self, x, adj, mask, apply_fc=True, apply_feature_proj=False):
        """Process dialogue-level input: (B, max_N, D), (B, max_N, max_N), (B, max_N)."""
        B, max_N = x.shape[0], x.shape[1]
        outputs = []
        for b in range(B):
            N = mask[b].sum().item()
            x_b = x[b, :N]  # (N, D)
            if N <= 1:
                feat = self.bn(x_b)
            else:
                adj_b = adj[b, :N, :N]
                edge_index, edge_type = _adj_to_edge_index_with_type(adj_b)
                edge_index = edge_index.to(x.device)
                edge_type = edge_type.to(x.device)
                if self.learnable_relations:
                    edge_weight = self.relation_weights[edge_type]
                else:
                    edge_weight = self._static_weights[edge_type]
                feat = self._forward_graphconv(self.bn(x_b), edge_index, edge_weight)
            if apply_feature_proj:
                feat = self.feature_proj(feat)
            if apply_fc:
                feat = self.fc(feat)
            out_dim = feat.shape[-1]
            padded = torch.zeros(max_N, out_dim, device=x.device)
            padded[:N] = feat
            outputs.append(padded.unsqueeze(0))
        return torch.cat(outputs, dim=0)  # (B, max_N, out_dim)

    def forward(self, x, adj=None, mask=None, dialogue_ids=None, utterance_ids=None):
        if x.dim() == 3 and adj is not None and mask is not None:
            return self._dialogue_forward(x, adj, mask, apply_fc=True, apply_feature_proj=False)
        # Old utterance-level forward (backward compat)
        if self.training:
            x = x * (torch.rand_like(x) > 0.05).float()
        x = self.bn(x)
        edge_index, edge_weight = self.create_similarity_edge_index(x, dialogue_ids, utterance_ids)
        x = self._forward_graphconv(x, edge_index, edge_weight)
        return self.fc(x)

    def create_similarity_edge_index(self, x, dialogue_ids=None, utterance_ids=None):
        if dialogue_ids is not None and utterance_ids is not None:
            return _build_dialogue_aware_edges(x, dialogue_ids, utterance_ids, self.k, self.tw)
        x_norm = x / (x.norm(dim=1, keepdim=True) + 1e-8)
        sim = torch.mm(x_norm, x_norm.t())
        k = min(self.k, x.size(0) - 1)
        _, idx = torch.topk(sim, k=k + 1, dim=-1)
        ei, ew = [], []
        for i in range(x.size(0)):
            for j in idx[i][1:]:
                w = sim[i, j].item()
                if abs(j - i) == 1:
                    w += self.tw
                ei.append([i, j.item()])
                ew.append(w)
        for i in range(x.size(0) - 1):
            if [i, i + 1] not in ei:
                ei += [[i, i + 1], [i + 1, i]]
                ew += [self.tw, self.tw]
        ei = torch.tensor(ei).t().contiguous()
        ew = torch.tensor(ew)
        return ei.to(x.device), ew.to(x.device)

    def extract_features(self, x, adj=None, mask=None, dialogue_ids=None, utterance_ids=None):
        if x.dim() == 3 and adj is not None and mask is not None:
            return self._dialogue_forward(x, adj, mask, apply_fc=False, apply_feature_proj=True)
        x = self.bn(x)
        edge_index, edge_weight = self.create_similarity_edge_index(x, dialogue_ids, utterance_ids)
        feat = self._forward_graphconv(x, edge_index, edge_weight)
        return self.feature_proj(feat)


class VisualGraphEncoder(nn.Module):
    def __init__(self, embed_dim, num_classes, graph_k=8, temporal_weight=1.0, dropout=0.15,
                 learnable_relations=False):
        super().__init__()
        self.gat1 = GATConv(embed_dim, 128, heads=4, concat=True)
        self.gat2 = GATConv(512, 256, heads=2, concat=True)
        self.gat3 = GATConv(512, 256, heads=1, concat=False)
        self.fc = nn.Linear(256, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.res1 = nn.Linear(embed_dim, 512)
        self.res2 = nn.Linear(512, 512)
        self.res3 = nn.Linear(512, 256)
        self.ln1 = nn.LayerNorm(512)
        self.ln2 = nn.LayerNorm(512)
        self.ln3 = nn.LayerNorm(256)
        self.bn = nn.BatchNorm1d(embed_dim)
        self.k = graph_k
        self.tw = temporal_weight

        self.feature_proj = nn.Sequential(
            nn.Linear(256, embed_dim),
            nn.LayerNorm(embed_dim, eps=1e-5),
            nn.GELU(),
        )

        self.learnable_relations = learnable_relations
        if learnable_relations:
            self.relation_weights = nn.Parameter(torch.tensor([0.0, 1.0, 0.7, 0.7, 0.4, 0.4]))
        else:
            self.register_buffer('_static_weights', torch.tensor([0.0, 1.0, 0.7, 0.7, 0.4, 0.4]))

    def _forward_gat(self, x, edge_index, edge_weight):
        """Shared GAT processing. Caller must apply BatchNorm first."""
        r = self.res1(x)
        x = self.gat1(x, edge_index, edge_weight)
        x = self.ln1(self.dropout(self.relu(x)) + r)
        r = self.res2(x)
        x = self.gat2(x, edge_index, edge_weight)
        x = self.ln2(self.dropout(self.relu(x)) + r)
        r = self.res3(x)
        x = self.gat3(x, edge_index, edge_weight)
        x = self.ln3(self.dropout(self.relu(x)) + r)
        x = self.dropout(x)
        return x

    def _dialogue_forward(self, x, adj, mask, apply_fc=True, apply_feature_proj=False):
        """Process dialogue-level input: (B, max_N, D), (B, max_N, max_N), (B, max_N)."""
        B, max_N = x.shape[0], x.shape[1]
        outputs = []
        for b in range(B):
            N = mask[b].sum().item()
            x_b = x[b, :N]  # (N, D)
            if N <= 1:
                feat = self.bn(x_b)
            else:
                adj_b = adj[b, :N, :N]
                edge_index, edge_type = _adj_to_edge_index_with_type(adj_b)
                edge_index = edge_index.to(x.device)
                edge_type = edge_type.to(x.device)
                if self.learnable_relations:
                    edge_weight = self.relation_weights[edge_type]
                else:
                    edge_weight = self._static_weights[edge_type]
                feat = self._forward_gat(self.bn(x_b), edge_index, edge_weight)
            if apply_feature_proj:
                feat = self.feature_proj(feat)
            if apply_fc:
                feat = self.fc(feat)
            out_dim = feat.shape[-1]
            padded = torch.zeros(max_N, out_dim, device=x.device)
            padded[:N] = feat
            outputs.append(padded.unsqueeze(0))
        return torch.cat(outputs, dim=0)  # (B, max_N, out_dim)

    def forward(self, x, adj=None, mask=None, dialogue_ids=None, utterance_ids=None):
        if x.dim() == 3 and adj is not None and mask is not None:
            return self._dialogue_forward(x, adj, mask, apply_fc=True, apply_feature_proj=False)
        # Old utterance-level forward (backward compat)
        if self.training:
            x = x * (torch.rand_like(x) > 0.05).float()
        x = self.bn(x)
        edge_index, edge_weight = self.create_spatial_edge_index(x, dialogue_ids, utterance_ids)
        x = self._forward_gat(x, edge_index, edge_weight)
        return self.fc(x)

    def create_spatial_edge_index(self, x, dialogue_ids=None, utterance_ids=None):
        if dialogue_ids is not None and utterance_ids is not None:
            return _build_dialogue_aware_edges(x, dialogue_ids, utterance_ids, self.k, self.tw)
        x_norm = x / (x.norm(dim=1, keepdim=True) + 1e-8)
        sim = torch.mm(x_norm, x_norm.t())
        k = min(self.k, x.size(0) - 1)
        _, idx = torch.topk(sim, k=k + 1, dim=-1)
        ei, ew = [], []
        for i in range(x.size(0)):
            for j in idx[i][1:]:
                w = sim[i, j].item()
                if abs(j - i) == 1:
                    w += self.tw
                ei.append([i, j.item()])
                ew.append(w)
        for i in range(x.size(0) - 1):
            if [i, i + 1] not in ei:
                ei += [[i, i + 1], [i + 1, i]]
                ew += [self.tw, self.tw]
        ei = torch.tensor(ei).t().contiguous()
        ew = torch.tensor(ew)
        return ei.to(x.device), ew.to(x.device)

    def extract_features(self, x, adj=None, mask=None, dialogue_ids=None, utterance_ids=None):
        if x.dim() == 3 and adj is not None and mask is not None:
            return self._dialogue_forward(x, adj, mask, apply_fc=False, apply_feature_proj=True)
        x = self.bn(x)
        edge_index, edge_weight = self.create_spatial_edge_index(x, dialogue_ids, utterance_ids)
        feat = self._forward_gat(x, edge_index, edge_weight)
        return self.feature_proj(feat)


class DistillationLoss(nn.Module):
    def __init__(self, temperature, alpha, poly_alpha=1.2, poly_gamma=1.2, ce_weight=None, reduction="mean"):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.ce_loss = CompositePolyLoss(poly_alpha, poly_gamma, ce_weight, reduction)
        self.kl_div = nn.KLDivLoss(reduction="batchmean")

    def forward(self, student_logits, teacher_logits, labels):
        soft_t = F.softmax(teacher_logits / self.temperature, dim=1)
        student_lp = F.log_softmax(student_logits / self.temperature, dim=1)
        distill = self.kl_div(student_lp, soft_t) * (self.temperature ** 2)
        class_loss = self.ce_loss(student_logits, labels)
        return self.alpha * class_loss + (1 - self.alpha) * distill


class MultimodalWeightedAttention(nn.Module):
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = dim ** -0.5

    def forward(self, q, k, v, mask=None):
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        return torch.matmul(attn_weights, v), attn_weights


class IdentityAttention(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, query, key, value, **kwargs):
        batch_size = query.size(0)
        src_len = query.size(1)
        tgt_len = key.size(1) if key is not None else src_len
        dummy_weights = torch.ones(batch_size, src_len, tgt_len, device=query.device) / tgt_len
        return query, dummy_weights


class FeatureGatingModule(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.LayerNorm(dim // 2),
            nn.ReLU(),
            nn.Linear(dim // 2, dim),
            nn.Sigmoid(),
        )

    def forward(self, x):
        gates = self.gate(x)
        return x * gates


class InformationGateModule(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Linear(dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, x, context):
        inp = torch.cat([x, context], dim=1)
        g = self.gate(inp)
        return x * g


class UnifiedModalityGate(nn.Module):
    """Single-step modality refinement: context-conditioned gating + learnable residual.

    Replaces InformationGate (selection) + FeatureGating (refinement) with
    one module. Avoids double injection of fused_context and the resulting
    representation homogenization.
    """

    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.LayerNorm(dim, eps=1e-5),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.Sigmoid(),
        )
        self.residual_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, x, context):
        inp = torch.cat([x, context], dim=-1)
        g = self.gate(inp)
        return x * g + x * self.residual_scale


class AdaptiveFusionGate(nn.Module):
    def __init__(self, fusion_dim, num_modalities, fusion_dropout=0.0):
        super().__init__()
        self.fusion_dim = fusion_dim
        self.num_modalities = num_modalities
        self.fusion_dropout = fusion_dropout
        self.context_net = nn.Sequential(
            nn.Linear(fusion_dim * num_modalities, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.GELU(),
            nn.Linear(fusion_dim // 2, num_modalities),
        )
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, modality_features, bias=None):
        concat_features = torch.cat(modality_features, dim=-1)
        logits = self.context_net(concat_features)
        if bias is not None:
            logits = logits + bias
        weights = self.softmax(logits)
        if self.training and self.fusion_dropout > 0:
            weights = F.dropout(weights, p=self.fusion_dropout)
            weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        fused = torch.zeros_like(modality_features[0])
        for i, feat in enumerate(modality_features):
            fused += feat * weights[:, i].unsqueeze(1)
        return fused, weights


class PLEFusionGate(nn.Module):
    """PLE/CGC-style fusion: modality-specific experts + shared expert, gated.

    Prevents single-modality dominance by giving each modality its own expert
    that is only trained for that modality's features.
    """
    def __init__(self, fusion_dim, num_modalities):
        super().__init__()
        self.num_modalities = num_modalities
        # Modality-specific experts
        self.specific_experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(fusion_dim, fusion_dim),
                nn.LayerNorm(fusion_dim),
                nn.GELU(),
                nn.Linear(fusion_dim, fusion_dim),
            )
            for _ in range(num_modalities)
        ])
        # Shared expert (sees all modalities)
        self.shared_expert = nn.Sequential(
            nn.Linear(fusion_dim * num_modalities, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.GELU(),
            nn.Linear(fusion_dim, fusion_dim),
        )
        # Gate: decides how much each expert contributes
        num_experts = num_modalities + 1  # specific + shared
        self.gate_net = nn.Sequential(
            nn.Linear(fusion_dim * num_modalities, fusion_dim // 2),
            nn.GELU(),
            nn.Linear(fusion_dim // 2, num_experts),
        )
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, modality_features, bias=None):
        concat = torch.cat(modality_features, dim=-1)

        # Specific experts: each sees only its own modality
        specific_outputs = []
        for i, feat in enumerate(modality_features):
            specific_outputs.append(self.specific_experts[i](feat))

        # Shared expert: sees all modalities
        shared_out = self.shared_expert(concat)

        # Gate: decides contribution
        experts = torch.stack(specific_outputs + [shared_out], dim=1)  # (B, M+1, D)
        gate_logits = self.gate_net(concat)
        if bias is not None:
            # bias only applies to specific experts, shared gets 0 bias
            bias_padded = torch.cat([bias, torch.zeros_like(bias[:, :1])], dim=1)
            gate_logits = gate_logits + bias_padded
        gate_weights = self.softmax(gate_logits)  # (B, M+1)

        # Weighted sum
        fused = (experts * gate_weights.unsqueeze(-1)).sum(dim=1)
        # Return per-modality weights for diagnostics (sum specific + shared/n)
        diag_weights = gate_weights.clone()
        diag_weights[:, :-1] = diag_weights[:, :-1] + diag_weights[:, -1:] / self.num_modalities
        return fused, diag_weights[:, :-1]  # (B, M)


class ResidualFusionExpert(nn.Module):
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim, eps=1e-5),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
        )

    def forward(self, x):
        return self.net(x)


class TGRFRouteMixer(nn.Module):
    def __init__(self, fusion_dim, anchor_modality, modalities, dropout=0.1, preserve_logit_bias=0.75):
        super().__init__()
        self.anchor_modality = anchor_modality
        self.modalities = list(modalities)
        self.other_modalities = [m for m in self.modalities if m != self.anchor_modality]
        router_input_dim = fusion_dim * (1 + 2 * len(self.other_modalities))
        self.route_bias = nn.Parameter(
            torch.tensor([preserve_logit_bias, 0.0], dtype=torch.float32),
            requires_grad=False,
        )
        self.router = nn.Sequential(
            nn.Linear(router_input_dim, fusion_dim),
            nn.LayerNorm(fusion_dim, eps=1e-5),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, 2),
        )

    def forward(self, features, temperature=1.0):
        anchor = features[self.anchor_modality]
        parts = [anchor]
        for m in self.other_modalities:
            other = features[m]
            parts.extend([other, torch.abs(anchor - other)])
        route_logits = self.router(torch.cat(parts, dim=-1)) + self.route_bias
        return F.softmax(route_logits / max(temperature, 1e-6), dim=-1)


class TGRFPreserveBranch(nn.Module):
    def __init__(self, fusion_dim, modalities, dropout=0.1, residual_scale=0.08):
        super().__init__()
        self.modalities = list(modalities)
        self.experts = nn.ModuleDict(
            {m: ResidualFusionExpert(fusion_dim, dropout) for m in self.modalities}
        )
        self.norms = nn.ModuleDict(
            {m: nn.LayerNorm(fusion_dim, eps=1e-5) for m in self.modalities}
        )
        self.scales = nn.ParameterDict(
            {m: nn.Parameter(torch.tensor(residual_scale)) for m in self.modalities}
        )

    @staticmethod
    def _clamped_scale(scale):
        return torch.clamp(scale, min=0.0, max=1.0)

    def forward(self, features):
        outputs = {}
        for m in self.modalities:
            scale = self._clamped_scale(self.scales[m])
            delta = self.experts[m](features[m])
            outputs[m] = self.norms[m](features[m] + scale * delta)
        return outputs


class TGRFCrossBranch(nn.Module):
    def __init__(self, fusion_dim, modalities, anchor_modality, dropout=0.1, residual_scale=0.08):
        super().__init__()
        self.modalities = list(modalities)
        self.anchor_modality = anchor_modality
        self.other_modalities = [m for m in self.modalities if m != self.anchor_modality]
        self.norms = nn.ModuleDict(
            {m: nn.LayerNorm(fusion_dim, eps=1e-5) for m in self.modalities}
        )
        self.scales = nn.ParameterDict(
            {m: nn.Parameter(torch.tensor(residual_scale)) for m in self.modalities}
        )
        self.anchor_to_mod_gates = nn.ModuleDict()
        self.anchor_to_mod_updates = nn.ModuleDict()
        self.mod_to_anchor_gates = nn.ModuleDict()
        self.mod_to_anchor_updates = nn.ModuleDict()
        for m in self.other_modalities:
            self.anchor_to_mod_gates[m] = nn.Sequential(
                nn.Linear(fusion_dim * 3, fusion_dim),
                nn.LayerNorm(fusion_dim, eps=1e-5),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(fusion_dim, fusion_dim),
                nn.Sigmoid(),
            )
            self.anchor_to_mod_updates[m] = ResidualFusionExpert(fusion_dim, dropout)
            self.mod_to_anchor_gates[m] = nn.Sequential(
                nn.Linear(fusion_dim * 3, fusion_dim),
                nn.LayerNorm(fusion_dim, eps=1e-5),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(fusion_dim, fusion_dim),
                nn.Sigmoid(),
            )
            self.mod_to_anchor_updates[m] = ResidualFusionExpert(fusion_dim, dropout)

    @staticmethod
    def _clamped_scale(scale):
        return torch.clamp(scale, min=0.0, max=1.0)

    def forward(self, features):
        anchor = features[self.anchor_modality]
        anchor_update = torch.zeros_like(anchor)
        outputs = {self.anchor_modality: anchor}
        for m in self.other_modalities:
            other = features[m]
            pair = torch.cat([anchor, other, torch.abs(anchor - other)], dim=-1)
            mod_gate = self.anchor_to_mod_gates[m](pair)
            mod_delta = self.anchor_to_mod_updates[m](anchor)
            outputs[m] = self.norms[m](
                other + self._clamped_scale(self.scales[m]) * mod_gate * mod_delta
            )

            anchor_gate = self.mod_to_anchor_gates[m](pair)
            anchor_delta = self.mod_to_anchor_updates[m](other)
            anchor_update = anchor_update + anchor_gate * anchor_delta

        outputs[self.anchor_modality] = self.norms[self.anchor_modality](
            anchor + self._clamped_scale(self.scales[self.anchor_modality]) * anchor_update
        )
        return outputs


class TGRFContextAggregator(nn.Module):
    def __init__(self, fusion_dim, modalities, anchor_modality, dropout=0.1):
        super().__init__()
        self.modalities = list(modalities)
        self.anchor_modality = anchor_modality
        self.other_modalities = [m for m in self.modalities if m != self.anchor_modality]
        context_input_dim = fusion_dim * (len(self.modalities) + len(self.other_modalities))
        self.context_bias = nn.Parameter(
            torch.tensor(
                [0.35 if m == self.anchor_modality else 0.0 for m in self.modalities],
                dtype=torch.float32,
            ),
            requires_grad=False,
        )
        self.context_router = nn.Sequential(
            nn.Linear(context_input_dim, fusion_dim),
            nn.LayerNorm(fusion_dim, eps=1e-5),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, len(self.modalities)),
        )

    def _build_context_input(self, refined):
        anchor = refined[self.anchor_modality]
        parts = [refined[m] for m in self.modalities]
        for m in self.other_modalities:
            parts.append(torch.abs(anchor - refined[m]))
        return torch.cat(parts, dim=-1)

    def forward(self, refined, modality_bias=None, use_context_router=True):
        if use_context_router:
            context_logits = self.context_router(self._build_context_input(refined)) + self.context_bias
            if modality_bias is not None and modality_bias.size(-1) == len(self.modalities):
                context_logits = context_logits + modality_bias
            context_weights = F.softmax(context_logits, dim=-1)
        else:
            anchor = refined[self.anchor_modality]
            context_weights = torch.zeros(
                anchor.size(0),
                len(self.modalities),
                device=anchor.device,
                dtype=anchor.dtype,
            )
            anchor_idx = self.modalities.index(self.anchor_modality)
            context_weights[:, anchor_idx] = 1.0

        fused_context = torch.zeros_like(refined[self.anchor_modality])
        for idx, m in enumerate(self.modalities):
            fused_context = fused_context + refined[m] * context_weights[:, idx : idx + 1]
        return fused_context, context_weights


class TGuidedRouterFusionBlock(nn.Module):
    """Replace static fusion + gating with sample-wise routing.

    The block chooses between:
      1) preserve expert: keep each modality close to its own evidence
      2) cross expert: let text selectively inject complementary signals
    """

    def __init__(
        self,
        fusion_dim,
        modalities=("v", "a", "t"),
        dropout=0.1,
        residual_scale=0.08,
        route_temperature=1.0,
        preserve_logit_bias=0.75,
        use_preserve_branch=True,
        use_cross_branch=True,
        use_router_mixer=True,
        use_context_router=True,
    ):
        super().__init__()
        self.modalities = list(modalities)
        self.anchor_modality = "t" if "t" in self.modalities else self.modalities[0]
        self.route_temperature = route_temperature
        self.disabled = False
        self.use_preserve_branch = use_preserve_branch
        self.use_cross_branch = use_cross_branch
        self.use_router_mixer = use_router_mixer
        self.use_context_router = use_context_router
        if not (self.use_preserve_branch or self.use_cross_branch):
            raise ValueError("At least one TGRF branch must remain enabled")

        self.route_mixer = TGRFRouteMixer(
            fusion_dim=fusion_dim,
            anchor_modality=self.anchor_modality,
            modalities=self.modalities,
            dropout=dropout,
            preserve_logit_bias=preserve_logit_bias,
        )
        self.preserve_branch = TGRFPreserveBranch(
            fusion_dim=fusion_dim,
            modalities=self.modalities,
            dropout=dropout,
            residual_scale=residual_scale,
        )
        self.cross_branch = TGRFCrossBranch(
            fusion_dim=fusion_dim,
            modalities=self.modalities,
            anchor_modality=self.anchor_modality,
            dropout=dropout,
            residual_scale=residual_scale,
        )
        self.context_aggregator = TGRFContextAggregator(
            fusion_dim=fusion_dim,
            modalities=self.modalities,
            anchor_modality=self.anchor_modality,
            dropout=dropout,
        )

    def disable(self):
        self.disabled = True

    def _single_branch_weights(self, features, preserve_on):
        anchor = features[self.anchor_modality]
        weights = torch.zeros(anchor.size(0), 2, device=anchor.device, dtype=anchor.dtype)
        weights[:, 0 if preserve_on else 1] = 1.0
        return weights

    def _resolve_route_weights(self, features):
        if self.use_preserve_branch and self.use_cross_branch:
            if self.use_router_mixer:
                return self.route_mixer(features, temperature=self.route_temperature)
            anchor = features[self.anchor_modality]
            return torch.full(
                (anchor.size(0), 2),
                0.5,
                device=anchor.device,
                dtype=anchor.dtype,
            )
        if self.use_preserve_branch:
            return self._single_branch_weights(features, preserve_on=True)
        if self.use_cross_branch:
            return self._single_branch_weights(features, preserve_on=False)
        raise ValueError("At least one TGRF branch must remain enabled")

    def forward(self, features, modality_bias=None):
        if self.disabled or len(self.modalities) == 1:
            anchor = features[self.anchor_modality]
            weights = self._single_branch_weights(features, preserve_on=True)
            refined = {m: features[m] for m in self.modalities}
            return {
                "refined": refined,
                "fused_context": anchor,
                "route_weights": weights,
                "context_weights": None,
            }

        route_weights = self._resolve_route_weights(features)
        preserve_weight = route_weights[:, :1]
        cross_weight = route_weights[:, 1:2]

        preserve_outputs = self.preserve_branch(features) if self.use_preserve_branch else None
        cross_outputs = self.cross_branch(features) if self.use_cross_branch else None

        refined = {}
        for m in self.modalities:
            if preserve_outputs is not None and cross_outputs is not None:
                refined[m] = preserve_weight * preserve_outputs[m] + cross_weight * cross_outputs[m]
            elif preserve_outputs is not None:
                refined[m] = preserve_outputs[m]
            else:
                refined[m] = cross_outputs[m]

        fused_context, context_weights = self.context_aggregator(
            refined,
            modality_bias=modality_bias,
            use_context_router=self.use_context_router,
        )

        return {
            "refined": refined,
            "fused_context": fused_context,
            "route_weights": route_weights,
            "context_weights": context_weights,
        }


class MixtureOfExperts(nn.Module):
    def __init__(self, input_dim, output_dim, num_experts=4, top_k=2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(input_dim, output_dim),
                    nn.LayerNorm(output_dim),
                    nn.GELU(),
                    nn.Dropout(0.1),
                )
                for _ in range(num_experts)
            ]
        )
        self.gate = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.GELU(),
            nn.Linear(input_dim // 2, num_experts),
        )

    def forward(self, x):
        gate_scores = self.gate(x)
        topk_scores, topk_indices = torch.topk(gate_scores, self.top_k, dim=-1)
        topk_scores = F.softmax(topk_scores, dim=-1)
        output = torch.zeros_like(self.experts[0](x))
        for i in range(self.top_k):
            expert_idx = topk_indices[:, i]
            expert_weight = topk_scores[:, i].unsqueeze(1)
            for b in range(x.size(0)):
                expert_out = self.experts[expert_idx[b]](x[b : b + 1])
                output[b : b + 1] += expert_out * expert_weight[b]
        return output


class FeatureAlignmentModule(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.align_proj = nn.ModuleDict(
            {"source": nn.Linear(dim, dim), "target": nn.Linear(dim, dim)}
        )
        self.temperature = nn.Parameter(torch.tensor(1.0))

    def forward(self, source, target):
        source_proj = self.align_proj["source"](source)
        target_proj = self.align_proj["target"](target)
        cost_matrix = torch.cdist(source_proj, target_proj, p=2)
        K = torch.exp(-cost_matrix / self.temperature)
        K = K / K.sum(dim=1, keepdim=True)
        aligned_source = torch.matmul(K, target)
        return aligned_source, K


class ContrastiveLearningModule(nn.Module):
    def __init__(self, feature_dim, temperature=0.07):
        super().__init__()
        self.temperature = temperature
        self.projection = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, 128),
        )

    def forward(self, features, labels):
        z = self.projection(features)
        z = F.normalize(z, dim=1)
        sim_matrix = torch.matmul(z, z.t()) / self.temperature
        labels = labels.unsqueeze(1)
        mask = torch.eq(labels, labels.t()).float()
        exp_sim = torch.exp(sim_matrix)
        log_prob = sim_matrix - torch.log(exp_sim.sum(dim=1, keepdim=True))
        mask = mask - torch.eye(mask.size(0), device=mask.device)
        mean_log_prob_pos = (mask * log_prob).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        loss = -mean_log_prob_pos.mean()
        return loss


class EvidentialDirichletFusion(nn.Module):
    """Evidence fusion for MELD T/A/V utterance classification.

    Supports two fusion rules:
      - Dempster (default): standard Dempster-Shafer combination
      - group_consensus: TMLC-style progressive majority-weighting (AAAI 2026)
    """

    def __init__(
        self,
        embed_dims,
        num_classes,
        modalities=("v", "a", "t"),
        fusion_dim=256,
        num_transformer_layers=1,
        num_heads=4,
        dropout=0.15,
        use_context=True,
        fusion_rule="dempster",
        smote_config=None,
        use_modality_gate=False,       # Strategy C: learnable per-sample modality weights
        evidence_gate_threshold=None,  # Strategy B: hard gate if total evidence S < threshold
    ):
        super().__init__()
        self.modalities = list(modalities)
        self.num_classes = num_classes
        self.fusion_dim = fusion_dim
        self.use_context = use_context and len(self.modalities) > 1
        self.eps = 1e-7
        self.fusion_rule = fusion_rule
        self.smote_config = smote_config
        self.use_modality_gate = use_modality_gate
        self.evidence_gate_threshold = evidence_gate_threshold  # e.g. num_classes + 2 = 9

        self.proj = nn.ModuleDict(
            {
                m: nn.Sequential(
                    nn.Linear(embed_dims[m], fusion_dim),
                    nn.LayerNorm(fusion_dim, eps=1e-5),
                    nn.GELU(),
                    nn.Dropout(dropout),
                )
                for m in self.modalities
            }
        )

        if self.use_context:
            enc_layer = nn.TransformerEncoderLayer(
                d_model=fusion_dim,
                nhead=num_heads,
                dim_feedforward=fusion_dim * 4,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.opinion_context_encoder = nn.TransformerEncoder(
                enc_layer, num_layers=num_transformer_layers
            )
        else:
            self.opinion_context_encoder = nn.Identity()

        # Strategy C: Learnable per-sample modality gate
        if self.use_modality_gate:
            gate_input_dim = len(self.modalities) * fusion_dim
            self.modality_gate = nn.Sequential(
                nn.Linear(gate_input_dim, fusion_dim // 2),
                nn.LayerNorm(fusion_dim // 2, eps=1e-5),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(fusion_dim // 2, len(self.modalities)),
                nn.Softmax(dim=-1),
            )
        else:
            self.modality_gate = None

        self.evidence_heads = nn.ModuleDict(
            {
                m: nn.Sequential(
                    nn.Linear(fusion_dim, fusion_dim // 2),
                    nn.LayerNorm(fusion_dim // 2, eps=1e-5),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(fusion_dim // 2, num_classes),
                )
                for m in self.modalities
            }
        )

    def disable_gates(self):
        return None

    def disable_channel_attention(self):
        self.use_context = False

    def _evidence_to_alpha(self, evidence):
        return evidence + 1.0

    def _alpha_to_opinion(self, alpha):
        strength = torch.sum(alpha, dim=1, keepdim=True).clamp_min(self.eps)
        evidence = alpha - 1.0
        belief = evidence / strength
        uncertainty = self.num_classes / strength
        return belief, uncertainty

    def _apply_gates(self, alpha_by_modality, projected):
        """Apply Strategy B (Evidence Gating) + Strategy C (Learned Modality Gate).

        Returns gated alpha dict.
        """
        gate_info = {}  # for diagnostics

        # Strategy C: Learned per-sample modality weights
        if self.use_modality_gate and self.modality_gate is not None:
            concat_all = torch.cat([projected[m] for m in self.modalities], dim=-1)
            gate_weights = self.modality_gate(concat_all)  # (B, num_modalities)
            for idx, m in enumerate(self.modalities):
                w = gate_weights[:, idx:idx + 1]  # (B, 1)
                # w * alpha + (1-w) * 1.0  →  pull toward uniform when gate is low
                uniform_alpha = torch.ones_like(alpha_by_modality[m])
                alpha_by_modality[m] = w * alpha_by_modality[m] + (1.0 - w) * uniform_alpha
            gate_info["learned_weights"] = gate_weights.detach()

        # Strategy B: Hard evidence gating
        if self.evidence_gate_threshold is not None and self.evidence_gate_threshold > 0:
            gated_count = 0
            for m in self.modalities:
                S = alpha_by_modality[m].sum(dim=-1, keepdim=True)  # total evidence
                keep = (S > self.evidence_gate_threshold).float()
                gated_count += (keep == 0).sum().item()
                uniform_alpha = torch.ones_like(alpha_by_modality[m])
                alpha_by_modality[m] = keep * alpha_by_modality[m] + (1.0 - keep) * uniform_alpha
            gate_info["evidence_gated"] = gated_count

        return alpha_by_modality, gate_info

    def _combine_two(self, alpha_1, alpha_2, return_conflict=False):
        belief_1, uncertainty_1 = self._alpha_to_opinion(alpha_1)
        belief_2, uncertainty_2 = self._alpha_to_opinion(alpha_2)

        belief_outer = torch.bmm(belief_1.unsqueeze(2), belief_2.unsqueeze(1))
        conflict = belief_outer.sum(dim=(1, 2)) - belief_outer.diagonal(dim1=1, dim2=2).sum(dim=1)
        denom = (1.0 - conflict).unsqueeze(1).clamp_min(self.eps)

        fused_belief = (
            belief_1 * belief_2
            + belief_1 * uncertainty_2
            + belief_2 * uncertainty_1
        ) / denom
        fused_uncertainty = (uncertainty_1 * uncertainty_2) / denom
        fused_strength = self.num_classes / fused_uncertainty.clamp_min(self.eps)
        fused_evidence = fused_belief * fused_strength
        fused_alpha = fused_evidence + 1.0
        if return_conflict:
            return fused_alpha, conflict
        return fused_alpha

    def _combine_many(self, alpha_by_modality, return_conflict=False):
        if self.fusion_rule == "group_consensus":
            return self._combine_group_consensus(alpha_by_modality, return_conflict)
        fused = None
        conflicts = []
        for m in self.modalities:
            alpha = alpha_by_modality[m]
            if fused is None:
                fused = alpha
            else:
                fused, conflict = self._combine_two(fused, alpha, return_conflict=True)
                conflicts.append(conflict)
        if return_conflict:
            if conflicts:
                conflict_tensor = torch.stack(conflicts, dim=1)
            else:
                batch_size = next(iter(alpha_by_modality.values())).size(0)
                device = next(iter(alpha_by_modality.values())).device
                conflict_tensor = torch.zeros(batch_size, 0, device=device)
            return fused, conflict_tensor
        return fused

    def _combine_group_consensus(self, alpha_by_modality, return_conflict=False):
        """TMLC Group Consensus Opinion Aggregation (Tang et al., AAAI 2026).

        Progressively fuses views in opinion space, giving increasing weight
        to the accumulated majority opinion: gamma_group = (V-1)/V, gamma_view = 1/V.
        This suppresses outlier views — critical for tail classes where individual
        modalities are unreliable.
        """
        import torch.nn.functional as F

        active = [m for m in self.modalities if m in alpha_by_modality]
        if len(active) == 0:
            raise ValueError("No modalities to fuse")
        if len(active) == 1:
            alpha = alpha_by_modality[active[0]]
            if return_conflict:
                batch_size = alpha.size(0)
                device = alpha.device
                return alpha, torch.zeros(batch_size, 0, device=device)
            return alpha

        # Start with first modality
        belief, uncertainty = self._alpha_to_opinion(alpha_by_modality[active[0]])
        group_size = 1
        conflicts = []

        for m in active[1:]:
            group_size += 1
            b_v, u_v = self._alpha_to_opinion(alpha_by_modality[m])

            gamma_group = (group_size - 1.0) / group_size  # majority weight
            gamma_view = 1.0 / group_size                    # newcomer weight

            denom = gamma_view * uncertainty + gamma_group * u_v

            # Track disagreement as pseudo-conflict for diagnostics
            belief_diff = (belief - b_v).abs().sum(dim=1, keepdim=True) / 2.0
            conflicts.append(belief_diff)

            belief = (gamma_group * belief * u_v + gamma_view * b_v * uncertainty) / denom.clamp_min(self.eps)
            uncertainty = (uncertainty * u_v) / denom.clamp_min(self.eps)

        strength = self.num_classes / uncertainty.clamp_min(self.eps)
        evidence = belief * strength
        fused_alpha = evidence + 1.0

        if return_conflict:
            if conflicts:
                conflict_tensor = torch.cat(conflicts, dim=1)
            else:
                batch_size = fused_alpha.size(0)
                device = fused_alpha.device
                conflict_tensor = torch.zeros(batch_size, 0, device=device)
            return fused_alpha, conflict_tensor
        return fused_alpha

    def extract_contextualized_features(self, features):
        """Extract per-modality features after projection + context encoding.

        Returns dict {modality: tensor (B, fusion_dim)}.
        These are the features right before evidence heads — the right place
        for global SMOTE in multi-modal fusion space.
        """
        device = features[list(features.keys())[0]].device
        batch_size = features[list(features.keys())[0]].size(0)

        projected = {}
        tokens = []
        active_modalities = []
        for m in self.modalities:
            if m in features:
                projected[m] = self.proj[m](features[m])
            else:
                projected[m] = torch.zeros(batch_size, self.fusion_dim, device=device)
            tokens.append(projected[m])
            active_modalities.append(m)

        if self.use_context:
            token_tensor = torch.stack(tokens, dim=1)
            contextualized = self.opinion_context_encoder(token_tensor)
            for idx, m in enumerate(active_modalities):
                projected[m] = contextualized[:, idx, :]

        return projected

    def forward_from_features(self, projected):
        """Compute evidence + fusion directly from contextualized features."""
        alpha_by_modality = {}
        aux_logits = {}
        for m in self.modalities:
            raw_evidence = self.evidence_heads[m](projected[m])
            evidence = F.softplus(raw_evidence)
            alpha = self._evidence_to_alpha(evidence)
            alpha_by_modality[m] = alpha
            aux_logits[m] = torch.log(alpha.clamp_min(self.eps))

        alpha_by_modality, gate_info = self._apply_gates(alpha_by_modality, projected)
        self._last_gate_info = gate_info

        fused_alpha = self._combine_many(alpha_by_modality, return_conflict=False)
        logits = torch.log(fused_alpha.clamp_min(self.eps))
        return logits, aux_logits, fused_alpha.detach()

    def _smote_augment(self, projected, labels):
        """Online SMOTE in fused multi-modal feature space for tail classes."""
        cfg = self.smote_config
        tail_classes = cfg.get("tail_classes", [])
        ratio = cfg.get("ratio", 0.5)
        modalities = sorted(projected.keys())
        device = projected[modalities[0]].device
        dims = {m: projected[m].size(-1) for m in modalities}

        concat_feats = torch.cat([projected[m] for m in modalities], dim=-1)

        synthetic_feats = []
        synthetic_labels = []

        for c in tail_classes:
            mask = (labels == c)
            class_indices = mask.nonzero(as_tuple=True)[0]
            if len(class_indices) < 2:
                continue

            class_feats = concat_feats[class_indices]
            n_synthetic = max(1, int(len(class_indices) * ratio))

            for _ in range(n_synthetic):
                i, j = torch.randperm(len(class_indices))[:2]
                lam = torch.rand(1, device=device)
                syn_feat = class_feats[i] * lam + class_feats[j] * (1 - lam)
                synthetic_feats.append(syn_feat)
                synthetic_labels.append(c)

        if not synthetic_feats:
            return projected, labels

        syn_feats = torch.stack(synthetic_feats)

        syn_projected = {}
        offset = 0
        for m in modalities:
            d = dims[m]
            syn_projected[m] = syn_feats[:, offset:offset + d]
            offset += d

        for m in modalities:
            projected[m] = torch.cat([projected[m], syn_projected[m]], dim=0)
        syn_labels = torch.tensor(synthetic_labels, device=device)
        labels = torch.cat([labels, syn_labels], dim=0)

        return projected, labels

    def forward(self, features, return_attention=False, labels=None):
        device = features[list(features.keys())[0]].device
        batch_size = features[list(features.keys())[0]].size(0)

        projected = {}
        tokens = []
        active_modalities = []
        for m in self.modalities:
            if m in features:
                projected[m] = self.proj[m](features[m])
            else:
                projected[m] = torch.zeros(batch_size, self.fusion_dim, device=device)
            tokens.append(projected[m])
            active_modalities.append(m)

        if self.use_context:
            token_tensor = torch.stack(tokens, dim=1)
            contextualized = self.opinion_context_encoder(token_tensor)
            for idx, m in enumerate(active_modalities):
                projected[m] = contextualized[:, idx, :]

        # Online SMOTE in fusion feature space for tail classes
        augmented_labels = None
        if self.training and self.smote_config is not None and labels is not None:
            projected, labels = self._smote_augment(projected, labels)
            augmented_labels = labels

        alpha_by_modality = {}
        aux_logits = {}
        uncertainties = {}
        for m in self.modalities:
            raw_evidence = self.evidence_heads[m](projected[m])
            evidence = F.softplus(raw_evidence)
            alpha = self._evidence_to_alpha(evidence)
            alpha_by_modality[m] = alpha
            aux_logits[m] = torch.log(alpha.clamp_min(self.eps))
            uncertainties[m] = self.num_classes / alpha.sum(dim=1).clamp_min(self.eps)

        # Strategy B + C: Modality gating before fusion
        alpha_by_modality, gate_info = self._apply_gates(alpha_by_modality, projected)
        self._last_gate_info = gate_info

        fused_alpha, conflict_tensor = self._combine_many(alpha_by_modality, return_conflict=True)
        logits = torch.log(fused_alpha.clamp_min(self.eps))

        if return_attention:
            return logits, aux_logits, {
                "alpha": alpha_by_modality,
                "fused_alpha": fused_alpha,
                "uncertainty": uncertainties,
                "conflict": conflict_tensor,
            }

        if augmented_labels is not None:
            return logits, aux_logits, fused_alpha.detach(), augmented_labels
        return logits, aux_logits, fused_alpha.detach()


class EvidentialFusionLoss(nn.Module):
    def __init__(
        self,
        main_weight=0.6,
        aux_weights=None,
        ce_weight=None,
        kl_weight=0.01,
        num_classes=7,
        reduction="mean",
        tail_weight=0.0,
        tail_classes=None,
    ):
        super().__init__()
        self.main_weight = main_weight
        self.aux_weights = (
            {"t": 0.15, "a": 0.1, "v": 0.05} if aux_weights is None else aux_weights
        )
        aux_sum = sum(self.aux_weights.values())
        if aux_sum > 0:
            scale = (1 - main_weight) / aux_sum
            self.aux_weights = {k: v * scale for k, v in self.aux_weights.items()}
        self.register_buffer(
            "class_weight",
            ce_weight.detach().clone() if ce_weight is not None else torch.ones(num_classes),
        )
        self.kl_weight = kl_weight
        self.num_classes = num_classes
        self.reduction = reduction
        self.eps = 1e-7
        self.tail_weight = tail_weight
        self.tail_classes = tail_classes or []

    def _dirichlet_kl_to_uniform(self, alpha):
        beta = torch.ones_like(alpha)
        sum_alpha = alpha.sum(dim=1, keepdim=True)
        sum_beta = beta.sum(dim=1, keepdim=True)
        log_b_alpha = torch.lgamma(alpha).sum(dim=1, keepdim=True) - torch.lgamma(sum_alpha)
        log_b_beta = torch.lgamma(beta).sum(dim=1, keepdim=True) - torch.lgamma(sum_beta)
        digamma_delta = torch.digamma(alpha) - torch.digamma(sum_alpha)
        kl = ((alpha - beta) * digamma_delta).sum(dim=1, keepdim=True) + log_b_beta - log_b_alpha
        return kl.squeeze(1)

    def _edl_ce(self, log_alpha, targets):
        alpha = torch.exp(log_alpha).clamp_min(1.0 + self.eps)
        one_hot = F.one_hot(targets, num_classes=alpha.size(1)).float()
        strength = alpha.sum(dim=1, keepdim=True).clamp_min(self.eps)
        nll = (one_hot * (torch.digamma(strength) - torch.digamma(alpha))).sum(dim=1)

        target_weights = self.class_weight.to(alpha.device)[targets]
        nll = nll * target_weights

        non_target_alpha = (alpha - 1.0) * (1.0 - one_hot) + 1.0
        kl = self._dirichlet_kl_to_uniform(non_target_alpha)
        loss = nll + self.kl_weight * kl
        if self.reduction == "sum":
            return loss.sum()
        if self.reduction == "none":
            return loss
        return loss.mean()

    def forward(self, main_logits, aux_logits, targets, contrastive_loss=None, fused_alpha=None):
        total_loss = self.main_weight * self._edl_ce(main_logits, targets)
        if aux_logits is not None:
            for m, logits in aux_logits.items():
                if m in self.aux_weights:
                    total_loss += self.aux_weights[m] * self._edl_ce(logits, targets)

        # Evidence-distance tail class re-weighting (TMLC-inspired).
        # For tail classes: low-u (confident) samples get higher weight;
        # high-u (ambiguous) samples are down-weighted to avoid fitting noise.
        if self.tail_weight > 0 and self.tail_classes and fused_alpha is not None:
            with torch.no_grad():
                S = fused_alpha.sum(dim=1)
                u = self.num_classes / S.clamp_min(self.eps)
                tail_mask = torch.zeros(targets.shape, dtype=torch.bool, device=targets.device)
                for c in self.tail_classes:
                    tail_mask = tail_mask | (targets == c)
                # inverse-uncertainty gating: confident samples → higher weight
                sample_weight = 1.0 / u.clamp_min(0.1)
                sample_weight = sample_weight / sample_weight.mean().clamp_min(self.eps)
                gate = torch.where(tail_mask, sample_weight, torch.ones_like(u))
            total_loss = total_loss * gate.mean()

        return total_loss


class IdentityInformationGate(nn.Module):
    """S15M's explicit InformationGate bypass."""

    def forward(self, x, context):
        return x


class SubspaceTokenEncoder(nn.Module):
    """Exact S15 K=6, d=256, L=2 post-interaction encoder."""

    def __init__(self, token_count, token_dim, ffn_dim, layers, dropout, heads=8):
        super().__init__()
        self.token_count = token_count
        self.token_dim = token_dim
        self.ffn_dim = ffn_dim
        self.layers = layers
        self.split = nn.Linear(256, token_count * token_dim)
        layer = nn.TransformerEncoderLayer(
            token_dim, heads, ffn_dim, dropout, activation="gelu",
            batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, layers)

    def forward(self, x):
        batch, modalities, _ = x.shape
        tokens = self.split(x).reshape(
            batch, modalities * self.token_count, self.token_dim
        )
        encoded = self.transformer(tokens)
        return encoded.reshape(
            batch, modalities, self.token_count, self.token_dim
        ).mean(dim=2)


class HierarchicalAttentionFusion(nn.Module):
    def __init__(
        self,
        embed_dims,
        num_classes,
        modalities=["v", "a", "t"],
        fusion_dim=256,
        num_transformer_layers=1,
        num_heads=4,
        dropout=0.15,
        modality_importance=None,
        use_moe=True,
        use_contrastive=True,
        fusion_dropout=0.0,
        load_balance_lambda=0.0,
        use_aitm=False,
        use_ple=False,
        fusion_mode="hierarchical",  # "hierarchical" or "concat"
        **kwargs,
    ):
        super().__init__()
        self.modalities = modalities
        self.fusion_dim = fusion_dim
        self.use_moe = use_moe
        self.use_contrastive = use_contrastive
        self.load_balance_lambda = load_balance_lambda
        self.use_aitm = use_aitm
        self.use_ple = use_ple
        self.fusion_mode = fusion_mode
        self.text_dropout = kwargs.get("text_dropout", 0.0)
        self.aux_from_projected = kwargs.get("aux_from_projected", False)
        self.aux_from_raw_projected = kwargs.get("aux_from_raw_projected", False)
        self.merge_info_gate = kwargs.get("merge_info_gate", False)
        self.align_self_weight = kwargs.get("align_self_weight", 0.7)
        self.use_tguided_router_fusion = kwargs.get("use_tguided_router_fusion", False)
        self.router_use_preserve_branch = kwargs.get("router_use_preserve_branch", True)
        self.router_use_cross_branch = kwargs.get("router_use_cross_branch", True)
        self.router_use_learned_mixer = kwargs.get("router_use_learned_mixer", True)
        self.router_use_context_router = kwargs.get("router_use_context_router", True)
        # Strict layerwise controls deliberately reuse the original three
        # classifier modules.  These flags add no module/parameter and consume
        # no RNG, which keeps A0 bit-identical to Clean at initialization.
        self.layerwise_variant = kwargs.get("layerwise_variant")
        if self.layerwise_variant is not None:
            from layerwise_objectives import resolve_layerwise_variant

            self.layerwise_spec = resolve_layerwise_variant(self.layerwise_variant)
            self.layerwise_variant = self.layerwise_spec.name
        else:
            self.layerwise_spec = None
        self.layerwise_alpha = float(kwargs.get("layerwise_alpha", 0.3))
        self.layerwise_gamma = float(kwargs.get("layerwise_gamma", 0.1))
        self.layerwise_margin = float(kwargs.get("layerwise_margin", 0.0))
        self.layerwise_tau = float(kwargs.get("layerwise_tau", 0.1))
        self._head_logits = None
        self._layerwise_taps = None
        self._fusion_weights = None  # stored for loss / diagnostics
        self._context_weights = None
        self.capacity_variant = kwargs.get("capacity_variant")
        self.information_gate_enabled = bool(kwargs.get("information_gate_enabled", True))

        if modality_importance is None:
            modality_importance = {"t": 0.65, "a": 0.2, "v": 0.15}
        self.modality_importance = {k: v for k, v in modality_importance.items() if k in modalities}
        if self.modality_importance:
            total_weight = sum(self.modality_importance.values())
            self.modality_importance = {k: v / total_weight for k, v in self.modality_importance.items()}

        if use_moe:
            self.proj = nn.ModuleDict(
                {
                    m: MixtureOfExperts(embed_dims[m], fusion_dim, num_experts=4, top_k=2)
                    for m in modalities
                }
            )
        else:
            self.proj = nn.ModuleDict(
                {
                    m: nn.Sequential(
                        nn.Linear(embed_dims[m], fusion_dim),
                        nn.LayerNorm(fusion_dim, eps=1e-5),
                        nn.GELU(),
                        nn.Dropout(dropout),
                    )
                    for m in modalities
                }
            )

        if len(modalities) > 1:
            self.feature_aligners = nn.ModuleDict(
                {
                    f"{m1}_{m2}": FeatureAlignmentModule(fusion_dim)
                    for i, m1 in enumerate(modalities)
                    for m2 in modalities[i + 1 :]
                }
            )

        if self.use_tguided_router_fusion:
            self.router_fusion_block = TGuidedRouterFusionBlock(
                fusion_dim=fusion_dim,
                modalities=modalities,
                dropout=dropout,
                residual_scale=kwargs.get("router_residual_scale", 0.08),
                route_temperature=kwargs.get("router_temperature", 1.0),
                preserve_logit_bias=kwargs.get("router_preserve_bias", 0.75),
                use_preserve_branch=self.router_use_preserve_branch,
                use_cross_branch=self.router_use_cross_branch,
                use_router_mixer=self.router_use_learned_mixer,
                use_context_router=self.router_use_context_router,
            )
            self.adaptive_fusion = None
        elif use_ple:
            self.adaptive_fusion = PLEFusionGate(fusion_dim, len(modalities))
        else:
            self.adaptive_fusion = AdaptiveFusionGate(fusion_dim, len(modalities), fusion_dropout)

        # Concat fusion: simple MLP over concatenated projected features
        if fusion_mode == "concat":
            concat_dim = fusion_dim * len(modalities)
            self.concat_fusion = nn.Sequential(
                nn.Linear(concat_dim, fusion_dim * 2),
                nn.LayerNorm(fusion_dim * 2, eps=1e-5),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(fusion_dim * 2, fusion_dim),
                nn.LayerNorm(fusion_dim, eps=1e-5),
                nn.GELU(),
            )

        # Logit-gate fusion: T predicts base, gate controls A/V correction
        if fusion_mode == "logit_gate":
            self.logit_gate_net = nn.Sequential(
                nn.Linear(fusion_dim * len(modalities), fusion_dim),
                nn.LayerNorm(fusion_dim, eps=1e-5),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(fusion_dim, len(modalities) - 1),  # one gate per non-T modality
                nn.Sigmoid(),
            )
        self.feature_selectors = nn.ModuleDict({m: FeatureGatingModule(fusion_dim) for m in modalities})

        self.quality_detector = nn.ModuleDict(
            {
                m: nn.Sequential(
                    nn.Linear(embed_dims[m], 128),
                    nn.LayerNorm(128, eps=1e-5),
                    nn.GELU(),
                    nn.Dropout(0.1),
                    nn.Linear(128, 64),
                    nn.GELU(),
                    nn.Linear(64, 1),
                    nn.Sigmoid(),
                )
                for m in modalities
            }
        )

        if self.use_tguided_router_fusion:
            self.info_gates = None
            self.gates = nn.ModuleDict()
        elif self.merge_info_gate:
            # Merged: single-step modality refinement (replaces InfoGate + FeatureGating)
            self.info_gates = None
            self.gates = nn.ModuleDict(
                {
                    m: UnifiedModalityGate(fusion_dim, dropout)
                    for m in modalities
                }
            )
        else:
            self.info_gates = nn.ModuleDict({m: InformationGateModule(fusion_dim) for m in modalities})
            if use_aitm:
                gate_dims = {m: fusion_dim * 3 if m in ('a', 'v') else fusion_dim * 2 for m in modalities}
            else:
                gate_dims = {m: fusion_dim * 2 for m in modalities}
            self.gates = nn.ModuleDict(
                {
                    m: nn.Sequential(
                        nn.Linear(gate_dims[m], fusion_dim),
                        nn.LayerNorm(fusion_dim, eps=1e-5),
                        nn.GELU(),
                        nn.Dropout(dropout),
                        nn.Linear(fusion_dim, fusion_dim),
                        nn.Sigmoid(),
                    )
                    for m in modalities
                }
            )

        if len(modalities) > 1:
            self.cross_attn = nn.ModuleDict(
                {
                    q: nn.MultiheadAttention(
                        embed_dim=fusion_dim,
                        num_heads=num_heads,
                        dropout=dropout,
                        batch_first=True,
                    )
                    for q in modalities
                }
            )
        else:
            self.cross_attn = None

        if use_contrastive:
            self.contrastive = ContrastiveLearningModule(fusion_dim)

        if self.capacity_variant == "S15M":
            self.transformer_encoder = SubspaceTokenEncoder(
                token_count=6,
                token_dim=256,
                ffn_dim=1024,
                layers=2,
                dropout=dropout,
                heads=num_heads,
            )
            if fusion_dim != 256:
                raise ValueError("S15M requires fusion_dim=256")
            if self.information_gate_enabled:
                raise ValueError("S15M requires InformationGate OFF")
            self.info_gates = nn.ModuleDict(
                {m: IdentityInformationGate() for m in self.modalities}
            )
        else:
            enc_layer = nn.TransformerEncoderLayer(
                d_model=fusion_dim,
                nhead=num_heads,
                dim_feedforward=fusion_dim * 4,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.transformer_encoder = nn.TransformerEncoder(enc_layer, num_layers=num_transformer_layers)
        self.pool_queries = nn.Parameter(torch.randn(1, 3, fusion_dim))
        self.pool = nn.MultiheadAttention(fusion_dim, num_heads, dropout=dropout, batch_first=True)
        self.feature_integrator = nn.Sequential(
            nn.Linear(fusion_dim * 3, fusion_dim * 2),
            nn.LayerNorm(fusion_dim * 2, eps=1e-5),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.LayerNorm(fusion_dim, eps=1e-5),
            nn.Dropout(dropout * 0.5),
        )
        self.num_classifier_heads = 3
        self.classifiers = nn.ModuleList([nn.Linear(fusion_dim, num_classes) for _ in range(self.num_classifier_heads)])
        self.mod_classifiers = nn.ModuleDict(
            {
                m: nn.Sequential(
                    nn.Linear(fusion_dim, fusion_dim // 2),
                    nn.GELU(),
                    nn.Dropout(dropout * 0.5),
                    nn.Linear(fusion_dim // 2, num_classes),
                )
                for m in modalities
            }
        )
        self.use_class_conditional = kwargs.get("use_class_conditional", False)
        if self.use_class_conditional:
            self.class_modality_bias = nn.Parameter(torch.zeros(num_classes, len(modalities)))
            self.class_predictor = nn.Sequential(
                nn.Linear(fusion_dim * len(modalities), fusion_dim),
                nn.GELU(),
                nn.Linear(fusion_dim, num_classes),
            )

    def disable_gates(self):
        if self.use_tguided_router_fusion:
            self.router_fusion_block.disable()
            return
        for m in self.modalities:
            self.gates[m] = nn.Sequential(nn.Identity())

    def disable_channel_attention(self):
        if self.cross_attn is not None:
            for m in self.modalities:
                self.cross_attn[m] = nn.Sequential(IdentityAttention())

    def disable_alignment(self):
        if hasattr(self, "feature_aligners"):
            del self.feature_aligners

    def assess_modality_quality(self, x, modality):
        score = self.quality_detector[modality](x).mean()
        feature_mean = torch.abs(x).mean()
        feature_std = x.std()
        feature_sparsity = (torch.abs(x) < 1e-4).float().mean()
        x_normalized = F.softmax(x.abs(), dim=-1)
        entropy = -(x_normalized * torch.log(x_normalized + 1e-8)).sum(dim=-1).mean()
        normalized_entropy = entropy / torch.log(torch.tensor(x.size(-1), dtype=torch.float32))
        statistical_quality = (feature_mean * feature_std) / (feature_sparsity + 0.1)
        statistical_quality = torch.sigmoid(statistical_quality)
        calibrated_score = score * 0.5 + statistical_quality * 0.3 + normalized_entropy * 0.2
        return torch.clamp(calibrated_score, 0.1, 1.0)

    def _apply_text_dropout(self, features):
        if not (self.training and getattr(self, "text_dropout", 0.0) > 0 and "t" in features):
            return features

        text_feat = features["t"]
        keep_shape = [text_feat.size(0)] + [1] * (text_feat.dim() - 1)
        keep_mask = (
            torch.rand(*keep_shape, device=text_feat.device) >= self.text_dropout
        ).to(text_feat.dtype)
        return {k: v * keep_mask if k == "t" else v for k, v in features.items()}

    def encode_modalities(self, features, labels=None, apply_text_dropout=True):
        if getattr(self, "fusion_mode", "hierarchical") != "hierarchical":
            raise ValueError("encode_modalities is only defined for hierarchical fusion mode")

        if apply_text_dropout:
            features = self._apply_text_dropout(features)
        projected = {}
        quality_scores = {}

        base_feature = features[list(features.keys())[0]]
        batch_size = base_feature.size(0)
        device = base_feature.device

        for m in self.modalities:
            if m in features:
                projected[m] = self.proj[m](features[m])
                projected[m] = self.feature_selectors[m](projected[m])
                quality_scores[m] = self.assess_modality_quality(features[m], m)
            else:
                projected[m] = torch.zeros(batch_size, self.fusion_dim, device=device)
                quality_scores[m] = torch.tensor(0.1, device=device)

        projected_raw = None
        if getattr(self, "aux_from_raw_projected", False):
            projected_raw = {m: projected[m].clone() for m in self.modalities}

        if len(self.modalities) > 1 and hasattr(self, "feature_aligners"):
            aligned_features = {}
            for i, m1 in enumerate(self.modalities):
                aligned_features[m1] = projected[m1]
                for m2 in self.modalities[i + 1 :]:
                    key = f"{m1}_{m2}"
                    if key in self.feature_aligners:
                        aligned, _ = self.feature_aligners[key](projected[m1], projected[m2])
                        aligned_features[m1] = (
                            aligned_features[m1] * self.align_self_weight
                            + aligned * (1.0 - self.align_self_weight)
                        )
            projected = aligned_features

        modality_list = [projected[m] for m in self.modalities]

        cc_bias = None
        if getattr(self, "use_class_conditional", False):
            if labels is not None:
                cc_bias = self.class_modality_bias[labels]
            else:
                concat_all = torch.cat(modality_list, dim=-1)
                pred_logits = self.class_predictor(concat_all)
                pred_probs = F.softmax(pred_logits, dim=-1)
                cc_bias = pred_probs @ self.class_modality_bias

        projected_pre_gate = None
        if getattr(self, "aux_from_projected", False):
            projected_pre_gate = {m: projected[m].clone() for m in self.modalities}

        if self.use_tguided_router_fusion:
            block_out = self.router_fusion_block(projected, modality_bias=cc_bias)
            self._fusion_weights = block_out["route_weights"]
            self._context_weights = block_out["context_weights"]
            gated = block_out["refined"]
        else:
            fused_context, fusion_weights = self.adaptive_fusion(modality_list, bias=cc_bias)
            self._fusion_weights = fusion_weights
            self._context_weights = None

            gated = {}
            if getattr(self, "merge_info_gate", False):
                for m in self.modalities:
                    if isinstance(self.gates[m], nn.Sequential) and isinstance(self.gates[m][0], nn.Identity):
                        gated[m] = projected[m]
                    else:
                        gated[m] = self.gates[m](projected[m], fused_context)
            elif getattr(self, "use_aitm", False):
                for m in self.modalities:
                    projected[m] = self.info_gates[m](projected[m], fused_context)
                for m in ["t"]:
                    combined = torch.cat([projected[m], fused_context], dim=-1)
                    if isinstance(self.gates[m], nn.Sequential) and isinstance(self.gates[m][0], nn.Identity):
                        gated[m] = projected[m]
                    else:
                        gate = self.gates[m](combined)
                        gated[m] = projected[m] * gate + projected[m] * 0.1
                for m in ["a", "v"]:
                    combined = torch.cat([projected[m], gated["t"], fused_context], dim=-1)
                    if isinstance(self.gates[m], nn.Sequential) and isinstance(self.gates[m][0], nn.Identity):
                        gated[m] = projected[m]
                    else:
                        gate = self.gates[m](combined)
                        gated[m] = projected[m] * gate + projected[m] * 0.1
            else:
                for m in self.modalities:
                    projected[m] = self.info_gates[m](projected[m], fused_context)
                for m in self.modalities:
                    combined = torch.cat([projected[m], fused_context], dim=-1)
                    if isinstance(self.gates[m], nn.Sequential) and isinstance(self.gates[m][0], nn.Identity):
                        gated[m] = projected[m]
                    else:
                        gate = self.gates[m](combined)
                        gated[m] = projected[m] * gate + projected[m] * 0.1

        return {
            "gated": gated,
            "projected_raw": projected_raw,
            "projected_pre_gate": projected_pre_gate,
            "quality_scores": quality_scores,
        }

    def classify_from_gated(
        self,
        gated,
        projected_raw=None,
        projected_pre_gate=None,
        return_attention=False,
        return_hidden=False,
        labels=None,
    ):
        cross_out = []
        attentions = {}

        if len(self.modalities) > 1 and self.cross_attn is not None:
            for q_mod in self.modalities:
                others = []
                for m in self.modalities:
                    if m != q_mod:
                        feat = gated[m].mean(dim=1) if len(gated[m].shape) == 3 else gated[m]
                        others.append(feat)
                if others:
                    kv = torch.stack(others, dim=1)
                    q_seq = gated[q_mod].unsqueeze(1) if len(gated[q_mod].shape) == 2 else gated[q_mod]
                    if isinstance(self.cross_attn[q_mod], nn.Sequential) and isinstance(
                        self.cross_attn[q_mod][0], IdentityAttention
                    ):
                        attn_out, attn_weights = q_seq, None
                    else:
                        attn_out, attn_weights = self.cross_attn[q_mod](query=q_seq, key=kv, value=kv)
                    attentions[q_mod] = attn_weights
                    cross_out.append(attn_out + q_seq * 0.2)
                else:
                    q_seq = gated[q_mod].unsqueeze(1) if len(gated[q_mod].shape) == 2 else gated[q_mod]
                    cross_out.append(q_seq)
                    attentions[q_mod] = None
        else:
            for m in self.modalities:
                mod_feat = gated[m].unsqueeze(1) if len(gated[m].shape) == 2 else gated[m]
                cross_out.append(mod_feat)
                attentions[m] = None

        fused_rep = torch.cat(cross_out, dim=1)
        transformed = self.transformer_encoder(fused_rep)
        batch_size = transformed.size(0)
        q = self.pool_queries.expand(batch_size, -1, -1)
        pooled, pool_weights = self.pool(q, transformed, transformed)
        pooled_flat = pooled.reshape(batch_size, -1)
        features_fused = self.feature_integrator(pooled_flat)

        early_modalities = []
        for modality in self.modalities:
            modality_tap = gated[modality]
            if modality_tap.ndim == 3:
                modality_tap = modality_tap.mean(dim=1)
            early_modalities.append(modality_tap)
        self._layerwise_taps = {
            "early": torch.stack(early_modalities, dim=1).mean(dim=1),
            "middle": transformed.mean(dim=1),
            "late": features_fused,
        }
        routes = (
            self.layerwise_spec.routes
            if self.layerwise_spec is not None
            else ("late", "late", "late")
        )
        all_logits = [
            classifier(self._layerwise_taps[route])
            for classifier, route in zip(self.classifiers, routes)
        ]
        stacked = torch.stack(all_logits)
        self._head_logits = stacked
        logits = stacked.mean(dim=0)
        self._head_variance = stacked.var(dim=0).mean(dim=-1)

        if projected_raw is not None:
            aux_source = projected_raw
        elif projected_pre_gate is not None:
            aux_source = projected_pre_gate
        else:
            aux_source = gated

        aux_logits = {}
        for m in self.modalities:
            mod_feat = aux_source[m]
            if len(mod_feat.shape) == 3:
                mod_feat = mod_feat.mean(dim=1)
            aux_logits[m] = self.mod_classifiers[m](mod_feat)

        contrastive_loss = None
        if self.training and getattr(self, "use_contrastive", False) and labels is not None:
            contrastive_loss = self.contrastive(features_fused, labels)

        if return_attention:
            result = (logits, aux_logits, attentions, pool_weights, contrastive_loss)
            return result + (features_fused,) if return_hidden else result

        if contrastive_loss is not None:
            result = (logits, aux_logits, contrastive_loss)
            return result + (features_fused,) if return_hidden else result

        result = (logits, aux_logits)
        return result + (features_fused,) if return_hidden else result

    def forward(self, features, return_attention=False, return_hidden=False, labels=None):
        features = self._apply_text_dropout(features)

        # ── Concat fusion shortcut ──
        if getattr(self, "fusion_mode", "hierarchical") == "concat":
            projected = {}
            base_feature = features[list(features.keys())[0]]
            for m in self.modalities:
                if m in features:
                    projected[m] = self.proj[m](features[m])
                    projected[m] = self.feature_selectors[m](projected[m])
                else:
                    projected[m] = torch.zeros(
                        base_feature.size(0), self.fusion_dim, device=base_feature.device
                    )
            concat_feats = torch.cat([projected[m] for m in self.modalities], dim=-1)
            features_fused = self.concat_fusion(concat_feats)

            all_logits = []
            for classifier in self.classifiers:
                all_logits.append(classifier(features_fused))
            stacked = torch.stack(all_logits)
            logits = stacked.mean(dim=0)
            self._head_variance = stacked.var(dim=0).mean(dim=-1)

            # Aux classifiers on projected features
            aux_logits = {}
            for m in self.modalities:
                aux_logits[m] = self.mod_classifiers[m](projected[m])

            contrastive_loss = None
            if self.training and getattr(self, 'use_contrastive', False) and labels is not None:
                contrastive_loss = self.contrastive(features_fused, labels)

            if return_attention:
                result = (logits, aux_logits, {}, None, contrastive_loss)
                return result + (features_fused,) if return_hidden else result
            if contrastive_loss is not None:
                result = (logits, aux_logits, contrastive_loss)
                return result + (features_fused,) if return_hidden else result
            return logits

        # ── Logit-gate fusion shortcut ──
        if getattr(self, "fusion_mode", "hierarchical") == "logit_gate":
            projected = {}
            base_feature = features[list(features.keys())[0]]
            for m in self.modalities:
                if m in features:
                    projected[m] = self.proj[m](features[m])
                    projected[m] = self.feature_selectors[m](projected[m])
                else:
                    projected[m] = torch.zeros(
                        base_feature.size(0), self.fusion_dim, device=base_feature.device
                    )
            # Per-modality logits
            mod_logits = {m: self.mod_classifiers[m](projected[m]) for m in self.modalities}
            T_logits = mod_logits['t']

            # Gate: concat all projected features → per-sample gate for each non-T modality
            concat_all = torch.cat([projected[m] for m in self.modalities], dim=-1)
            gates = self.logit_gate_net(concat_all)  # (B, M-1)

            # Apply corrections: final = T + Σ λ_m * (M_logits - T_logits)
            logits = T_logits
            gate_idx = 0
            for m in self.modalities:
                if m == 't':
                    continue
                lam = gates[:, gate_idx:gate_idx+1]  # (B, 1)
                delta = mod_logits[m] - T_logits
                logits = logits + lam * delta
                gate_idx += 1

            # Build pseudo 3-head output for compatibility
            stacked = torch.stack([logits, logits, logits])
            self._head_variance = torch.zeros_like(logits[:, 0])

            aux_logits = mod_logits
            contrastive_loss = None
            if self.training and getattr(self, 'use_contrastive', False) and labels is not None:
                contrastive_loss = self.contrastive(
                    torch.cat([projected[m] for m in self.modalities], dim=-1), labels)

            if return_attention:
                result = (stacked.mean(dim=0), aux_logits, {}, None, contrastive_loss)
                return result + (None,) if return_hidden else result
            if contrastive_loss is not None:
                result = (stacked.mean(dim=0), aux_logits, contrastive_loss)
                return result + (None,) if return_hidden else result
            return stacked.mean(dim=0)

        stage = self.encode_modalities(features, labels=labels, apply_text_dropout=False)
        return self.classify_from_gated(
            stage["gated"],
            projected_raw=stage["projected_raw"],
            projected_pre_gate=stage["projected_pre_gate"],
            return_attention=return_attention,
            return_hidden=return_hidden,
            labels=labels,
        )


# ── HWR H2L1 Refiner (from DFGCN) ──

class HWRReasoningBlock(nn.Module):
    """Shared update block for low-level latent state z and answer state y."""

    def __init__(self, hidden_dim: int, n_heads: int, dropout: float, expansion: int = 4):
        super().__init__()
        self.attn_norm = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(hidden_dim, n_heads, dropout=dropout, batch_first=True)
        self.attn_drop = nn.Dropout(dropout)
        self.ff_norm = nn.LayerNorm(hidden_dim)
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * expansion),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * expansion, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, state: torch.Tensor, injection: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        out = state + injection
        seq_len = out.size(1)
        causal_block = torch.triu(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=out.device),
            diagonal=1,
        )
        attn_in = self.attn_norm(out)
        attn_out, _ = self.attn(
            query=attn_in,
            key=attn_in,
            value=attn_in,
            attn_mask=causal_block,
            key_padding_mask=valid_mask <= 0,
            need_weights=False,
        )
        out = out + self.attn_drop(attn_out)
        out = out + self.ff(self.ff_norm(out))
        return torch.where(valid_mask.unsqueeze(-1).bool(), out, state)


class HWRH2L1Refiner(nn.Module):
    """HWR reflective adapter with fixed h=2, l=1.

    Inputs:
        hidden: [B, T, H] contextual hidden states (from fusion transformer).
        anchor_logits: [B, T, C] anchor emotion logits.
        valid_mask: [B, T] utterance mask.

    Output:
        final_logits: [B, T, C].
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        n_classes: int = 7,
        n_heads: int = 8,
        dropout: float = 0.1,
        residual_scale: float = 0.1,
        gate_temp: float = 0.25,
        deploy_gate_conf_drop: float | None = 0.1,
        neutral_class: int | None = 0,
    ):
        super().__init__()
        self.residual_scale = residual_scale
        self.gate_tau = math.log(n_classes) * 0.65
        self.gate_temp = gate_temp
        self.deploy_gate_conf_drop = deploy_gate_conf_drop
        self.neutral_class = neutral_class

        self.history_attn = nn.MultiheadAttention(hidden_dim, n_heads, dropout=dropout, batch_first=True)
        self.logit_to_answer = nn.Linear(n_classes, hidden_dim)
        self.memory_seed = nn.Linear(hidden_dim * 2, hidden_dim)
        self.objective_seed = nn.Linear(hidden_dim * 2, hidden_dim)
        self.reasoner = HWRReasoningBlock(hidden_dim, n_heads, dropout)
        self.latent_norm = nn.LayerNorm(hidden_dim)
        self.answer_norm = nn.LayerNorm(hidden_dim)
        self.objective_norm = nn.LayerNorm(hidden_dim)
        self.out = nn.Linear(hidden_dim, n_classes)

    def read_history(self, hidden: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        seq_len = hidden.size(1)
        block_current_and_future = torch.triu(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=hidden.device),
            diagonal=0,
        )
        block_current_and_future[0, 0] = False
        memory, _ = self.history_attn(
            query=hidden,
            key=hidden,
            value=hidden,
            attn_mask=block_current_and_future,
            key_padding_mask=valid_mask <= 0,
            need_weights=False,
        )
        has_history = torch.arange(seq_len, device=hidden.device).view(1, seq_len) > 0
        keep = (valid_mask > 0) & has_history
        return torch.nan_to_num(memory) * keep.unsqueeze(-1).to(hidden.dtype)

    def entropy_gate(self, logits: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=-1)
        entropy = -(probs * probs.clamp_min(1e-8).log()).sum(dim=-1, keepdim=True)
        gate = torch.sigmoid((entropy - self.gate_tau) / self.gate_temp)
        return gate * valid_mask.unsqueeze(-1).to(logits.dtype)

    def run_h_cycle(
        self,
        latent: torch.Tensor,
        answer: torch.Tensor,
        objective: torch.Tensor,
        gate: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        latent = self.latent_norm(self.reasoner(latent, objective + gate * answer, valid_mask))
        answer = self.answer_norm(self.reasoner(answer, objective + gate * latent, valid_mask))
        return latent, answer

    def deploy_gate(self, anchor_logits, refined_logits, valid_mask):
        if self.deploy_gate_conf_drop is None and self.neutral_class is None:
            block = torch.zeros_like(valid_mask, dtype=torch.bool).unsqueeze(-1)
            return refined_logits, block

        anchor_probs = F.softmax(anchor_logits, dim=-1)
        refined_probs = F.softmax(refined_logits, dim=-1)
        anchor_pred = anchor_probs.argmax(dim=-1)
        refined_pred = refined_probs.argmax(dim=-1)
        changed = (anchor_pred != refined_pred) & (valid_mask > 0)

        if self.neutral_class is None:
            neutral_shift = torch.zeros_like(changed)
        else:
            previous_pred = torch.roll(anchor_pred, shifts=1, dims=1)
            previous_valid = torch.roll(valid_mask > 0, shifts=1, dims=1)
            previous_valid[:, 0] = False
            neutral_shift = changed & (anchor_pred == self.neutral_class) & previous_valid & (anchor_pred != previous_pred)

        if self.deploy_gate_conf_drop is None:
            conf_drop = torch.zeros_like(changed)
        else:
            conf_gain = refined_probs.max(dim=-1).values - anchor_probs.max(dim=-1).values
            conf_drop = changed & (conf_gain < -float(self.deploy_gate_conf_drop))

        block = (neutral_shift | conf_drop).unsqueeze(-1)
        return torch.where(block, anchor_logits, refined_logits), block

    def forward(self, hidden: torch.Tensor, anchor_logits: torch.Tensor, valid_mask: torch.Tensor):
        memory = self.read_history(hidden, valid_mask)
        objective = self.objective_norm(self.objective_seed(torch.cat([hidden, memory], dim=-1)))
        latent = self.latent_norm(self.memory_seed(torch.cat([memory, hidden], dim=-1)))
        answer = self.answer_norm(hidden + self.logit_to_answer(anchor_logits))
        gate = self.entropy_gate(anchor_logits, valid_mask)

        # h-cycle 1
        latent, answer = self.run_h_cycle(latent, answer, objective, gate, valid_mask)
        delta = self.out(answer) * self.residual_scale
        step1_logits = anchor_logits + gate * delta

        # h-cycle 2
        latent, answer = self.run_h_cycle(latent, answer, objective, gate, valid_mask)
        delta = self.out(answer) * self.residual_scale
        refined_logits = anchor_logits + gate * delta

        final_logits, deploy_block = self.deploy_gate(anchor_logits, refined_logits, valid_mask)
        return final_logits, {
            "anchor_logits": anchor_logits,
            "step1_logits": step1_logits,
            "gate": gate,
            "delta_logits": final_logits - anchor_logits,
        }


class FusionWithHWR(nn.Module):
    """HierarchicalAttentionFusion + HWR H2L1 post-hoc refinement.

    Forward accepts utterance-level features or dialogue-level features (with _mask).
    When _mask is present, HWR is applied after fusion to refine logits using
    dialogue history context.
    """

    def __init__(self, fusion_model, hwr_refiner):
        super().__init__()
        self.fusion = fusion_model
        self.hwr = hwr_refiner
        self.modalities = fusion_model.modalities

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.fusion, name)

    def forward(self, features, return_attention=False, labels=None):
        mask = features.get('_mask', None)
        if mask is not None and mask.any():
            return self._forward_hwr(features, mask, labels)
        return self.fusion(features, return_attention=return_attention, labels=labels)

    def _forward_hwr(self, features, mask, labels=None):
        B, T = mask.shape
        device = mask.device

        # 1. Flatten utterance features for fusion
        flat_features = {}
        for m in self.modalities:
            if m in features:
                flat_features[m] = features[m][mask]  # (total_N, Dm)

        # 2. Fusion forward → hidden + anchor
        fusion_out = self.fusion(flat_features, return_hidden=True, labels=labels)
        main_logits, aux_logits = fusion_out[:2]
        hidden = fusion_out[-1]  # features_fused, (total_N, 256)

        # 3. Group back to dialogue level
        anchor_dia = torch.zeros(B, T, main_logits.size(-1), device=device)
        hidden_dia = torch.zeros(B, T, hidden.size(-1), device=device)
        flat_idx = mask.nonzero(as_tuple=True)
        anchor_dia[flat_idx[0], flat_idx[1]] = main_logits
        hidden_dia[flat_idx[0], flat_idx[1]] = hidden

        # 4. HWR refinement
        valid_mask = mask.float()
        refined_dia, _ = self.hwr(hidden_dia, anchor_dia, valid_mask)
        # refined_dia: (B, T, 7)

        # 5. Flatten valid utterances back
        refined_flat = refined_dia[mask]  # (total_N, 7)

        return refined_flat, aux_logits

    def disable_gates(self):
        self.fusion.disable_gates()

    def disable_channel_attention(self):
        self.fusion.disable_channel_attention()


class PostFusionStateRefiner(nn.Module):
    """Causal state-memory refiner over dialogue-level fused hidden states.

    The current utterance prediction only reads the previous dialogue state.
    The state is updated afterward using the current fused hidden and anchor
    prediction, so the refinement stays causal.
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        n_classes: int = 7,
        state_dim: int | None = None,
        dropout: float = 0.1,
        residual_scale: float = 0.15,
        gate_temp: float = 0.35,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_classes = n_classes
        self.state_dim = state_dim or hidden_dim
        self.residual_scale = residual_scale
        self.gate_tau = math.log(n_classes) * 0.65
        self.gate_temp = gate_temp

        self.logit_to_state = nn.Linear(n_classes, self.state_dim)
        self.state_cell = nn.GRUCell(hidden_dim + self.state_dim, self.state_dim)
        self.state_dropout = nn.Dropout(dropout)
        self.state_to_hidden = nn.Sequential(
            nn.Linear(self.state_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.refine_norm = nn.LayerNorm(hidden_dim)
        self.delta_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_classes),
        )

    def entropy_gate(self, anchor_probs: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        entropy = -(anchor_probs * anchor_probs.clamp_min(1e-8).log()).sum(dim=-1, keepdim=True)
        gate = torch.sigmoid((entropy - self.gate_tau) / self.gate_temp)
        return gate * valid_mask.unsqueeze(-1).to(anchor_probs.dtype)

    def forward(self, hidden: torch.Tensor, anchor_logits: torch.Tensor, valid_mask: torch.Tensor):
        valid_mask = valid_mask.to(hidden.dtype)
        anchor_probs = F.softmax(anchor_logits, dim=-1)
        gate = self.entropy_gate(anchor_probs, valid_mask)

        batch_size, seq_len, _ = hidden.shape
        state = hidden.new_zeros(batch_size, self.state_dim)
        refined_hidden_steps = []

        for t in range(seq_len):
            valid_t = valid_mask[:, t : t + 1]
            state_context = self.state_to_hidden(self.state_dropout(state))
            refined_hidden_t = self.refine_norm(hidden[:, t, :] + state_context)
            refined_hidden_t = torch.where(valid_t.bool(), refined_hidden_t, hidden[:, t, :])
            refined_hidden_steps.append(refined_hidden_t)

            state_input = torch.cat([hidden[:, t, :], self.logit_to_state(anchor_probs[:, t, :])], dim=-1)
            next_state = self.state_cell(state_input, state)
            state = torch.where(valid_t.bool(), next_state, state)

        refined_hidden = torch.stack(refined_hidden_steps, dim=1)
        delta_logits = self.delta_head(refined_hidden) * self.residual_scale
        refined_logits = anchor_logits + gate * delta_logits
        refined_logits = torch.where(valid_mask.unsqueeze(-1).bool(), refined_logits, anchor_logits)

        return refined_logits, {
            "anchor_logits": anchor_logits,
            "gate": gate,
            "delta_logits": gate * delta_logits,
            "refined_hidden": refined_hidden,
        }


class FusionWithPostState(nn.Module):
    """HierarchicalAttentionFusion + post-fusion causal state refinement."""

    def __init__(self, fusion_model, state_refiner):
        super().__init__()
        self.fusion = fusion_model
        self.state_refiner = state_refiner
        self.modalities = fusion_model.modalities
        self._last_state_info = None

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.fusion, name)

    def forward(self, features, return_attention=False, return_hidden=False, labels=None):
        mask = features.get("_mask", None)
        if mask is not None and mask.any():
            return self._forward_post_state(features, mask, return_attention, return_hidden, labels)
        return self.fusion(
            features,
            return_attention=return_attention,
            return_hidden=return_hidden,
            labels=labels,
        )

    def _forward_post_state(self, features, mask, return_attention=False, return_hidden=False, labels=None):
        batch_size, seq_len = mask.shape
        device = mask.device

        flat_features = {}
        for m in self.modalities:
            if m in features:
                flat_features[m] = features[m][mask]

        fusion_out = self.fusion(
            flat_features,
            return_attention=return_attention,
            return_hidden=True,
            labels=labels,
        )
        main_logits, aux_logits = fusion_out[:2]
        hidden = fusion_out[-1]
        extras = fusion_out[2:-1]

        anchor_dia = torch.zeros(batch_size, seq_len, main_logits.size(-1), device=device)
        hidden_dia = torch.zeros(batch_size, seq_len, hidden.size(-1), device=device)
        flat_idx = mask.nonzero(as_tuple=True)
        anchor_dia[flat_idx[0], flat_idx[1]] = main_logits
        hidden_dia[flat_idx[0], flat_idx[1]] = hidden

        refined_dia, state_info = self.state_refiner(hidden_dia, anchor_dia, mask.float())
        self._last_state_info = state_info

        refined_flat = refined_dia[mask]
        refined_hidden_flat = state_info["refined_hidden"][mask]

        result = (refined_flat, aux_logits, *extras)
        if return_hidden:
            return result + (refined_hidden_flat,)
        return result

    def disable_gates(self):
        self.fusion.disable_gates()

    def disable_channel_attention(self):
        self.fusion.disable_channel_attention()


class DialogueStateGRURefiner(nn.Module):
    """Causal dialogue-state refiner over gated modality features."""

    _CROSS_SWAP = {"t": "a", "a": "v", "v": "t"}

    def __init__(
        self,
        hidden_dim: int = 256,
        modalities=("v", "a", "t"),
        mode: str = "text",
        state_dim: int | None = None,
        dropout: float = 0.1,
        residual_scale: float = 0.15,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.modalities = tuple(modalities)
        self.mode = mode
        self.state_dim = state_dim or hidden_dim
        self.residual_scale = residual_scale

        if mode not in {"text", "modal", "cross"}:
            raise ValueError(f"Unsupported dialogue state mode: {mode}")

        if mode == "text":
            self.active_modalities = ("t",)
        else:
            self.active_modalities = tuple(m for m in self.modalities)

        self.state_cells = nn.ModuleDict(
            {m: nn.GRUCell(hidden_dim, self.state_dim) for m in self.active_modalities}
        )
        self.state_to_hidden = nn.ModuleDict(
            {
                m: nn.Sequential(
                    nn.Linear(self.state_dim, hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                )
                for m in self.active_modalities
            }
        )
        self.inject_gates = nn.ModuleDict(
            {
                m: nn.Sequential(
                    nn.Linear(hidden_dim * 2, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                    nn.Linear(hidden_dim, 1),
                    nn.Sigmoid(),
                )
                for m in self.active_modalities
            }
        )
        self.refine_norms = nn.ModuleDict(
            {m: nn.LayerNorm(hidden_dim) for m in self.active_modalities}
        )

    def _source_modality(self, target: str) -> str:
        if self.mode != "cross":
            return target
        source = self._CROSS_SWAP.get(target, target)
        return source if source in self.active_modalities else target

    def forward(self, hidden_by_modality, valid_mask):
        valid_mask = valid_mask.to(dtype=next(iter(hidden_by_modality.values())).dtype)
        batch_size, seq_len = valid_mask.shape

        refined = {m: feat.clone() for m, feat in hidden_by_modality.items()}
        states = {
            m: hidden_by_modality[m].new_zeros(batch_size, self.state_dim)
            for m in self.active_modalities
        }
        gate_trace = {
            m: hidden_by_modality[m].new_zeros(batch_size, seq_len, 1)
            for m in self.active_modalities
        }

        for t in range(seq_len):
            valid_t = valid_mask[:, t : t + 1]
            prev_states = {m: states[m] for m in self.active_modalities}

            for target in self.active_modalities:
                source = self._source_modality(target)
                state_context = self.state_to_hidden[target](prev_states[source])
                gate_input = torch.cat([hidden_by_modality[target][:, t, :], state_context], dim=-1)
                gate = self.inject_gates[target](gate_input) * valid_t
                gate_trace[target][:, t, :] = gate
                refined_t = self.refine_norms[target](
                    hidden_by_modality[target][:, t, :] + self.residual_scale * gate * state_context
                )
                refined[target][:, t, :] = torch.where(
                    valid_t.bool(),
                    refined_t,
                    hidden_by_modality[target][:, t, :],
                )

            for source in self.active_modalities:
                next_state = self.state_cells[source](hidden_by_modality[source][:, t, :], states[source])
                states[source] = torch.where(valid_t.bool(), next_state, states[source])

        info = {
            "mode": self.mode,
            "gate": gate_trace,
            "final_state": states,
            "active_modalities": self.active_modalities,
        }
        return refined, info


class FusionWithDialogueState(nn.Module):
    """HierarchicalAttentionFusion + causal dialogue state before cross-attention."""

    def __init__(self, fusion_model, state_refiner):
        super().__init__()
        self.fusion = fusion_model
        self.state_refiner = state_refiner
        self.modalities = fusion_model.modalities
        self._last_state_info = None

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.fusion, name)

    def forward(self, features, return_attention=False, return_hidden=False, labels=None):
        mask = features.get("_mask", None)
        if mask is not None and mask.any():
            return self._forward_dialogue_state(features, mask, return_attention, return_hidden, labels)
        return self.fusion(
            features,
            return_attention=return_attention,
            return_hidden=return_hidden,
            labels=labels,
        )

    def _forward_dialogue_state(self, features, mask, return_attention=False, return_hidden=False, labels=None):
        batch_size, seq_len = mask.shape

        flat_features = {}
        for m in self.modalities:
            if m in features:
                flat_features[m] = features[m][mask]

        stage = self.fusion.encode_modalities(flat_features, labels=labels)

        gated_dia = {}
        for m in self.modalities:
            gated_flat = stage["gated"][m]
            gated_seq = gated_flat.new_zeros(batch_size, seq_len, gated_flat.size(-1))
            gated_seq[mask] = gated_flat
            gated_dia[m] = gated_seq

        refined_dia, state_info = self.state_refiner(gated_dia, mask.float())
        self._last_state_info = state_info
        refined_flat = {m: refined_dia[m][mask] for m in self.modalities}

        return self.fusion.classify_from_gated(
            refined_flat,
            projected_raw=stage["projected_raw"],
            projected_pre_gate=stage["projected_pre_gate"],
            return_attention=return_attention,
            return_hidden=return_hidden,
            labels=labels,
        )

    def disable_gates(self):
        self.fusion.disable_gates()

    def disable_channel_attention(self):
        self.fusion.disable_channel_attention()


class MultitaskFusionLoss(nn.Module):
    def __init__(
        self,
        main_weight=0.6,
        aux_weights=None,
        poly_alpha=1.2,
        poly_gamma=1.2,
        ce_weight=None,
        contrastive_weight=0.1,
        loss_type="poly",
        focal_gamma=2.0,
        samples_per_class=None,
        cb_beta=0.9999,
        label_smoothing=0.0,
        aux_focal_weight=0.0,
        aux_focal_gamma=2.0,
        poly_focal_alpha=1.0,
        focal_reweight_gamma=0.0,
        broken_focal_gamma=0.0,
        conflict_reweight_eta=0.0,
        num_classes=None,
        prototype_weight=0.0,
        prototype_temperature=0.7,
        prototype_momentum=0.95,
        confusion_weight=0.0,
        confusion_margin=0.2,
        confusion_topk=2,
        neutral_weight=0.0,
        neutral_margin=0.2,
        neutral_index=0,
        normalize_aux_weights=True,
    ):
        super().__init__()
        self.main_weight = main_weight
        self.contrastive_weight = contrastive_weight
        self.loss_type = loss_type
        self.label_smoothing = label_smoothing
        self.aux_focal_weight = aux_focal_weight
        self.aux_focal = None
        if aux_focal_weight > 0:
            self.aux_focal = FocalLoss(gamma=aux_focal_gamma, alpha=None, reduction="mean")
        self.focal_reweight_gamma = focal_reweight_gamma
        self.broken_focal_gamma = broken_focal_gamma
        self.conflict_reweight_eta = conflict_reweight_eta
        self.poly_alpha = poly_alpha
        self.poly_gamma = poly_gamma
        self.num_classes = num_classes
        self.aux_weights = (
            {"t": 0.15, "a": 0.1, "v": 0.05}
            if aux_weights is None else dict(aux_weights)
        )
        aux_sum = sum(self.aux_weights.values())
        if normalize_aux_weights and aux_sum > 0:
            scale = (1 - main_weight) / aux_sum
            self.aux_weights = {k: v * scale for k, v in self.aux_weights.items()}
        if loss_type == "focal":
            self.ce_loss = FocalLoss(gamma=focal_gamma, alpha=ce_weight, reduction="mean")
        elif loss_type == "cb_focal":
            if samples_per_class is None:
                raise ValueError("samples_per_class required for cb_focal loss type")
            self.ce_loss = ClassBalancedFocalLoss(
                samples_per_class=samples_per_class, beta=cb_beta,
                gamma=focal_gamma, reduction="mean"
            )
        elif loss_type == "poly_focal":
            self.ce_loss = PolyFocalLoss(
                gamma=focal_gamma, poly_alpha=poly_focal_alpha, reduction="mean"
            )
        else:
            self.ce_loss = CompositePolyLoss(poly_alpha, poly_gamma, ce_weight, "mean")
        self.prototype_weight = prototype_weight
        self.prototype_temperature = prototype_temperature
        self.prototype_momentum = prototype_momentum
        self.confusion_weight = confusion_weight
        self.confusion_margin = confusion_margin
        self.confusion_topk = confusion_topk
        self.neutral_weight = neutral_weight
        self.neutral_margin = neutral_margin
        self.neutral_index = neutral_index
        self.requires_hidden = prototype_weight > 0 or confusion_weight > 0
        self.register_buffer("class_prototypes", torch.empty(0))
        if num_classes is not None:
            self.register_buffer(
                "prototype_initialized", torch.zeros(num_classes, dtype=torch.bool)
            )
            self.register_buffer(
                "prototype_counts", torch.zeros(num_classes, dtype=torch.long)
            )
        else:
            self.register_buffer("prototype_initialized", torch.zeros(0, dtype=torch.bool))
            self.register_buffer("prototype_counts", torch.zeros(0, dtype=torch.long))
        self._last_task_components = {}

    def _ensure_prototype_state(self, fused_hidden):
        if self.num_classes is None:
            raise ValueError("num_classes is required for task-aware prototype losses")
        if self.class_prototypes.numel() == 0:
            self.class_prototypes = fused_hidden.new_zeros(
                self.num_classes, fused_hidden.size(-1)
            )

    def _update_prototypes(self, fused_hidden, targets):
        if not self.requires_hidden or fused_hidden is None or fused_hidden.numel() == 0:
            return
        self._ensure_prototype_state(fused_hidden)
        with torch.no_grad():
            detached = fused_hidden.detach()
            for cls in targets.unique().tolist():
                cls_mask = targets == cls
                cls_mean = detached[cls_mask].mean(dim=0)
                if self.prototype_initialized[cls]:
                    self.class_prototypes[cls].mul_(self.prototype_momentum).add_(
                        cls_mean, alpha=1.0 - self.prototype_momentum
                    )
                else:
                    self.class_prototypes[cls].copy_(cls_mean)
                    self.prototype_initialized[cls] = True
                self.prototype_counts[cls] += int(cls_mask.sum().item())

    def _compute_prototype_loss(self, fused_hidden, targets):
        if (
            self.prototype_weight <= 0
            or fused_hidden is None
            or fused_hidden.numel() == 0
            or self.class_prototypes.numel() == 0
        ):
            return fused_hidden.new_zeros(()) if fused_hidden is not None else torch.tensor(0.0)

        valid = self.prototype_initialized
        if valid.sum() < 2:
            return fused_hidden.new_zeros(())

        target_valid = valid[targets]
        if not target_valid.any():
            return fused_hidden.new_zeros(())

        feats = F.normalize(fused_hidden[target_valid], dim=-1)
        protos = F.normalize(self.class_prototypes, dim=-1)
        sim = torch.matmul(feats, protos.t()) / self.prototype_temperature
        sim = sim.masked_fill(~valid.unsqueeze(0), -1e4)
        return F.cross_entropy(sim, targets[target_valid])

    def _compute_confusion_loss(self, main_logits, targets):
        if (
            self.confusion_weight <= 0
            or self.class_prototypes.numel() == 0
            or self.prototype_initialized.sum() < 2
        ):
            return main_logits.new_zeros(())

        valid = self.prototype_initialized
        protos = F.normalize(self.class_prototypes[valid], dim=-1)
        full_sim = main_logits.new_full((self.num_classes, self.num_classes), -1.0)
        valid_idx = valid.nonzero(as_tuple=False).squeeze(1)
        proto_sim = torch.matmul(protos, protos.t()).detach()
        full_sim[valid_idx.unsqueeze(1), valid_idx.unsqueeze(0)] = proto_sim

        relation = ((full_sim[targets] + 1.0) * 0.5).clamp_min(0.0)
        relation.scatter_(1, targets.unsqueeze(1), 0.0)
        relation = relation * valid.unsqueeze(0).float()

        if self.confusion_topk and self.confusion_topk > 0:
            topk = min(self.confusion_topk, relation.size(1) - 1)
            if topk > 0:
                top_vals, top_idx = relation.topk(topk, dim=1)
                top_mask = torch.zeros_like(relation)
                top_mask.scatter_(1, top_idx, 1.0)
                relation = relation * top_mask

        relation_sum = relation.sum(dim=1, keepdim=True)
        nonzero_rows = relation_sum.squeeze(1) > 0
        if not nonzero_rows.any():
            return main_logits.new_zeros(())

        weights = torch.zeros_like(relation)
        weights[nonzero_rows] = relation[nonzero_rows] / relation_sum[nonzero_rows].clamp_min(1e-8)
        target_logits = main_logits.gather(1, targets.unsqueeze(1))
        margins = F.relu(self.confusion_margin + main_logits - target_logits)
        target_mask = F.one_hot(targets, num_classes=main_logits.size(1)).float()
        margins = margins * (1.0 - target_mask)
        return (weights * margins).sum(dim=1).mean()

    def _compute_neutral_loss(self, main_logits, targets):
        if (
            self.neutral_weight <= 0
            or self.neutral_index is None
            or self.neutral_index < 0
            or self.neutral_index >= main_logits.size(1)
        ):
            return main_logits.new_zeros(())

        neutral_logits = main_logits[:, self.neutral_index]
        target_logits = main_logits.gather(1, targets.unsqueeze(1)).squeeze(1)
        is_neutral = targets == self.neutral_index
        per_sample = main_logits.new_zeros(targets.size(0))

        if is_neutral.any():
            other_logits = main_logits[is_neutral].clone()
            other_logits[:, self.neutral_index] = -1e4
            strongest_other = other_logits.max(dim=1).values
            per_sample[is_neutral] = F.relu(
                self.neutral_margin + strongest_other - neutral_logits[is_neutral]
            )

        non_neutral = ~is_neutral
        if non_neutral.any():
            per_sample[non_neutral] = F.relu(
                self.neutral_margin + neutral_logits[non_neutral] - target_logits[non_neutral]
            )

        return per_sample.mean()

    def forward(
        self,
        main_logits,
        aux_logits,
        targets,
        contrastive_loss=None,
        fused_alpha=None,
        fused_hidden=None,
    ):
        if aux_logits is None:
            return self.ce_loss(main_logits, targets)

        if self.training and self.requires_hidden and fused_hidden is not None:
            self._update_prototypes(fused_hidden, targets)

        # Per-sample main loss (needed for focal re-weighting)
        if self.focal_reweight_gamma > 0:
            ce_per_sample = F.cross_entropy(main_logits, targets, reduction="none")
            pt = torch.exp(-ce_per_sample)  # softmax prob of target class
            focal_weights = (1 - pt) ** self.focal_reweight_gamma  # NO detach
            if self.conflict_reweight_eta > 0 and aux_logits is not None and len(aux_logits) > 1:
                aux_probs = [F.softmax(logits.detach(), dim=-1) for logits in aux_logits.values()]
                mean_prob = torch.stack(aux_probs, dim=0).mean(dim=0).clamp_min(1e-8)
                conflict = torch.stack(
                    [
                        F.kl_div(
                            mean_prob.log(),
                            prob.clamp_min(1e-8),
                            reduction="none",
                        ).sum(dim=-1)
                        for prob in aux_probs
                    ],
                    dim=0,
                ).mean(dim=0)
                focal_weights = focal_weights * torch.exp(-self.conflict_reweight_eta * conflict)
            # PolyLoss per-sample: CE + alpha * (1-pt)^(poly_gamma+1)
            main_loss_per_sample = ce_per_sample + self.poly_alpha * (1 - pt) ** (self.poly_gamma + 1)
            focal_reweighted = (main_loss_per_sample * focal_weights).mean()
            main_loss = focal_reweighted
        else:
            main_loss = self.ce_loss(main_logits, targets)

        if self.broken_focal_gamma > 0:
            # Original baseline: PolyLoss × (1-p_t)^gamma with .detach()
            pt = F.softmax(main_logits, dim=-1)[range(len(targets)), targets].detach()
            focal_weights = (1 - pt) ** self.broken_focal_gamma
            smoothing = self.label_smoothing if self.label_smoothing > 0 else 0.1
            smoothed_targets = torch.zeros_like(main_logits).scatter_(1, targets.unsqueeze(1), 1.0)
            smoothed_targets = smoothed_targets * (1.0 - smoothing) + smoothing / smoothed_targets.size(1)
            log_probs = F.log_softmax(main_logits, dim=-1)
            smoothed_loss = -(smoothed_targets * log_probs).sum(dim=-1).mean()
            weighted_main_loss = (main_loss * focal_weights).mean() * 0.8 + smoothed_loss * 0.2
        elif self.label_smoothing > 0:
            smoothed_targets = torch.zeros_like(main_logits).scatter_(
                1, targets.unsqueeze(1), 1.0
            )
            smoothed_targets = (
                smoothed_targets * (1.0 - self.label_smoothing)
                + self.label_smoothing / smoothed_targets.size(1)
            )
            log_probs = F.log_softmax(main_logits, dim=-1)
            smoothed_loss = -(smoothed_targets * log_probs).sum(dim=-1).mean()
            weighted_main_loss = main_loss * 0.8 + smoothed_loss * 0.2
        else:
            weighted_main_loss = main_loss

        if self.aux_focal is not None:
            weighted_main_loss = weighted_main_loss + self.aux_focal_weight * self.aux_focal(main_logits, targets)

        # The effective historical objective was unit-weighted across tasks.
        # Encode L_main + L_T + L_A + L_V directly.
        total_loss = weighted_main_loss
        for m, logits in aux_logits.items():
            if m in self.aux_weights:
                total_loss += self.ce_loss(logits, targets)

        if contrastive_loss is not None:
            total_loss += self.contrastive_weight * contrastive_loss

        proto_loss = self._compute_prototype_loss(fused_hidden, targets)
        conf_loss = self._compute_confusion_loss(main_logits, targets)
        neutral_loss = self._compute_neutral_loss(main_logits, targets)

        if self.prototype_weight > 0:
            total_loss += self.prototype_weight * proto_loss
        if self.confusion_weight > 0:
            total_loss += self.confusion_weight * conf_loss
        if self.neutral_weight > 0:
            total_loss += self.neutral_weight * neutral_loss

        self._last_task_components = {
            "proto": float(proto_loss.detach().item()) if proto_loss.numel() else 0.0,
            "conf": float(conf_loss.detach().item()) if conf_loss.numel() else 0.0,
            "neutral": float(neutral_loss.detach().item()) if neutral_loss.numel() else 0.0,
        }

        return total_loss
