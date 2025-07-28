import math
import random
import ast
import csv

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

MIN_SEQ_LEN = 5


class AKT_text(nn.Module):
    def __init__(
        self,
        n_questions,
        n_pid=0,
        d_model=256,
        d_fc=512,
        n_heads=8,
        dropout=0.05,
        shortcut=False,
        question_embedding_dim=1024,
        question_embeddings_path="/mnt/ceph_rbd/data/eedi/questions.csv",
    ):
        super().__init__()
        self.n_questions = n_questions
        self.q_embed = nn.Embedding(n_questions + 1, d_model)
        self.s_embed = nn.Embedding(2, d_model)

        # Add question text embedding projection
        self.question_embedding_dim = question_embedding_dim
        self.text_embed_proj = nn.Linear(question_embedding_dim, d_model)
        
        # Load question embeddings
        self.question_embeddings = self._load_question_embeddings(question_embeddings_path)

        if n_pid > 0:
            self.q_diff_embed = nn.Embedding(n_questions + 1, d_model)
            self.s_diff_embed = nn.Embedding(2, d_model)
            self.p_diff_embed = nn.Embedding(n_pid + 1, 1)

        self.n_heads = n_heads
        self.block1 = DTransformerLayer(d_model, n_heads, dropout)
        self.block2 = DTransformerLayer(d_model, n_heads, dropout)
        self.block3 = DTransformerLayer(d_model, n_heads, dropout)

        self.out = nn.Sequential(
            nn.Linear(d_model * 2, d_fc),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_fc, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
        )

        self.dropout_rate = dropout

    def _load_question_embeddings(self, path):
        """Load question embeddings from CSV file"""
        embeddings = {}
        with open(path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                question_id = int(row['QuestionId'])
                embedding_str = row['QuestionEmbedding']
                try:
                    embedding = ast.literal_eval(embedding_str)
                    embeddings[question_id] = torch.tensor(embedding, dtype=torch.float32)
                except (ValueError, SyntaxError):
                    # Create zero embedding if parsing fails
                    embeddings[question_id] = torch.zeros(self.question_embedding_dim, dtype=torch.float32)
        return embeddings

    def get_question_text_embeddings(self, q):
        """Get text embeddings for questions"""
        batch_size, seq_len = q.shape
        text_embeddings = torch.zeros(batch_size, seq_len, self.question_embedding_dim, 
                                    device=q.device, dtype=torch.float32)
        
        for b in range(batch_size):
            for s in range(seq_len):
                q_id = q[b, s].item()
                if q_id > 0 and q_id in self.question_embeddings:
                    text_embeddings[b, s] = self.question_embeddings[q_id].to(q.device)
        
        return text_embeddings

    def forward(self, q_emb, s_emb, lens, n=1):
        # AKT
        hq = self.block1(q_emb, q_emb, q_emb, lens, peek_cur=True, n=n)
        hs = self.block2(s_emb, s_emb, s_emb, lens, peek_cur=True, n=n)
        return self.block3(hq, hq, hs, lens, peek_cur=False, n=n)

    def predict(self, q, s, pid=None, n=1):
        # set prediction mask
        q = q.masked_fill(q < 0, 0)
        s = s.masked_fill(s < 0, 0)

        # Get question embeddings
        q_emb = self.q_embed(q)
        
        # Get text embeddings and project them
        text_emb = self.get_question_text_embeddings(q)
        text_emb_proj = self.text_embed_proj(text_emb)
        
        # Combine question embeddings with text embeddings
        q_emb = q_emb + text_emb_proj
        
        s_emb = self.s_embed(s) + q_emb

        if pid is not None:
            pid = pid.masked_fill(pid < 0, 0)
            p_diff = self.p_diff_embed(pid)

            q_diff_emb = self.q_diff_embed(q)
            q_emb += q_diff_emb * p_diff

            s_diff_emb = self.s_diff_embed(s) + q_diff_emb
            s_emb += s_diff_emb * p_diff

        seqlen = q.size(1) - n + 1
        h = self(
            q_emb[:, :seqlen, :],
            s_emb[:, :seqlen, :],
            (s[:, :seqlen] >= 0).sum(dim=1),
            n,
        )
        y = self.out(torch.cat([q_emb[:, n - 1 :, :], h], dim=-1)).squeeze(-1)

        if pid is not None:
            return y, h, (p_diff**2).sum() * 1e-5
        else:
            return y, h, 0.0

    def get_loss(self, q, s, pid=None):
        logits, _, reg_loss = self.predict(q, s, pid)
        masked_labels = s[s >= 0].float()
        masked_logits = logits[s >= 0]
        return (
            F.binary_cross_entropy_with_logits(
                masked_logits, masked_labels, reduction="mean"
            )
            + reg_loss
        )


class DTransformerLayer(nn.Module):
    def __init__(self, d_model, n_heads, dropout, kq_same=True):
        super().__init__()
        self.masked_attn_head = MultiHeadAttention(d_model, n_heads, kq_same)

        self.dropout_rate = dropout
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def device(self):
        return next(self.parameters()).device

    def forward(self, query, key, values, lens, peek_cur=False, n=1):
        # construct mask
        seqlen = query.size(1)
        mask = torch.ones(seqlen, seqlen).tril(0 if peek_cur else -1)
        skip_mask = ~torch.ones(seqlen - n + 1).diag(-n + 1).bool()
        mask = (mask.bool() & skip_mask)[None, None, :, :].to(self.device())

        # mask manipulation
        if self.training:
            mask = mask.expand(query.size(0), -1, -1, -1)

            for b in range(query.size(0)):
                # sample for each batch
                if lens[b] < MIN_SEQ_LEN:
                    # skip for short sequences
                    continue
                idx = random.sample(
                    range(lens[b] - 1), max(1, int(lens[b] * self.dropout_rate))
                )
                for i in idx:
                    mask[b, :, i + 1 :, i] = 0

        # apply transformer layer
        query_ = self.masked_attn_head(query, key, values, mask)
        query = query + self.dropout(query_)
        return self.layer_norm(query)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, kq_same=True, bias=True):
        super().__init__()
        self.d_model = d_model
        self.d_k = d_model // n_heads
        self.h = n_heads

        self.q_linear = nn.Linear(d_model, d_model, bias=bias)
        if kq_same:
            self.k_linear = self.q_linear
        else:
            self.k_linear = nn.Linear(d_model, d_model, bias=bias)
        self.v_linear = nn.Linear(d_model, d_model, bias=bias)

        self.out_proj = nn.Linear(d_model, d_model, bias=bias)
        self.gammas = nn.Parameter(torch.zeros(n_heads, 1, 1))
        torch.nn.init.xavier_uniform_(self.gammas)

    def forward(self, q, k, v, mask):
        bs = q.size(0)

        # perform linear operation and split into h heads
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)

        # transpose to get dimensions bs * h * sl * d_k
        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)

        # calculate attention using function we will define next
        v_ = attention(
            q,
            k,
            v,
            mask,
            self.gammas,
        )

        # concatenate heads and put through final linear layer
        concat = v_.transpose(1, 2).contiguous().view(bs, -1, self.d_model)

        output = self.out_proj(concat)

        return output


def attention(q, k, v, mask, gamma=None):
    # attention score with scaled dot production
    d_k = k.size(-1)
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    bs, head, seqlen, _ = scores.size()

    # include temporal effect
    if gamma is not None:
        x1 = torch.arange(seqlen).float().expand(seqlen, -1).to(gamma.device)
        x2 = x1.transpose(0, 1).contiguous()

        with torch.no_grad():
            scores_ = scores.masked_fill(mask == 0, -1e32)
            scores_ = F.softmax(scores_, dim=-1)

            distcum_scores = torch.cumsum(scores_, dim=-1)
            disttotal_scores = torch.sum(scores_, dim=-1, keepdim=True)
            position_effect = torch.abs(x1 - x2)[None, None, :, :]
            dist_scores = torch.clamp(
                (disttotal_scores - distcum_scores) * position_effect, min=0.0
            )
            dist_scores = dist_scores.sqrt().detach()

        gamma = -1.0 * F.softplus(gamma).unsqueeze(0)
        total_effect = torch.clamp((dist_scores * gamma).exp(), min=1e-5, max=1e5)

        scores *= total_effect

    # normalize attention score
    scores.masked_fill_(mask == 0, -1e32)
    scores = F.softmax(scores, dim=-1)
    scores = scores.masked_fill(mask == 0, 0)  # set to hard zero to avoid leakage

    # calculate output
    output = torch.matmul(scores, v)
    return output