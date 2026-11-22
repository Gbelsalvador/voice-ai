from __future__ import annotations

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class AdditiveAttention(nn.Module):
    def __init__(self, encoder_dim: int, decoder_dim: int, attention_dim: int = 128):
        super().__init__()
        self.key = nn.Linear(encoder_dim, attention_dim, bias=False)
        self.query = nn.Linear(decoder_dim, attention_dim, bias=False)
        self.energy = nn.Linear(attention_dim, 1, bias=False)

    def forward(self, encoder_outputs, decoder_state, text_lengths):
        # encoder_outputs [B, L, E], decoder_state [B, D]
        scores = self.energy(torch.tanh(self.key(encoder_outputs) + self.query(decoder_state).unsqueeze(1))).squeeze(-1)
        positions = torch.arange(encoder_outputs.size(1), device=encoder_outputs.device).unsqueeze(0)
        mask = positions >= text_lengths.unsqueeze(1)
        scores = scores.masked_fill(mask, -1e9)
        weights = torch.softmax(scores, dim=-1)
        context = torch.bmm(weights.unsqueeze(1), encoder_outputs).squeeze(1)
        return context, weights


class TTSModel(nn.Module):
    """Petit modèle texte->log-mel, entraîné intégralement depuis une initialisation aléatoire.

    Le décodeur autoregressif utilise une attention additive. Le waveform est ensuite
    reconstruit par Griffin-Lim (non appris) dans infer_tts.py.
    """
    def __init__(self, vocab_size: int, n_mels: int = 80, embedding_dim: int = 128, hidden: int = 128):
        super().__init__()
        self.n_mels = n_mels
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.encoder = nn.LSTM(embedding_dim, hidden, batch_first=True, bidirectional=True)
        enc_dim = hidden * 2
        self.init_h = nn.Linear(enc_dim, enc_dim)
        self.prenet = nn.Sequential(
            nn.Linear(n_mels, 128), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(128, 128), nn.ReLU(), nn.Dropout(0.5),
        )
        self.attention = AdditiveAttention(enc_dim, enc_dim)
        self.decoder_cell = nn.GRUCell(128 + enc_dim, enc_dim)
        self.mel_projection = nn.Linear(enc_dim + enc_dim, n_mels)
        self.stop_projection = nn.Linear(enc_dim + enc_dim, 1)

    def encode(self, text, text_lengths):
        x = self.embedding(text)
        packed = pack_padded_sequence(x, text_lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, _ = self.encoder(packed)
        enc, _ = pad_packed_sequence(packed_out, batch_first=True, total_length=text.size(1))
        mask = torch.arange(enc.size(1), device=enc.device).unsqueeze(0) < text_lengths.unsqueeze(1)
        pooled = (enc * mask.unsqueeze(-1)).sum(1) / text_lengths.clamp(min=1).unsqueeze(1)
        h = torch.tanh(self.init_h(pooled))
        return enc, h

    def decode_step(self, previous_mel, encoder_outputs, text_lengths, hidden):
        pre = self.prenet(previous_mel)
        context, attention = self.attention(encoder_outputs, hidden, text_lengths)
        hidden = self.decoder_cell(torch.cat([pre, context], dim=-1), hidden)
        joined = torch.cat([hidden, context], dim=-1)
        mel = self.mel_projection(joined)
        stop = self.stop_projection(joined).squeeze(-1)
        return mel, stop, hidden, attention

    def forward(self, text, text_lengths, target_mels):
        # target_mels [B, T, n_mels], teacher forcing intégral pendant l'apprentissage.
        enc, hidden = self.encode(text, text_lengths)
        previous = target_mels.new_zeros(target_mels.size(0), self.n_mels)
        mel_outputs, stop_outputs, attention_outputs = [], [], []
        for t in range(target_mels.size(1)):
            mel, stop, hidden, attn = self.decode_step(previous, enc, text_lengths, hidden)
            mel_outputs.append(mel)
            stop_outputs.append(stop)
            attention_outputs.append(attn)
            previous = target_mels[:, t]
        return torch.stack(mel_outputs, 1), torch.stack(stop_outputs, 1), torch.stack(attention_outputs, 1)

    @torch.no_grad()
    def generate(self, text, text_lengths, max_steps: int = 1000, stop_threshold: float = 0.5, min_steps: int = 20):
        self.eval()
        enc, hidden = self.encode(text, text_lengths)
        previous = enc.new_zeros(enc.size(0), self.n_mels)
        outputs, attentions = [], []
        for step in range(max_steps):
            mel, stop, hidden, attn = self.decode_step(previous, enc, text_lengths, hidden)
            outputs.append(mel)
            attentions.append(attn)
            previous = mel
            if step + 1 >= min_steps and torch.sigmoid(stop).item() >= stop_threshold:
                break
        return torch.stack(outputs, 1), torch.stack(attentions, 1)
