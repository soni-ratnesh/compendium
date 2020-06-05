import dill
import torch
from application.model.model import Encoder, Decoder, Attention, Seq2Seq

# setup device to cpu
device = torch.device('cpu')

# Load source and destination field
with open("./data/ARTICLE.Field", "rb")as f:
    ARTICLE = dill.load(f)
with open("./data/SUMMARY.Field", "rb")as f:
    SUMMARY = dill.load(f)

INPUT_DIM = len(ARTICLE.vocab)
OUTPUT_DIM = len(SUMMARY.vocab)
ENC_EMB_DIM = 256
DEC_EMB_DIM = 256
ENC_HID_DIM = 512
DEC_HID_DIM = 512
ENC_DROPOUT = 0.5
DEC_DROPOUT = 0.5
ARTICLE_PAD_IDX = ARTICLE.vocab.stoi[ARTICLE.pad_token]

attn = Attention(ENC_HID_DIM, DEC_HID_DIM)
enc = Encoder(INPUT_DIM, ENC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, ENC_DROPOUT)
dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, DEC_DROPOUT, attn)

model = Seq2Seq(enc, dec, ARTICLE_PAD_IDX, device)

