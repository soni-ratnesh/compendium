import dill
import torch
import spacy
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

model.load_state_dict(torch.load('./data/model.brain', map_location=device))


def predict(sentence, src_field, trg_field, model, max_len=50):
    model.eval()
    nlp = spacy.load('en')
    tokens = [token.text.lower() for token in nlp(sentence)]
    tokens = [src_field.init_token] + tokens + [src_field.eos_token]
    src_indexes = [src_field.vocab.stoi[token] for token in tokens]

    src_tensor = torch.LongTensor(src_indexes).unsqueeze(1)
    src_len = torch.LongTensor([len(src_indexes)])

    with torch.no_grad():
        encoder_outputs, hidden = model.encoder(src_tensor, src_len)

    mask = model.create_mask(src_tensor)
    trg_indexes = [trg_field.vocab.stoi[trg_field.init_token]]
    attentions = torch.zeros(max_len, 1, len(src_indexes))

    for i in range(max_len):

        trg_tensor = torch.LongTensor([trg_indexes[-1]])

        with torch.no_grad():
            output, hidden, attention = model.decoder(trg_tensor, hidden, encoder_outputs, mask)

        attentions[i] = attention
        pred_token = output.argmax(1).item()
        trg_indexes.append(pred_token)

        if pred_token == trg_field.vocab.stoi[trg_field.eos_token]:
            break

    trg_tokens = [trg_field.vocab.itos[i] for i in trg_indexes]

    return trg_tokens[1:], attentions[:len(trg_tokens) - 1]
