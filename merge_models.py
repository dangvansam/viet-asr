import sentencepiece.sentencepiece_model_pb2 as model
import os

m1 = model.ModelProto()
with open('temp_tokenizer/24892724e6a249268d657127026e63fa_tokenizer.model', 'rb') as f:
    m1.ParseFromString(f.read())

m2 = model.ModelProto()
with open('data/vi_tokenizer.model', 'rb') as f:
    m2.ParseFromString(f.read())

existing = set(p.piece for p in m1.pieces)

added = 0
for p in m2.pieces:
    if p.piece not in existing and not (p.piece.startswith('<') and p.piece.endswith('>')):
        new_p = m1.pieces.add()
        new_p.piece = p.piece
        new_p.score = p.score
        new_p.type = p.type
        existing.add(p.piece)
        added += 1

print(f"Added {added} tokens.")
print(f"Total vocab size is now: {len(m1.pieces)}")

with open('data/merged_tokenizer.model', 'wb') as f:
    f.write(m1.SerializeToString())
