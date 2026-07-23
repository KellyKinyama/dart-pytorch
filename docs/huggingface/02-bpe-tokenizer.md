# 02 — HuggingFace BPE tokenizer

Pure-Dart loader/encoder/decoder for HuggingFace
[byte-level BPE](https://huggingface.co/learn/nlp-course/en/chapter6/5)
`tokenizer.json` files. Compatible with both:

- GPT-2 family (`distilgpt2`, `gpt2`, `gpt2-medium`, GPT-J).
- GPT-NeoX family (Pythia).

Both tokenizers implement the same algorithm — the differences are
purely which merge pairs they picked during training. The loader
handles both automatically.

Lives in
[`lib/core/data/hf_bpe_tokenizer.dart`](../../lib/core/data/hf_bpe_tokenizer.dart);
re-exported from [`package:dart_pytorch/dart_pytorch.dart`](../../lib/dart_pytorch.dart).

## API

```dart
final tok = HFBpeTokenizer.loadFile('models/pythia-160m/tokenizer.json');
final ids  = tok.encode('Once upon a time,');   // -> List<int>
final text = tok.decode(ids);                    // -> String
final eot  = tok.endOfTextId;                    // -> int? (<|endoftext|>)
final v    = tok.vocabSize;                      // -> int
```

Also constructable from an already-parsed JSON map via
`HFBpeTokenizer.fromJson(Map)`.

## What the algorithm does

Byte-level BPE turns UTF-8 text into token ids in three steps:

1. **UTF-8 encode** the input to a byte stream.
2. **Map each byte** through a 256→Unicode "byte to unicode" table
   (space `0x20` becomes `Ġ` = U+0120, newline becomes `Ċ` = U+010A,
   etc.) so BPE operates on a set of visible glyphs.
3. **Pre-split** on a fixed regex (contractions, letter runs, digit
   runs, punctuation runs, whitespace runs), then run BPE merges
   greedily within each span using the rank table from `tokenizer.json`.

Decoding inverts steps 3→2→1: concat tokens, invert the byte→unicode
map, UTF-8 decode.

The pre-token regex is verbatim from OpenAI's original `encoder.py`:

```
's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
```

Dart's `RegExp(..., unicode: true)` handles `\p{L}` / `\p{N}` natively.

## `tokenizer.json` format quirks

HF's `tokenizer.json` has evolved. Two versions coexist in the wild:

- **Older** (e.g. `distilgpt2` circa 2020): `model.type` is missing
  from the JSON, but `vocab` + `merges` are present.
- **Newer** (Pythia, LLaMA, GPT-J, etc.): `model.type == "BPE"` set
  explicitly.

The loader accepts both:

```dart
if (type != null && type != 'BPE') {
  throw ArgumentError('unsupported model type: $type');
}
```

## Special tokens

`added_tokens` in `tokenizer.json` map ids → literal strings that must
never be split by BPE. During `encode`, we do a longest-match greedy
split on the input against all added-token literals, then run BPE on
the non-matching spans. This is how `<|endoftext|>` (id 50256 for
GPT-2, 0 for Pythia) survives round-tripping.

`endOfTextId` is exposed as a convenience for stop-condition logic in
future generation improvements.

## Notes and gotchas

- The tokenizer is **byte-level**, not character-level. `encode(s)`
  never produces `<unk>` — every UTF-8 byte sequence has a
  representation, even if it's a long chain of single-byte tokens.
- Decoding tolerates arbitrary token id sequences; ids out of range
  emit `?` in the output rather than throwing (matches the HF
  reference behaviour).
- `Ġ` at the start of a token means "preceded by space." A common
  mistake is to strip it during a plain string join — always go
  through `decode()`.
- On 30-token generations the encode/decode cost is invisible next to
  even a single forward pass. No need for fancy optimisation.

## Tests

[`test/hf_bpe_tokenizer_test.dart`](../../test/hf_bpe_tokenizer_test.dart)
runs 6 checks against a real `models/pythia-160m/tokenizer.json` when
present (test is skipped otherwise so CI doesn't require the model):

- `endOfTextId == 0`
- `vocabSize == 50304`
- Round-trip on `"Hello, world!"` and unicode strings
- Special-token pass-through (`<|endoftext|>` produces its own id)
- Whitespace-prefix handling (`" the"` vs `"the"` give different ids)
