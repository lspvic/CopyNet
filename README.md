# CopyNet Implementation with Tensorflow and nmt

CopyNet Paper: [Incorporating Copying Mechanism in Sequence-to-Sequence Learning](https://arxiv.org/abs/1603.06393).

CopyNet mechanism is wrapped with an exsiting RNN cell and used as an normal RNN cell.

Official [nmt](https://github.com/tensorflow/nmt) is also modified to enable CopyNet  mechanism.

## Vocabulary Setting

Since in copynet scenarios the target sequence contains words from source sentences, the best choice is to use a **shared vocabulary** for source vocabulary and target vcabulary. And we use **generated  vocabulary**, namely, the target vocabulary excluding  words from source sequences (copied words). In this codebase, `vocab_size` and `gen_vocab_size` are variables representing
shared vocabulary size and generated vocabulalry size.

## Usage

### 1. Use with tf.contrib.seq2seq

Just wrapper an any existing rnn cell(`BasicLSTMCell`, `AttentionWrapper` and so on).

```python
cell = any_rnn_cell

copynet_cell = CopyNetWrapper(cell, encoder_outputs, encoder_input_ids,
    vocab_size, gen_vocab_size)
decoder_initial_state = copynet_cell.zero_state(batch_size,
    tf.float32).clone(cell_state=decoder_initial_state)

helper = tf.contrib.seq2seq.TrainingHelper(...)
decoder = tf.contrib.seq2seq.BasicDecoder(copynet_cell, helper,
    decoder_initial_state, output_layer=None)
decoder_outputs, final_state, coder_seq_length = tf.contrib.seq2seq.dynamic_decode(decoder=decoder)
decoder_logits, decoder_ids = decoder_outputs
```

### 2. Use with tensorflow official nmt

Full nmt usages are in [nmt](https://github.com/tensorflow/nmt).

`--copynet` argument added to nmt command line to enable copy mechanism.

`--share_vocab` argument must be set.

`--gen_vocab_size` argument represents the size of generated vocabulary (excluding copy words from target vocabulary), if is not set, it equals the size of whole vocabulary.

```bash
python nmt.nmt.nmt.py --copynet --share_vocab --gen_vocab_size=2345 ...other_nmt_arguments
```

