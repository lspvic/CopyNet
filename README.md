# CopyNet Tensorflow Implementation

CopyNet Paper: [Incorporating Copying Mechanism in Sequence-to-Sequence Learning](https://arxiv.org/abs/1603.06393).

CopyNet mechanism is wrapped with an exsiting RNN cell and used as an normal RNN cell.

Official [nmt](https://github.com/tensorflow/nmt) is also modified to enable CopyNet  mechanism.

## Usage

### 1. Use with tf.contrib.seq2seq

Just wrapper an any existing rnn cell(`BasicLSTMCell`, `AttentionWrapper` and so on).

```python
cell = any_rnn_cell

copynet_cell = CopyNetWrapper(cell, encoder_outputs, encoder_input_ids,
    encoder_vocab_size,decoder_vocab_size)
decoder_initial_state = copynet_cell.zero_state(batch_size,
    tf.float32).clone(cell_state=decoder_initial_state)

helper = tf.contrib.seq2seq.TrainingHelper(...)
decoder = tf.contrib.seq2seq.BasicDecoder(copynet_cell, helper,
    decoder_initial_state, output_layer=None)
decoder_outputs, final_state, coder_seq_length = tf.contrib.seq2seq.dynamic_decode(decoder=decoder)
decoder_logits, decoder_ids = decoder_outputs
```

### 2. Use with tensorflow official nmt

Just add `--copynet` argument to nmt command line, full nmt usage is in [nmt](https://github.com/tensorflow/nmt).

```bash
python nmt.nmt.nmt.py --copynet ...other_nmt_arguments
```

