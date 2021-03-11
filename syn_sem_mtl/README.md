# OpenNMT Simple 1.1

1. For source side, we add '-vocab_file' option to expand Vocab.
2. In order to keep the same vocab to other exps likes amr parsing, we force specials to possess [unk_token, pad_token, init_token and eos_token].
3. Add Ensemble Translation.

