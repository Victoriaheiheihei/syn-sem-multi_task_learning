import torch
from torchtext import data

class BertField(data.Field):
    def __init__(self, fix_length=512, dtype=torch.long):
        self.fix_length = fix_length
        self.dtype = dtype
        self.is_target = False
        self.sequential = True
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')

    def preprocess(self, x):
        """Load a single example using this field, tokenizing if necessary.
        If `sequential=True`, the input will be tokenized. Then the input
        will be optionally lowercased and passed to the user-provided
        `preprocessing` Pipeline."""
        x = [[101]] + [self.tokenizer.encode(w.rstrip('\n'), add_special_tokens=False) for w in x] + [[102]]
        return x

    def _pad(self, tensors, padding_value=0, total_length=None):
        size = [len(tensors)] + [max(tensor.size(i) for tensor in tensors)
                                  for i in range(len(tensors[0].size()))]
        if total_length is not None:
            assert total_length >= size[1]
            size[1] = total_length
        out_tensor = tensors[0].data.new(*size).fill_(padding_value)
        for i, tensor in enumerate(tensors):
            out_tensor[i][[slice(0, i) for i in tensor.size()]] = tensor
        return out_tensor

    def pad(self, minibatch):
        """Pad a batch of examples using this field.
        Pads to self.fix_length if provided, otherwise pads to the length of
        the longest example in the batch. Prepends self.init_token and appends
        self.eos_token if those attributes are not None. Returns a tuple of the
        padded list and a list containing lengths of each example if
        `self.include_lengths` is `True` and `self.sequential` is `True`, else just
        returns the padded list. If `self.sequential` is `False`, no padding is applied.
        """
        lens = [min(self.fix_length, max(len(ids) for ids in seq)) for seq in minibatch]
        padded = [self._pad([torch.tensor(ids[:i]) for ids in seq], 0, i)
            for i, seq in zip(lens, minibatch)]
        padded = self._pad(padded, 0)
        return padded

    def build_vocab(self, *args, **kwargs):
        pass

    def numericalize(self, arr, device=None):
        var = arr.to(dtype=self.dtype, device=device)
        var = var.contiguous()
        return var
