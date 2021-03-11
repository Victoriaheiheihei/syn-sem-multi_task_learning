from bert_field import BertField

sent="I am a goof person ."
field= BertField()
encode_sent = field.preprocess(sent.split())
print(encode_sent)
print(field.tokenizer.convert_ids_to_tokens(sum(encode_sent, [])))