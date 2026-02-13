from transPolymer_tokenizer import PolymerSmilesTokenizer

psmiles = "*CC(*)(CC(C)O)C(=O)Oc1ccc(Oc2ccc([N+](=O)[O-])cc2)cc1"  # <- replace with your pSMILES
blocksize = 128

tokenizer = PolymerSmilesTokenizer.from_pretrained("roberta-base", max_len=blocksize)

enc = tokenizer(
    psmiles,
    add_special_tokens=True,
    max_length=blocksize,
    return_token_type_ids=False,
    padding="max_length",
    truncation=True,
    return_attention_mask=True,
)

print("tokens:", tokenizer.convert_ids_to_tokens(enc["input_ids"]))
print("input_ids:", enc["input_ids"])
print("attention_mask:", enc["attention_mask"])