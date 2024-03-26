# if you do not have transformers, please !pip install transformers
# dependencies:
#   transformers
#   torch
#   nltk
from transformers import BertTokenizer
from transformers import BertForSequenceClassification
from transformers import AdamW

# if you do not have torch, please refer to https://pytorch.org/ [INSTALL PYTORCH]
import torch
import re

from nltk.corpus import stopwords
import nltk

nltk.download("stopwords")
stop = stopwords.words("english")

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)


def encoding_process(_content):
    get_ids = []
    count = 0
    for text in _content:
        input_ids = tokenizer.encode(
            text,  # text is a phrase, i.e. "grew b 1965 watching loving thunderbirds..."
            add_special_tokens=True,
            max_length=256,
            pad_to_max_length=True,
            return_tensors="pt",
        )
        get_ids.append(input_ids)
        count = count + 1
        # if count < 2:
        # print(f'text: {text}, \ninput_ids: {input_ids}')

    get_ids = torch.cat(get_ids, dim=0)
    return get_ids


my_input = "I regret that I was not able to see this movie before, it seems to have something that is worth"
proc_input = my_input.lower()
proc_input = re.sub(r"[^\w\s]+", "", proc_input)
proc_input = proc_input.replace("<br />", "")
proc_input = " ".join([word for word in proc_input.split() if word not in (stop)])
tokens = encoding_process([proc_input])

model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=2,
    output_attentions=False,
    output_hidden_states=False,
)
optimizer = AdamW(model.parameters(), lr=2e-5)
checkpoint = torch.load("/app/imdb_bert.pth", map_location=torch.device("cpu"))
model.load_state_dict(checkpoint["model_state_dict"])
optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
model.eval()  # enabling the eval mode to test with new samples# .

# gives: SequenceClassifierOutput(loss=None, logits=tensor([[-1.5959,  1.6339]], grad_fn=<AddmmBackward0>), hidden_states=None, attentions=None)
outputs = model(tokens)
# however len(res)=1, so I take the first and only:
probs = outputs[0].softmax(1)
# executing argmax function to get the candidate label
target_names = ["Negative", "Positive"]
print(target_names[probs.argmax()])
