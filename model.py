from transformers import AutoTokenizer, AutoModel
import torch

class ClinicalBERT(torch.nn.Module):
    def __init__(self):
        super(ClinicalBERT, self).__init__()
        self.bert = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
        self.tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")

    def tokenize(self, text):
        tokens = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        return tokens["input_ids"], tokens["attention_mask"]

    def forward(self, input_ids, attention_mask):
        bert_output = self.bert(input_ids, attention_mask=attention_mask)
        
        last_hidden_state = bert_output.last_hidden_state
        cls_token = last_hidden_state[:, 0, :]

        return cls_token
    
if __name__ == "__main__":

    model = ClinicalBERT()
    text = "The patient has a fever and a cough."
    input_ids, attention_mask = model.tokenize(text)
    output = model(input_ids, attention_mask)
    print(output.shape)