import os
import json
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer, BertModel, logging
from torch.nn.utils.rnn import pad_sequence
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
os.environ["TOKENIZERS_PARALLELISM"] = "false"
logging.set_verbosity_error()  # ignore warning
from torch.utils.data import DataLoader


def get_bert_embeddings(names, model_name="/opt/data/private/linsusu/LLM/model/bert-base-uncased", device="cuda" if torch.cuda.is_available() else "cpu",
    method="cls"  # "cls" or "mean_pooling"
):
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name).to(device)
    model.eval()

    name2bert = {}

    with torch.no_grad():
        for name in names:
            inputs = tokenizer(name, return_tensors="pt", padding="max_length", truncation=True, max_length=30)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = model(**inputs)
            last_hidden = outputs.last_hidden_state  # [1, seq_len, hidden]
            
            if method == "cls":
                embedding = last_hidden[0, 0, :].cpu().tolist()
            elif method == "mean_pooling":
                attention_mask = inputs['attention_mask']
                mask = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
                summed = (last_hidden * mask).sum(1)
                summed_mask = mask.sum(1)
                embedding = (summed / summed_mask).squeeze(0).cpu().tolist()
            else:
                raise ValueError(f"Unknown method: {method}. Choose 'cls' or 'mean_pooling'.")

            name2bert[name] = embedding

    return name2bert


def load_name2embedding_dict(vocab_path='/opt/data/private/linsusu/LLM/data/kg/CVB0G_12.json'):
    """
    Load name-embedding mapping from a JSON file.
    """
    with open(vocab_path, 'r', encoding='utf-8') as f:
        vocab = json.load(f)

    name2embedding = {}

    # entity
    for ent in vocab.get("entities", []):
        name = ent["name"].lower() 
        embedding = ent["embedding"]
        name2embedding[name] = embedding

    # relation
    for rel in vocab.get("relations", []):
        name = rel["name"].lower()
        embedding = rel["embedding"]
        name2embedding[name] = embedding

    return name2embedding


def generate_input_embed_test(sample, prompt_template, model, tokenizer,device="cuda", max_length=512):

    struct_embedding = sample["struct_embedding"]  # [N, 64]
    sem_embedding = sample["sem_embedding"]        # [N, 768]
    kg_length = struct_embedding.shape[0] + 3  # <start_of_text> <kg_start> ... <kg_end>

    input_embeds_list = []
    attention_masks = []
    label_list = []
    struct_embeds_list = []
    sem_embeds_list = []
    question_list = []
    for question, answer in zip(sample["inputs"], sample["outputs"]):
        prompt_text = prompt_template.replace("<question>", question)
        inputs = tokenizer(prompt_text, truncation=True, max_length=max_length, add_special_tokens=False)["input_ids"]
        labels = answer

        input_ids = torch.tensor(inputs, dtype=torch.long).to(device)        
        inputs_embeds = model.get_input_embeddings()(input_ids).squeeze(0)  # [T, hidden_dim]

        attention_mask = [1] * (inputs_embeds.shape[0] + kg_length)
        attention_mask = torch.tensor(attention_mask)  # tensor[N+3+T]

        assert (inputs_embeds.shape[0] + kg_length) == attention_mask.shape[0], \
        f"Mismatch: inputs_embeds={inputs_embeds.shape[0]+ kg_length}, attention_mask={attention_mask.shape[0]}"
 
        input_embeds_list.append(inputs_embeds)
        attention_masks.append(attention_mask)
        label_list.append(labels)
        question_list.append(question)
        struct_embeds_list.append(struct_embedding)
        sem_embeds_list.append(sem_embedding)
    
    return{
        'struct_embeds': struct_embeds_list, #List of [N, 64] tensors
        'sem_embeds': sem_embeds_list,  # List of [N, 768] tensors
        'input_embeds': input_embeds_list, # List of [T, D] tensors
        'attention_mask': attention_masks,   # List of [N+3+T] tensors
        'labels': label_list,   # List of string 
        'question': question_list
    }

def collate_fn(batch):
    """
    Collate function for DataLoader to pad the sequences in the batch
    and ensure all tensors have the same size.
    """
    input_embeds = [item["input_embeds"] for item in batch]
    attention_mask = [item["attention_mask"] for item in batch]
    labels = [item["labels"] for item in batch]

    input_embeds_padded = pad_sequence(input_embeds, batch_first=True, padding_value=0.0)
    attn_mask_padded = pad_sequence(attention_mask, batch_first=True, padding_value=0)
    labels_padded = pad_sequence(labels, batch_first=True, padding_value=-100)  # labels pad用-100
    return {
        'input_embeds': input_embeds_padded,  # [B, T_max_input, D]
        'attention_mask': attn_mask_padded,   # [B, T_max_input]
        'labels': labels_padded               # [B, T_max_input]
    }
    


class GraphQADataset_test(torch.utils.data.Dataset):
    def __init__(self, struct_dir, qa_dir, prompt_template, model, tokenizer, method="cls"):
        self.samples = []
        self.tokenizer = tokenizer
        self.prompt_template = prompt_template
        self.model = model
        self.input_embeds = []
        self.attention_masks = []
        self.labels = []
        self.questions = []
        self.struct_embeds = []
        self.sem_embeds = []
        for fname in os.listdir(struct_dir):
            if not fname.endswith(".json"):
                continue

            struct_path = os.path.join(struct_dir, fname)
            qa_path = os.path.join(qa_dir, fname)
            if not os.path.exists(qa_path):
                continue

            # load struct embedding
            name2struct = load_name2embedding_dict(struct_path)
            names = list(name2struct.keys())

            # load sem embedding
            name2sem = get_bert_embeddings(names, method=method)

            struct_embeddings = [name2struct[name] for name in names]
            sem_embeddings = [name2sem[name] for name in names]

            # load QA pairs
            with open(qa_path, "r", encoding="utf-8") as f:
                qa_data = json.load(f)

            self.samples.append({
                "graph_id": fname,
                "struct_embedding": torch.tensor(struct_embeddings, dtype=torch.float32),  # tensor [n,64]
                "sem_embedding": torch.tensor(sem_embeddings, dtype=torch.float32),        # tensor [n,768]
                "inputs": [qa["input"] for qa in qa_data],
                "outputs": [qa["output"] for qa in qa_data]
            })
        for sample in self.samples:
            processed_data = generate_input_embed_test(
                sample,
                self.prompt_template,
                self.model,
                self.tokenizer,
            )

            for input_emb in processed_data['input_embeds']:
                self.input_embeds.append(input_emb)


            for mask in processed_data['attention_mask']:
                self.attention_masks.append(mask)

            for label in processed_data['labels']:
                self.labels.append(label)

            for question in processed_data['question']:
                self.questions.append(question) 

            for struct in processed_data['struct_embeds']:
                self.struct_embeds.append(struct)
            
            for sem in processed_data['sem_embeds']:
                self.sem_embeds.append(sem)

    def __len__(self):
        return len(self.input_embeds)

    def __getitem__(self, idx):

        return {
            'struct_embeds': self.struct_embeds[idx], #  [N, 64] 
            'sem_embeds': self.sem_embeds[idx],  # [N, 768] 
            'input_embeds': self.input_embeds[idx],  # [T, D]
            'attention_mask': self.attention_masks[idx],  # [T+N+3]
            'labels': self.labels[idx],  # [T+N+3]
            'question': self.questions[idx]
        }


if __name__ == '__main__':
    struct_dir = "/opt/data/private/linsusu/LLM/data/train_kg"
    qa_dir = "/opt/data/private/linsusu/LLM/data/train_all"
    model_name = '/opt/data/private/linsusu/LLM/model/llama3-8b-4bit'
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    prompt_template = """
    Above is the information about a knowledge graph.
    And here is the question about that:
    <question>
    """
    dataset = GraphQADataset(struct_dir=struct_dir, qa_dir=qa_dir, prompt_template=prompt_template, model=model, tokenizer=tokenizer, method="cls")
    

    print('okk')
    train_dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    for batch in train_dataloader:
        print(batch)