from dataloader import *
from model import *
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import time
import argparse
from utilize import *


def evaluate(model_name, projector_ckpt_path, struct_dir, qa_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load tokenizer and  base_model
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, local_files_only=True).to(device)

    # load projector weights
    projector = Projector().to(device)
    projector.load_state_dict(torch.load(projector_ckpt_path, map_location=device))

    # get embedding of special token
    kg_start_token = "<|reserved_special_token_0|>"
    kg_end_token = "<|reserved_special_token_1|>"
    start_token = "<|begin_of_text|>"
    kg_start_emb = model.get_input_embeddings()(torch.tensor([tokenizer.convert_tokens_to_ids(kg_start_token)], device=device)).unsqueeze(1)
    kg_end_emb = model.get_input_embeddings()(torch.tensor([tokenizer.convert_tokens_to_ids(kg_end_token)], device=device)).unsqueeze(1)
    start_emb = model.get_input_embeddings()(torch.tensor([tokenizer.convert_tokens_to_ids(start_token)], device=device)).unsqueeze(1)

    # load data
    prompt_template = """
    # Task: Based on the assembly knowledge graph information above, answer the question
    # Question
    <question>
    # Answer
    """
    test_dataset = GraphQADataset_test(struct_dir, qa_dir, prompt_template, model, tokenizer)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True)

    correct = 0
    wrong_pairs = []
    total_length = 0  
    total_count = 0   
    model.eval()
    projector.eval()
    with torch.no_grad():
        for i, batch in enumerate(test_dataloader):
            input_embeds = batch["input_embeds"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"]
            struct_embeds = batch['struct_embeds'].to(device)
            sem_embeds = batch['sem_embeds'].to(device)

            fused_embeds = projector(struct_embeds, sem_embeds)
            input_all = torch.cat([start_emb, kg_start_emb, fused_embeds, kg_end_emb, input_embeds], dim=1)
            input_all = input_all.to(dtype=model.dtype)

            total_length += input_all.size(1)  
            total_count += 1  

            outputs = model.generate(input_ids=None, inputs_embeds=input_all, attention_mask=attention_mask, max_new_tokens=50)

            generated_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)

            result = generated_texts[0]
            label = labels[0]
            result_norm = normalize(result)
            label_norm = normalize(label)

            if result_norm == label_norm:
                correct += 1
            else:
                wrong_pairs.append((label, result))

    avg_length = total_length / total_count if total_count > 0 else 0  
    print(f"\nTotal samples: {total_count}")
    print(f"Correct predictions: {correct}")
    print(f"Incorrect predictions: {len(wrong_pairs)}")
    print(f"Average input_all sequence length: {avg_length:.2f}")



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="/opt/data/private/linsusu/LLM/demo/weight/base_model/llama3-8b-4bit", help="Path or HF hub name of the model")
    parser.add_argument("--projector_ckpt_path", type=str, default="weight/projector/projector_weights.pt", help="Checkpoint path of projector weights")
    parser.add_argument("--struct_dir", type=str, default="nonexistent_data/test1/kg", help="Directory of structure embeddings")
    parser.add_argument("--qa_dir", type=str, default="nonexistent_data/test1/qa", help="Directory of QA pairs")
    args = parser.parse_args()

    start_time = time.time()
    evaluate(
        model_name=args.model_name,
        projector_ckpt_path=args.projector_ckpt_path,
        struct_dir=args.struct_dir,
        qa_dir=args.qa_dir
    )
    print(f"cost time: {time.time()-start_time:.2f}s")

    