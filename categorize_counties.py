from transformers import BitsAndBytesConfig
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch 
import pandas as pd
from tqdm import tqdm
model_name = "Qwen/Qwen2.5-Coder-7B-Instruct"
LLM_OUTPUT_BACKUP = "county_mapping_backup.txt"
LLM_OUTPUT = "county_mapping.txt"
# assumes you have qwen 2.5 1b already downloaded
# using hpc cluster so will change to 7b instruct
tokenizer = AutoTokenizer.from_pretrained(model_name)
bnb_config = BitsAndBytesConfig(load_in_4bit=True)
qwen_model = AutoModelForCausalLM.from_pretrained(
    model_name, quantization_config=bnb_config, device_map="auto", local_files_only=True
)

def qwen_prompting(messages, batch_size=64):
    """
    messages: an array of strings. Each string is a message to ask the llm. Ex. ) What county does [school name] reside in North Carolina?
    batch_size: randomly selected. 
    """
    
    outputs_all = []

    for i in tqdm(range(0, len(messages), batch_size),desc="Batches"):
        batch = messages[i:i+batch_size]

        llm_prompts = [[
            {"role": "system", "content": "You are an AI assistant that takes a school in north carolina and finds out which county in north carolina does the school reside in. Only give me the answer like so Schoolname:County"},
            {"role": "user", "content": msg}
        ] for msg in batch]

        inputs = tokenizer.apply_chat_template(
            llm_prompts,
            add_generation_prompt=True,
            tokenize=True,
            padding=True,
            truncation=True,
            return_tensors="pt",
            return_dict=True
        ).to(qwen_model.device)

        with torch.inference_mode():
            outputs = qwen_model.generate(
                **inputs,
                max_new_tokens=75,
                do_sample=False,
                use_cache=True
            )

        generated = outputs[:, inputs["input_ids"].shape[1]:]
        decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
        outputs_all.extend(decoded)

    return outputs_all

def query(school_name):
    return f"{school_name} belongs to what county in North Carolina?"

if __name__ == "__main__":
    # this script will only run on the HPC, if you have a fast computer then go for it 
    data = pd.read_csv('school-data.csv')
    school_names = data['names'].unique().tolist()
    messages = []
    for school in school_names:
        if pd.notna(school):
            messages.append(query(school))
            
    qwens_answers = qwen_prompting(messages)
    print("erasing back up file..")
    print("writing to backup file ...")
    with open(LLM_OUTPUT_BACKUP,'w+') as file:
        for message in messages:
            file.write(message+"\n")
    print("wrote into backup file with no issues, now will move backup file into the main txt file")
    with open(LLM_OUTPUT,'w+') as file:
        for message in messages:
            file.write(message+"\n")
            
            
            
        



