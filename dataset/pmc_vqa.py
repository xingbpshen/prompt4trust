# %% [markdown]
# ### Check Performance of Qwen VQA model, zero-shot ###

# %%
import os
import re
import random
from PIL import Image
from tqdm import tqdm
import torch
import pandas as pd 
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2VLForConditionalGeneration, MllamaForConditionalGeneration, AutoProcessor
from transformers import set_seed

def parse_answer_prob(text):
    """Extracts predicted answer letter and confidence score from model output."""
    answer_match = re.search(r"answer is\s+([A-D])", text, re.IGNORECASE)
    if not answer_match:
        answer_match = re.search(r"Answer:\s*([A-D])", text, re.IGNORECASE)
    # Try format: "**Confidence: 90%**" or "Confidence: 90"
    confidence_match = re.search(r"confidence\s+(\d{1,3})", text, re.IGNORECASE)
    if not confidence_match:
        confidence_match = re.search(r"Confidence:\s*(\d{1,3})", text, re.IGNORECASE)
    if answer_match and confidence_match:
        pred = answer_match.group(1).upper()
        confidence = min(float(confidence_match.group(1)), 100.0) / 100
        return pred, confidence
    return None, 0.0

def get_answer_letter_from_text(answer_text, choices):
    answer_text = answer_text.strip().lower()
    for choice in choices:
        letter, choice_text = choice.split(":", 1)
        if choice_text.strip().lower() == answer_text:
            return letter.strip()
    return None 

def build_messages(image, question, choices):
    """Builds multimodal messages for the processor."""
    return [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image
                },
                {
                    "type": "text",
                    "text": (
                        "You are answering a multiple-choice question with four options (A, B, C, or D). "
                        "Clearly state your final answer and confidence in the following format only:\n\n"
                        "'In conclusion, the answer is {LETTER} with confidence {CONFIDENCE}.'\n\n"
                        "Replace {LETTER} with one of A, B, C, or D, and {CONFIDENCE} with a number between 0 and 100.\n"
                        "After giving the answer, explain your reasoning based on the image and the question."
                    )
                },
                {
                    "type": "text",
                    "text": f"{question}\n\nOptions:\n" + '\n'.join(choices) + f"\n\nAnswer:"
                }
            ]
        }
    ]

def evaluate_model_on_samples(model, processor, train_df, img_dir, device="cuda", num_samples=100, seed = 42, temperature=1.0):
    # Tracking everything
    total_attempts = 0
    total_correct = 0

    # Tracking only valid-format responses
    valid_count = 0
    valid_correct = 0
    valid_conf_sum = 0.0
    
    # set_seed(1)
    # random.seed(seed)
    sampled_indices = random.sample(range(len(train_df)), num_samples)

    for idx in tqdm(sampled_indices):
        try:
            fig_path = train_df.loc[idx, 'Figure_path']
            full_path = os.path.join(img_dir, fig_path)
            image = Image.open(full_path).convert("RGB")
            question = train_df.loc[idx, 'Question']
            answer_text = train_df.loc[idx, 'Answer']
            choices = [
                train_df.loc[idx, 'Choice A'],
                train_df.loc[idx, 'Choice B'],
                train_df.loc[idx, 'Choice C'],
                train_df.loc[idx, 'Choice D']
            ]

            true_letter = get_answer_letter_from_text(answer_text, choices)

            if not true_letter:
                continue  
            messages = build_messages(image, question, choices)
            text_prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    
            inputs = processor(
                text=[text_prompt],
                images=[image],
                padding=True,
                return_tensors="pt"
            )
            inputs = inputs.to(device) 

            output_ids = model.generate(**inputs, max_new_tokens= 256, temperature = 0.1, top_p = .95, do_sample = True)

            output_texts = processor.batch_decode(output_ids, skip_special_tokens=True)
        
            pred, conf = parse_answer_prob(output_texts[0])
            # Overall stats: counts any response
            total_attempts += 1
            if pred is not None and pred.upper() == true_letter.upper():
                total_correct += 1

            # only count if valid format parsed
            if pred is not None and conf > 0.0:
                valid_count += 1
                valid_conf_sum += conf
                if pred.upper() == true_letter.upper():
                    valid_correct += 1

        except Exception as e:
            print(f"Error at idx {idx}: {e}")
            continue

    print(f"\n--- OVERALL ---")
    print(f"Attempted: {total_attempts}")
    print(f"Correct (any format): {total_correct} ({100 * total_correct / total_attempts:.2f}%)")

    print(f"\n--- VALID FORMAT ONLY ---")
    print(f"Valid responses: {valid_count}")
    print(f"Accuracy (valid only): {100 * valid_correct / valid_count:.2f}%" if valid_count else "No valid responses")
    print(f"Avg confidence (valid only): {100 * valid_conf_sum / valid_count:.2f}%" if valid_count else "No valid responses")

 

# %%
img_dir = '/network/scratch/a/anita.kriz/vccrl-llm/data/PMC-VQA/images' #TODO
train_csv_path = '/network/scratch/a/anita.kriz/vccrl-llm/data/PMC-VQA/test_50.csv' #TODO

train_df = pd.read_csv(train_csv_path)

# Setup (DO NOT do model.to(device))

model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"

model = MllamaForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
processor = AutoProcessor.from_pretrained(model_id)

device = torch.device("cuda:0")  #TODO


# %%
evaluate_model_on_samples(model = model, processor = processor, train_df = train_df, img_dir = img_dir , num_samples = 50)

