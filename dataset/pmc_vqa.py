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
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2VLForConditionalGeneration, AutoProcessor
from transformers import set_seed

# %%
def parse_answer_prob(text):
    """Extracts predicted answer letter and confidence score from model output."""
    answer_match = re.search(r"answer is\s+([A-D])", text, re.IGNORECASE)
    confidence_match = re.search(r"confidence\s+(\d{1,3})", text, re.IGNORECASE)
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
    correct = 0
    total_conf = 0
    count = 0
    
    set_seed(1)
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

            if pred and pred.upper() == true_letter.upper():
                correct += 1

            total_conf += conf
            count += 1

        except Exception as e:
            print(f"Error at idx {idx}: {e}")
            continue

    accuracy = correct / count if count > 0 else 0
    avg_confidence = total_conf / count if count > 0 else 0
    print(f"\nEvaluated {count} examples.")
    print(f"Accuracy: {accuracy:.3f}")
    print(f"Average Confidence: {avg_confidence:.3f}")

    return accuracy, avg_confidence


# %%
img_dir = '/network/scratch/a/anita.kriz/vccrl-llm/data/PMC-VQA/images' #TODO
train_csv_path = '/network/scratch/a/anita.kriz/vccrl-llm/data/PMC-VQA/test_50_current.csv' #TODO

train_df = pd.read_csv(train_csv_path)

# Setup (DO NOT do model.to(device))
model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-2B-Instruct",
    torch_dtype="auto",
    device_map="auto"
)
processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")

device = torch.device("cuda:0")  #TODO


# %%
accuracy, avg_conf = evaluate_model_on_samples(model = model, processor = processor, train_df = train_df, img_dir = img_dir , num_samples = 50)

# results = []
# for temp in [1.0]:
#     accuracy, avg_conf = evaluate_model_on_samples(model = model, processor = processor, train_df = train_df, img_dir = img_dir , num_samples = 50, temperature = temp)
#     results.append({
#         "temperature": temp,
#         "accuracy": accuracy,
#         "avg_confidence": avg_conf
#     })

# save_path = "/network/scratch/a/anita.kriz/public-llms/results"
# os.makedirs(save_path, exist_ok=True)  # Create directory if it doesn't exist
# save_file = os.path.join(save_path, "eval_results.csv")

# df = pd.DataFrame(results)
# df.to_csv(save_file, index=False)
# print(f"\nResults saved to: {save_file}")
