from prompt import build_downstream_prompt
import dataset
from vllm import LLM, SamplingParams
from engine import parse_answer_prob_vqa, compute_accuracy, compute_ece, compute_brier_score
from tqdm import tqdm


def run_csc(args, config):
    assert config.dataset.name == 'pmcvqa'
    assert 'Qwen2-VL-2B-Instruct' in config.model.downstream

    eval_dataset = dataset.get_dataset(
        args=args,
        config=config,
        split=config.dataset.split_names[1]
    )

    # model and params for csc
    llm = LLM(model=config.model.downstream, tensor_parallel_size=4, gpu_memory_utilization=0.8)
    sampling_params = SamplingParams(temperature=0.5, top_p=0.95, top_k=100, max_tokens=512, n=21)

    # eval loop
    answer_rec = []
    consist_rec = []
    avg_conf_rec = []
    for sample in tqdm(eval_dataset):
        question = sample['question']
        gt_answer = sample['gt_answer']
        options = sample['options']
        img_path = sample['image_path']

        # here the prompt is already CoT
        prompt_csc = build_downstream_prompt(
            dataset_name=config.dataset.name,
            question_text=question,
            option_list=options,
            hint_text=None)

        conversation = [
            {'role': 'user',
             'content':
                 [
                     {'type': 'text', 'text': prompt_csc},
                     {'type': 'image', 'image': img_path}
                 ]
             }
        ]

        outputs = llm.chat(conversation,
                           sampling_params=sampling_params,
                           use_tqdm=False,
                           chat_template_content_format='openai')
        # note that we set n in sampling params, so it samples n times
        first_sampled_ans = None
        cnt = 0
        prob_sum_nominator = 0
        prob_sum_denominator = 0
        for output_n in outputs[0].outputs:
            generated_text = output_n.text
            ans, prob = parse_answer_prob_vqa(generated_text)
            if first_sampled_ans is None:
                first_sampled_ans = ans
            else:
                if ans == first_sampled_ans:
                    cnt += 1
                    prob_sum_nominator += prob
                prob_sum_denominator += prob
        consist = cnt / (len(outputs[0].outputs) - 1)   # note to minus 1 here because we exclude the first sampled ans
        avg_conf = prob_sum_nominator / (prob_sum_denominator + 1e-12)  # avoid div by 0

        answer_rec.append(first_sampled_ans)
        consist_rec.append(consist)
        avg_conf_rec.append(avg_conf)

    print('Accuracy: ', compute_accuracy(eval_dataset["gt_answer"], answer_rec))
    print('ECE (CSC): ', compute_ece(eval_dataset["gt_answer"], answer_rec, consist_rec))
    print('Brier score (CSC): ', compute_brier_score(eval_dataset["gt_answer"], answer_rec, consist_rec))
    print('ECE (CSA): ', compute_ece(eval_dataset["gt_answer"], answer_rec, avg_conf_rec))
    print('Brier score (CSA): ', compute_brier_score(eval_dataset["gt_answer"], answer_rec, avg_conf_rec))
