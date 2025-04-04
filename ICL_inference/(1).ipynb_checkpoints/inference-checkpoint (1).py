from ICL_modules import functions
import torch

def inference_standard_template(
    prompt, # Fixed parameter sign
    label_space # Fixed parameter sign
):
    return [1/len(label_space)] * len(label_space)

def standard_ICL_inference(
    prompt: str,
    model: callable,
    tokenizer: callable,
    label_space: list[str],
    cache_empty: callable = torch.cuda.empty_cache(), 
    calibration_function: callable = None,
    return_hidden_state: bool = False,
    return_full_vocab_prob: bool = False
):
    with torch.no_grad():
        if cache_empty is not None:
            cache_empty()
        tknzd_data = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device) 
        result = model(tknzd_data, output_hidden_states = True)
        full_vocab_prob = result['logits'][0][-1].detach().to(torch.float).cpu().numpy()
        last_hidden_state = result.hidden_states[-1][-1][-1].detach().to(torch.float).cpu().numpy()
        tokenized_label_space = [tokenizer(label).input_ids[-1] for label in label_space]
        label_space_logits = [full_vocab_prob[token] for token in tokenized_label_space]
        label_space_prob = functions.softmax(label_space_logits)
        del tknzd_data
        del result
        if calibration_function is not None:
            ret = calibration_function(label_space_prob, full_vocab_prob, last_hidden_state)
        else:
            ret = label_space_prob
        if return_hidden_state:
            ret = (ret, last_hidden_state)
        if return_full_vocab_prob:
            if return_hidden_state:
                ret.append(full_vocab_prob)
            else:
                ret = (ret, full_vocab_prob)
        return ret
    
def batched_ICL_inference(
    prompt: list[str],
    model: callable,
    tokenizer: callable,
    label_space: list[str],
    cache_empty: callable = torch.cuda.empty_cache(), 
    batch_calibration_function: callable = None, 
    inside_calibration_function: callable = None,
):
    with torch.no_grad():
        ori_results = []
        count = 0
        for single_prompt in prompt:
            ori_results.append(standard_ICL_inference_with_torch_Causal_LM(
                prompt = single_prompt, 
                model = model, 
                tokenizer = tokenizer, 
                label_space = label_space, 
                cache_empty = cache_empty, 
                calibration_function = inside_calibration_function
            ))
            print("\r", end="")
            print("Process: {}%, {} in {}".format(
                int((count + 1) / len(prompt) * 100), 
                (count + 1), 
                len(prompt)
            ), ">>" * int((count) / len(prompt) * 32), end="")
            count += 1
        if batch_calibration_function is not None:
            return batch_calibration_function(ori_results)
        else:
            return ori_results