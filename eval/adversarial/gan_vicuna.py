import json
import os
import openai
import re
import sys
import math
from copy import deepcopy
import random
import numpy as np
from transformers import LlamaForCausalLM, LlamaTokenizer
import time
import pdb
import torch

import abc
import gc
import json
import math
import os
import sys
import time
from typing import Iterable, Optional, Dict
import warnings
from tqdm import tqdm

import psutil
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    LlamaTokenizer,
    LlamaForCausalLM,
    AutoModel,
    AutoModelForSeq2SeqLM,
    T5Tokenizer,
    AutoConfig,
)
from transformers.generation.logits_process import (
    LogitsProcessorList,
    RepetitionPenaltyLogitsProcessor,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
)

from fastchat.conversation import get_conv_template, SeparatorStyle
from fastchat.model.model_adapter import (
    load_model,
    get_conversation_template,
    get_generate_stream_function,
)
from fastchat.modules.gptq import GptqConfig
from fastchat.modules.awq import AWQConfig
from fastchat.utils import is_partial_stop, is_sentence_complete, get_context_length


# tokenizer = LlamaTokenizer.from_pretrained("lmsys/vicuna-13b-v1.3")
# model = LlamaForCausalLM.from_pretrained("lmsys/vicuna-13b-v1.3", device_map="auto")

def prepare_logits_processor(
    temperature: float, repetition_penalty: float, top_p: float, top_k: int
) -> LogitsProcessorList:
    processor_list = LogitsProcessorList()
    # TemperatureLogitsWarper doesn't accept 0.0, 1.0 makes it a no-op so we skip two cases.
    if temperature >= 1e-5 and temperature != 1.0:
        processor_list.append(TemperatureLogitsWarper(temperature))
    if repetition_penalty > 1.0:
        processor_list.append(RepetitionPenaltyLogitsProcessor(repetition_penalty))
    if 1e-8 <= top_p < 1.0:
        processor_list.append(TopPLogitsWarper(top_p))
    if top_k > 0:
        processor_list.append(TopKLogitsWarper(top_k))
    return processor_list


@torch.inference_mode()
def generate_stream(
    model,
    tokenizer,
    params: Dict,
    device: str,
    context_len: int,
    stream_interval: int = 2,
    judge_sent_end: bool = False,
):
    # Read parameters
    prompt = params["prompt"]
    len_prompt = len(prompt)
    temperature = float(params.get("temperature", 1.0))
    repetition_penalty = float(params.get("repetition_penalty", 1.0))
    top_p = float(params.get("top_p", 1.0))
    top_k = int(params.get("top_k", -1))  # -1 means disable
    max_new_tokens = int(params.get("max_new_tokens", 256))
    echo = bool(params.get("echo", True))
    stop_str = params.get("stop", None)
    stop_token_ids = params.get("stop_token_ids", None) or []
    stop_token_ids.append(tokenizer.eos_token_id)

    logits_processor = prepare_logits_processor(
        temperature, repetition_penalty, top_p, top_k
    )
    input_ids = tokenizer(prompt).input_ids

    if model.config.is_encoder_decoder:
        max_src_len = context_len
    else:  # truncate
        max_src_len = context_len - max_new_tokens - 1

    input_ids = input_ids[-max_src_len:]
    output_ids = list(input_ids)
    input_echo_len = len(input_ids)

    if model.config.is_encoder_decoder:
        encoder_output = model.encoder(
            input_ids=torch.as_tensor([input_ids], device=device)
        )[0]
        start_ids = torch.as_tensor(
            [[model.generation_config.decoder_start_token_id]],
            dtype=torch.int64,
            device=device,
        )

    past_key_values = out = None
    sent_interrupt = False
    for i in range(max_new_tokens):
        if i == 0:  # prefill
            if model.config.is_encoder_decoder:
                out = model.decoder(
                    input_ids=start_ids,
                    encoder_hidden_states=encoder_output,
                    use_cache=True,
                )
                logits = model.lm_head(out[0])
            else:
                out = model(torch.as_tensor([input_ids], device=device), use_cache=True)
                logits = out.logits
            past_key_values = out.past_key_values
        else:  # decoding
            if model.config.is_encoder_decoder:
                out = model.decoder(
                    input_ids=torch.as_tensor(
                        [[token] if not sent_interrupt else output_ids], device=device
                    ),
                    encoder_hidden_states=encoder_output,
                    use_cache=True,
                    past_key_values=past_key_values if not sent_interrupt else None,
                )
                sent_interrupt = False

                logits = model.lm_head(out[0])
            else:
                out = model(
                    input_ids=torch.as_tensor(
                        [[token] if not sent_interrupt else output_ids], device=device
                    ),
                    use_cache=True,
                    past_key_values=past_key_values if not sent_interrupt else None,
                )
                sent_interrupt = False
                logits = out.logits
            past_key_values = out.past_key_values

        if logits_processor:
            if repetition_penalty > 1.0:
                tmp_output_ids = torch.as_tensor([output_ids], device=logits.device)
            else:
                tmp_output_ids = None
            last_token_logits = logits_processor(tmp_output_ids, logits[:, -1, :])[0]
        else:
            last_token_logits = logits[0, -1, :]

        if device == "mps":
            # Switch to CPU by avoiding some bugs in mps backend.
            last_token_logits = last_token_logits.float().to("cpu")

        if temperature < 1e-5 or top_p < 1e-8:  # greedy
            _, indices = torch.topk(last_token_logits, 2)
            tokens = [int(index) for index in indices.tolist()]
        else:
            probs = torch.softmax(last_token_logits, dim=-1)
            indices = torch.multinomial(probs, num_samples=2)
            tokens = [int(token) for token in indices.tolist()]
        token = tokens[0]
        output_ids.append(token)

        if token in stop_token_ids:
            stopped = True
        else:
            stopped = False

        # Yield the output tokens
        if i % stream_interval == 0 or i == max_new_tokens - 1 or stopped:
            if echo:
                tmp_output_ids = output_ids
                rfind_start = len_prompt
            else:
                tmp_output_ids = output_ids[input_echo_len:]
                rfind_start = 0

            output = tokenizer.decode(
                tmp_output_ids,
                skip_special_tokens=True,
                spaces_between_special_tokens=False,
                clean_up_tokenization_spaces=True,
            )
            # TODO: For the issue of incomplete sentences interrupting output, apply a patch and others can also modify it to a more elegant way
            if judge_sent_end and stopped and not is_sentence_complete(output):
                if len(tokens) > 1:
                    token = tokens[1]
                    output_ids[-1] = token
                else:
                    output_ids.pop()
                stopped = False
                sent_interrupt = True

            partially_stopped = False
            if stop_str:
                if isinstance(stop_str, str):
                    pos = output.rfind(stop_str, rfind_start)
                    if pos != -1:
                        output = output[:pos]
                        stopped = True
                    else:
                        partially_stopped = is_partial_stop(output, stop_str)
                elif isinstance(stop_str, Iterable):
                    for each_stop in stop_str:
                        pos = output.rfind(each_stop, rfind_start)
                        if pos != -1:
                            output = output[:pos]
                            stopped = True
                            break
                        else:
                            partially_stopped = is_partial_stop(output, each_stop)
                            if partially_stopped:
                                break
                else:
                    raise ValueError("Invalid stop field type.")

            # Prevent yielding partial stop sequence
            if not partially_stopped:
                yield {
                    "text": output,
                    "usage": {
                        "prompt_tokens": input_echo_len,
                        "completion_tokens": i,
                        "total_tokens": input_echo_len + i,
                    },
                    "finish_reason": None,
                }

        if stopped:
            break

    # Finish stream event, which contains finish reason
    if i == max_new_tokens - 1:
        finish_reason = "length"
    elif stopped:
        finish_reason = "stop"
    else:
        finish_reason = None

    yield {
        "text": output,
        "usage": {
            "prompt_tokens": input_echo_len,
            "completion_tokens": i,
            "total_tokens": input_echo_len + i,
        },
        "finish_reason": finish_reason,
    }

    # Clean
    del past_key_values, out
    gc.collect()
    torch.cuda.empty_cache()
    if device == "xpu":
        torch.xpu.empty_cache()


class ChatIO(abc.ABC):
    @abc.abstractmethod
    def prompt_for_input(self, role: str) -> str:
        """Prompt for input from a role."""

    @abc.abstractmethod
    def prompt_for_output(self, role: str):
        """Prompt for output from a role."""

    @abc.abstractmethod
    def stream_output(self, output_stream):
        """Stream output."""

    @abc.abstractmethod
    def print_output(self, text: str):
        """Print output."""

class SimpleChatIO(ChatIO):
    def __init__(self, multiline: bool = False):
        self._multiline = multiline

    def prompt_for_input(self, role) -> str:
        if not self._multiline:
            return input(f"{role}: ")

        prompt_data = []
        line = input(f"{role} [ctrl-d/z on empty line to end]: ")
        while True:
            prompt_data.append(line.strip())
            try:
                line = input()
            except EOFError as e:
                break
        return "\n".join(prompt_data)

    def prompt_for_output(self, role: str):
        print(f"{role}: ", end="", flush=True)

    def stream_output(self, output_stream):
        pre = 0
        for outputs in output_stream:
            output_text = outputs["text"]
            output_text = output_text.strip().split(" ")
            now = len(output_text) - 1
            if now > pre:
                # print(" ".join(output_text[pre:now]), end=" ", flush=True)
                pre = now
        # print(" ".join(output_text[pre:]), flush=True)
        return " ".join(output_text)

    def print_output(self, text: str):
        print(text)

model_path = "lmsys/vicuna-13b-v1.5"
device = 'cuda'
num_gpus = 4
max_gpu_memory = "64GB"
chatio = SimpleChatIO(False)
load_8bit = False
max_new_tokens = 2048
cpu_offloading = False
conv_template = None
conv_system_msg = None
repetition_penalty = 1

# Model
model, tokenizer = load_model(
    model_path,
    device=device,
    num_gpus=num_gpus,
    max_gpu_memory=max_gpu_memory,
    load_8bit=load_8bit,
    cpu_offloading=cpu_offloading,
    gptq_config=None,
    awq_config=None,
    revision="main",
    debug=True,
)
generate_stream_func = get_generate_stream_function(model, model_path)

# Set context length
context_len = get_context_length(model.config)

# Chat
def new_chat():
    if conv_template:
        conv = get_conv_template(conv_template)
    else:
        conv = get_conversation_template(model_path)
    if conv_system_msg is not None:
        conv.set_system_message(conv_system_msg)
    return conv

def reload_conv(conv):
    """
    Reprints the conversation from the start.
    """
    for message in conv.messages[conv.offset :]:
        # chatio.prompt_for_output(message[0])
        chatio.print_output(message[1])


def Promting(prompt, temperature):
    conv = new_chat()
    inp = prompt

    conv.append_message(conv.roles[0], inp)
    conv.append_message(conv.roles[1], None)
    input_prompt = conv.get_prompt()

    gen_params = {
        "model": model_path,
        "prompt": input_prompt,
        "temperature": temperature,
        "repetition_penalty": repetition_penalty,
        "max_new_tokens": max_new_tokens,
        "stop": conv.stop_str,
        "stop_token_ids": conv.stop_token_ids,
        "echo": False,
    }

    output_stream = generate_stream_func(
            model,
            tokenizer,
            gen_params,
            device,
            context_len=context_len,
            judge_sent_end=True,
    )
    
    outputs = chatio.stream_output(output_stream=output_stream)
    answer = outputs.strip()
    # input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")
    # outputs = model.generate(input_ids, max_length=2048)
    # answer = tokenizer.batch_decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].replace(prompt, '')
    # answer = answer.replace('</s>', '')

    return answer


def Promting_Dis(prompt):
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")
    tokens = model.generate(input_ids, max_new_tokens=128, return_dict_in_generate=True, output_scores=True)
    transition_scores = model.compute_transition_scores(tokens.sequences, tokens.scores, normalize_logits=True)
    input_ids = torch.cat([input_ids, torch.zeros(1, tokens['sequences'].shape[1]-input_ids.shape[1]).to("cuda")], dim=1).to("cuda")
    answer =tokens['sequences'] - input_ids
    answer = answer[0][answer[0]!=0]
    answer = answer.to("cuda")
    return_output = {"tokens":[], "top_logprobs":[]}
    for i in range(len(answer)):
        token_decode = tokenizer.decode(answer[i:i+1])
        log_probs = transition_scores[0][i].item()
        return_output['tokens'].append(token_decode)
        return_output['top_logprobs'].append(log_probs)
    # pdb.set_trace()

    # print(return_output)

    return return_output


def replace_definition(new_definition, generator_prompt_temp):
    generator_prompt_temp.strip('\n')
    generator_prompt_split = generator_prompt_temp.splitlines(True)
    generator_prompt_split[0] = new_definition.strip('\n') + '\n'
    str = ''
    generator_prompt_new = str.join(generator_prompt_split)

    return generator_prompt_new


def replace_example(new_example, generator_prompt_temp, example_index):
    try:
        generator_prompt_temp.strip('\n')
        generator_prompt_split = generator_prompt_temp.split('\n\n')
        definition = generator_prompt_split[0]
        input = re.findall(r"(Input\:\s.*)\nOutput\:\s", new_example, flags=re.DOTALL)
        output = re.findall(r"(Output\:\s.*)", new_example)
        new_example = input[0] + '\n' + output[0]
        generator_prompt_split[example_index] = new_example
        generator_prompt_new = definition + '\n'
        for i in generator_prompt_split[1:]:
            generator_prompt_new += '\n' + i + '\n'
        generator_prompt_new = generator_prompt_new.strip('\n') + '\n'
        return generator_prompt_new
    except:
        return generator_prompt_temp
    
    

def replace_discriminator(new_example, discriminator_prompt_ori):
    discriminator_prompt_ori.strip('\n') + '\n\n'
    examples = discriminator_prompt_ori.split('\n\n', -1)
    discriminator_prompt_new = examples[0] + '\n'
    for example in examples[1:]:
        if example != '':
            example = example.splitlines(True)
            str = ''
            example[-4] = re.split(r'(\.\s|\?\s|\.|\?)', example[-4])
            values = example[-4][::2][:-1]
            delimiters = example[-4][1::2]
            for i in range(len(values)-1):
                str += values[i] + delimiters[i]
            example[-4] = str + new_example.strip('\n')

            str = ''
            str = str.join(example[0:-3])
            discriminator_prompt_new += '\n' + str + '\n'
    
    return discriminator_prompt_new


def replace_definition_dis(definition, discriminator_prompt_ori):
    discriminator_prompt_ori.strip('\n') + '\n\n'
    examples = discriminator_prompt_ori.split('\n\n', -1)
    discriminator_prompt_new = examples[0] + '\n'

    for example in examples[1:]:
        if example != '':
            example = example.splitlines(True)

            str = ''
            example[-4] = re.split(r'(\.\s|\?\s|\.|\?)', example[-4])
            values = example[-4][::2][:-1]
            delimiters = example[-4][1::2]
            str = ' ' + values[-1] + delimiters[-1]
            example[-4] = definition[0] +  str.strip('\n') + '\n'

            str = ''
            str = str.join(example)
            discriminator_prompt_new += '\n' + str + '\n'
    
    return discriminator_prompt_new


def generator_prompt(definition, positive, negative):
    prompt = definition[0] + '\n'
    for instance in positive:
        prompt += "\nInput: " + instance['input'] + '\nOutput: ' + instance["output"] + '\n'

    return prompt


def discriminator_prompt(definition, positive, negative):
    prompt = "Judge the answer is correct ground truth or generated fake answer.\n\n"
    for instance in positive:
        prompt += "Input: " + instance['input'] + '\nOutput: ' + instance["output"] + '\n' + definition[0]  + ' Is above output correct ground truth?' + "\n(A) Yes, it is correct ground truth.\n(B) No, it is generated fake output.\nThe answer is: (A) Yes, it is correct ground truth." + '\n\n'

    return prompt


def generator(generator_prompt, instance):
    prompt = generator_prompt + '\nInput: ' + instance + '\nOutput: '

    prediction = Promting(prompt, 0)

    return prediction


def discriminator(discriminator_prompt, definition, instance, prediction):
    qu = instance.strip('\n')

    examples = discriminator_prompt.split('\n\n', -1)
    example = examples[1]
    example = example.splitlines(True)
    example[-1] = example[-1].split(':')[0] + ': '
    str1 = ''
    str1 = str1.join(example[-4:])

    prefix = discriminator_prompt + '\nInput: ' + str(qu) + '\nOutput: ' + str(prediction) + '\n' + str1

    output = Promting_Dis(prefix)

    # pdb.set_trace()

    try:
        index_A = output['tokens'].index('A')
        log_probability = output['top_logprobs'][index_A]
    except:
        try:
            index_B = output['tokens'].index('B')
            log_probability = output['top_logprobs'][index_B]
            log_probability = math.log(1 - math.exp(log_probability))
        except:
            log_probability = -10
    
    return log_probability


def Loss(generator_prompt, discriminator_prompt, definition, true_instances, train_instances, negative_data):
    score = 0
    for instance in true_instances:
        # For GSM8K
        # log_probability = discriminator(discriminator_prompt, definition, instance["input"], instance["output"])
        
        # For other tasks
        log_probability = discriminator(discriminator_prompt, definition, instance["input"], instance["output"][-1])
        score += log_probability
        print(log_probability)

    print("finish true")

    # pdb.set_trace()

    for instance in train_instances:
        prediction = generator(generator_prompt, instance["input"])
        log_probability = discriminator(discriminator_prompt, definition, instance["input"], prediction)
        print(log_probability)
        score += math.log(1 - math.exp(log_probability))

    return score


def Update_generator(generator_prompt_ori, discriminator_prompt, definition, true_instances, train_instances, loss_function, negative_data):

    for i in range(5):
        print("*************")
        print(i)
        prefix = 'Polish the task instruction to be clearer. Keep the task instruction as declarative.' + '\n\nTask instruction: ' + definition[0] + '\n\nImproved task instruction: '
        new_definition = Promting(prefix, 0.4).strip('\n').lstrip('\n').strip(' ').replace('\n', ' ').replace('\r', ' ').replace('<s>', '')
        print(prefix)
        print('-------------')
        print(new_definition)
        generator_prompt_def = replace_definition(new_definition, generator_prompt_ori)
        loss_current = Loss(generator_prompt_def, discriminator_prompt, definition, true_instances, train_instances, negative_data)
        print(generator_prompt_def)
        print(loss_current)
        if loss_current < loss_function:
            generator_prompt_ori = generator_prompt_def
            discriminator_prompt = discriminator_prompt
            definition = [new_definition]
            loss_function = loss_current
            break


    examples = re.split(r"\n\n", generator_prompt_ori)
    for j in range(1,len(examples)):
        # example = re.findall(r"(Input\:\s.*\nOutput\:\s.*)", examples[j], flags=re.DOTALL)
        example = examples[j]
        print("For example")
        print(j)
        if example == '':
            break
        for i in range(5):
            print("*************")
            print(i)
            prefix = definition[0] + ' Polish the example to make it more representative. Keep the main content. Keep the format as Input: and Output:.' + '\n\nExample: ' + example + '\nImproved example: '
            new_example = Promting(prefix, 0.4).replace('<s>', '')
            print(prefix)
            print(new_example)
            print("-------------")
            pattern = r"Input: (.*?)\nOutput: (.*?)\n"
            match = re.findall(pattern, new_example+'\n', re.DOTALL)
            try:
                new_example = "Input: " + match[0][0] + "\nOutput: " + match[0][1]
                print(new_example)
                print("=============")
            except:
                new_example = example
            generator_prompt_ex = replace_example(new_example, generator_prompt_ori, j)
            loss_current = Loss(generator_prompt_ex, discriminator_prompt, definition, true_instances, train_instances, negative_data)
            print(generator_prompt_ex)
            print("-------------")
            print(loss_current)
            if loss_current < loss_function:
                generator_prompt_ori = generator_prompt_ex
                loss_function = loss_current
                break

    
    return generator_prompt_ori, definition



def Update_discriminator(generator_prompt, discriminator_prompt_ori, definition, true_instances, train_instances, loss_function, negative_data):
    discriminator_prompt_ori = replace_definition_dis(definition, discriminator_prompt_ori)

    for i in range(5):
        print("*************")
        print(i)
        definition_dis = discriminator_prompt_ori.strip('\n').splitlines(True)[0]
        prefix = 'Polish the task instruction to be clearer. Keep the task instruction as declarative.' + '\n\nTask instruction: ' + definition_dis + '\nImproved task instruction: '
        print(prefix)
        print("-------------")
        new_definition = Promting(prefix, 0.4).strip('\n').lstrip('\n').strip(' ').replace('\n', ' ').replace('\r', ' ').replace('<s>', '')
        print(new_definition)
        discriminator_prompt_def = replace_definition(new_definition, discriminator_prompt_ori)
        loss_current = Loss(generator_prompt, discriminator_prompt_def, definition, true_instances, train_instances, negative_data)
        print(discriminator_prompt_def)
        print(loss_current)   
        if loss_current > loss_function:
            discriminator_prompt_ori = discriminator_prompt_def
            loss_function = loss_current
            break

    example = discriminator_prompt_ori.split('\n\n', -1)[1]
    example = example.splitlines(True)
    example[-4] = example[-4].split('. ')[-1]
    str = ''
    str = str.join(example[-4:])
    for i in range(5):
        print("*************")
        print(i)
        prefix = 'Polish the multiple-choice question and the answer to make it more representative. Keep the main content. Keep the format as multiple-choice question and the answer.\n\nMultiple-choice question and the answer: ' + str + '\n\nImproved multiple-choice question and the answer: '
        print(prefix)
        print("-------------")
        new_example = Promting(prefix, 0.4).replace('\n\n', '\n').replace('<s>', '')
        new_example = re.sub('\n+', '\n', new_example)
        new_example = re.split(r'(\.\s|\?\s|\.|\?)', new_example)
        values = new_example[::2][:-1]
        delimiters = new_example[1::2]
        new_example = ''
        for index in range(len(values)):
            new_example += values[index].strip(' ') + delimiters[index] + '\n'
        new_example = new_example.replace('\n\n', '\n')
        print(new_example)
        discriminator_prompt_new = replace_discriminator(new_example, discriminator_prompt_ori)
        loss_current = Loss(generator_prompt, discriminator_prompt_new, definition, true_instances, train_instances, negative_data)
        print(discriminator_prompt_new)
        print(loss_current)
        if loss_current > loss_function:
            discriminator_prompt_ori = discriminator_prompt_new 
            loss_function = loss_current
            break
    
    return discriminator_prompt_ori, loss_function


def OptimizePrompt(generator_prompt, discriminator_prompt, definition, true_instances_full, train_instances_full, negative_data):
    num_shots = 3
    num_sample = 5

    generator_prompt_set = []

    for i in range(num_shots):    
        true_instances = random.sample(true_instances_full, num_sample)
        train_instances = random.sample(train_instances_full, num_sample)
        print("Optimize Iteration")
        print(i)
        print(generator_prompt)
        loss_function = Loss(generator_prompt, discriminator_prompt, definition, true_instances, train_instances, negative_data)
        print("Before Optimize")
        print(loss_function)

        discriminator_prompt, loss_function = Update_discriminator(generator_prompt, discriminator_prompt, definition, true_instances, train_instances, loss_function, negative_data)
        print("After Discriminator")
        print(loss_function)
        generator_prompt, definition = Update_generator(generator_prompt, discriminator_prompt, definition, true_instances, train_instances, loss_function, negative_data)
        
        loss_function = Loss(generator_prompt, discriminator_prompt, definition, true_instances, train_instances, negative_data)
        print("After Optimize")
        print(loss_function)

        print(generator_prompt)
        generator_prompt_set.append(generator_prompt)

    return generator_prompt_set


def Attempt(generator_prompt, instance):
    prediction = generator(generator_prompt, instance)
    print(instance)
    print(prediction)
    return prediction


def main(argv):
    localtime = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())
    print(localtime)
    tasks_dir = "/home/bizon/Desktop/Adv-ICL/tasks/"
    num_true_instances = 100
    num_train_instances = 100


    print("in create predictions")
    # For translation tasks
    # for track in ["xlingual"]:

    # For other tasks
    for track in ["default"]:
        test_tasks = [l.strip() for l in open(f"/home/bizon/Desktop/Adv-ICL/splits/{track}/test_tasks.txt")]
        for task in test_tasks[int(argv[1]):int(argv[2])]:
            print(task)
            file = os.path.join(tasks_dir, 'testset_'+ task + ".json")
            with open(file) as fin:
                task_data = json.load(fin)

            GENERATOR_PROMPT = generator_prompt(task_data["Definition"],task_data["Positive Examples"],task_data["Negative Examples"])
            DISCRIMINATOR_PROMPT = discriminator_prompt(task_data["Definition"],task_data["Positive Examples"],[])
            true_instances = random.sample(task_data["Instances"], num_true_instances)
            train_instances = random.sample(task_data["Instances"], num_train_instances)
            print(GENERATOR_PROMPT)
            GENERATOR_PROMPT_set = OptimizePrompt(GENERATOR_PROMPT, DISCRIMINATOR_PROMPT, task_data["Definition"], true_instances, train_instances, task_data["Negative Examples"])
            
            test_instances = task_data["Instances"][:10]

            
            for i in range(len(GENERATOR_PROMPT_set)):
                GENERATOR_PROMPT = GENERATOR_PROMPT_set[i]
                print(GENERATOR_PROMPT)
                name_file = "/home/bizon/Desktop/Adv-ICL/eval/output/" + "[gan-vicuna]_" + str(task) + "_" + localtime + ".jsonl"
                print(name_file)
                with open(name_file, "w") as fout:
                    for instance in test_instances:
                        print("***********************")
                        print(test_instances.index(instance))
                        prediction = Attempt(GENERATOR_PROMPT,instance["input"])
                        fout.write(json.dumps({
                        "id": instance["id"], 
                        "prediction": prediction},
                        ) + "\n")
                        # break

if __name__ == "__main__":
    main(sys.argv[1:])
