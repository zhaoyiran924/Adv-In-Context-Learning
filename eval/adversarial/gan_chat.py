import json
import os

import os
import openai
import json
import re
import sys
import math
from copy import deepcopy
import random
import numpy as np
import pdb
import time

from tqdm import tqdm

from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
)

@retry(wait=wait_exponential(min=8, max=100), stop=stop_after_attempt(6))
def Promting(messages, temperature, number):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo", messages=messages, temperature=temperature, n=number)
    
    return response['choices']

@retry(wait=wait_exponential(min=8, max=100), stop=stop_after_attempt(6))
def Promting_Dis(messages):
    prefix = messages[0]['content'] + '\n\n'
    for i in messages[1:]:
        if i['role'] == 'user':
            prefix = prefix + i['content']
        else:
            prefix = prefix + i['content'] + '\n\n'

    response = openai.Completion.create(
    engine="text-davinci-002", prompt=prefix, temperature=0, max_tokens=374, top_p=1, logprobs=1)

    return response['choices'][0]


def Construct_Message(task_instruction, instance):
    messages = []
    messages.append({"role":"system", "content":task_instruction})
    messages.append({"role":"user", "content": instance})

    return messages


def replace_definition(new_definition, generator_prompt_temp):
    generator_prompt_temp[0]['content'] = new_definition

    return generator_prompt_temp


def replace_example(new_example, generator_prompt_temp, example_index):

    definition = [generator_prompt_temp[0]]
    input = re.findall(r"Input\:\s(.*)\nOutput\:\s", new_example, flags=re.DOTALL)
    output = re.findall(r"Output\:\s(.*)", new_example)
    
    generator_prompt_temp[2*example_index + 1] = {"role":"user", "content":input[0]}
    generator_prompt_temp[2*example_index + 2] = {"role":"assistant", "content":output[0]}

    generator_prompt_new = definition
    for i in generator_prompt_temp[1:]:
        generator_prompt_new.append(i)
    
    return generator_prompt_new

    

def replace_discriminator(new_example, discriminator_prompt_ori):
    try:
        new_input = new_example.split('\n')
        temp = new_input[-1].split(':')
        first = new_input[0] + '\n' + new_input[1] + '\n' + new_input[2] + '\n' + temp[0] + ': '
        second = temp[1].strip(' ')

        examples = discriminator_prompt_ori
        discriminator_prompt_new = [examples[0]]
        for example in examples[1:]:
            if example['role'] == "user":
                example = example['content'].splitlines(True)
                str = ''
                example[-4] = re.split(r'(\.\s|\?\s|\.|\?)', example[-4])
                values = example[-4][::2][:-1]
                delimiters = example[-4][1::2]
                for i in range(len(values)-1):
                    str += values[i] + delimiters[i]
                example[-4] = str + first.strip('\n')
                str = ''
                str = str.join(example[0:-3])

                discriminator_prompt_new.append({"role":"user", "content":str})
            elif example['role'] == "assistant":
                discriminator_prompt_new.append({"role":"assistant", "content":second})
        
        return discriminator_prompt_new
    except:
        return discriminator_prompt_ori[:]

def replace_definition_dis(definition, discriminator_prompt_ori):
    examples = discriminator_prompt_ori
    discriminator_prompt_new = [examples[0]]

    for example in examples[1:]:
        if example['role'] == 'user':
            example = example['content'].splitlines(True)
            str = ''
            example[-4] = re.split(r'(\.\s|\?\s|\.|\?)', example[-4])
            values = example[-4][::2][:-1]
            delimiters = example[-4][1::2]
            str = ' ' + values[-1] + delimiters[-1]
            example[-4] = definition[0] +  str.strip('\n') + '\n'

            str = ''
            str = str.join(example)

            discriminator_prompt_new.append({"role":"user", "content":str})
        elif example['role'] == 'assistant':
            discriminator_prompt_new.append({"role":"assistant", "content":example['content']})
    
    return discriminator_prompt_new

def generator_prompt(definition, positive):
    task_instruction = definition[0]
    examples = []
    for instance in positive:
        examples.append([instance['input'], instance["output"]])
    
    messages = []
    messages.append({"role":"system", "content":task_instruction})
    for i in range(len(examples)):
        messages.append({"role":"user", "content":examples[i][0]})
        messages.append({"role":"assistant", "content":examples[i][1]})

    return messages


def discriminator_prompt(definition, positive, negative):
    task_instruction = "Judge the answer is correct ground truth or generated fake answer."
    examples = []
    for instance in positive:
        input_instance = "Input: " + instance['input'] + '\nOutput: ' + instance["output"] + '\n' + definition[0]  + ' Is above output correct ground truth?' + "\n(A) Yes, it is correct ground truth.\n(B) No, it is generated fake output.\nThe answer is: "
        output_instance = "(A) Yes, it is correct ground truth."
        examples.append([input_instance, output_instance])

    messages = []
    messages.append({"role":"system", "content":task_instruction})
    for i in range(len(examples)):
        messages.append({"role":"user", "content":examples[i][0]})
        messages.append({"role":"assistant", "content":examples[i][1]})
    
    return messages


def generator(generator_prompt, instance):
    generator_prompt_temp = generator_prompt[:]
    generator_prompt_temp.append({"role":"user", "content": instance})

    prediction = Promting(generator_prompt_temp, 0, 1)
    # print(prediction)

    return prediction


def discriminator(discriminator_prompt, instance, prediction):
    qu = instance.strip('\n')

    example = discriminator_prompt[1]['content']
    example = example.splitlines(True)
    example[-1] = example[-1].split(':')[0] + ': '
    str1 = ''
    str1 = str1.join(example[-4:])

    prefix = 'Input: ' + str(qu) + '\nOutput: ' + str(prediction) + '\n' + str1

    discriminator_prompt_temp = discriminator_prompt[:]

    discriminator_prompt_temp.append({"role":"user", "content": prefix})

    output = Promting_Dis(discriminator_prompt_temp)['logprobs']

    try:
        index_A = output['tokens'].index('A')
        log_probability = output['top_logprobs'][index_A]['A']
    except:
        try:
            index_B = output['tokens'].index('B')
            log_probability = output['top_logprobs'][index_B]['B']
            log_probability = math.log(1 - math.exp(log_probability))
        except:
            log_probability = -10
    
    return log_probability


def Loss(generator_prompt, discriminator_prompt, true_instances, train_instances):
    score = 0

    for instance in true_instances:
        log_probability = discriminator(discriminator_prompt, instance["input"], instance["output"][-1])
        score += log_probability

    for instance in train_instances:
        prediction = generator(generator_prompt, instance["input"])
        log_probability = discriminator(discriminator_prompt, instance["input"], prediction)
        score += math.log(1 - math.exp(log_probability))


    return score


def Update_generator(generator_prompt_ori, discriminator_prompt, definition, true_instances, train_instances, loss_function, negative_data):

    definition_dis = generator_prompt_ori[0]['content']
    task_instruction = 'Diversify the task instruction to be clearer. Keep the task instruction as declarative.'
    instance_task = '\n\nTask instruction: ' + definition_dis + '\n\nImproved task instruction: '
    messages = Construct_Message(task_instruction, instance_task)
    new_definition_set = Promting(messages, 0.4, 5)
    loss_new_definition_set = []
    generator_prompt_def_set = []

    for new_definition in new_definition_set:
        new_definition = new_definition['message']['content'].strip('\n').replace('\n', ' ')
        print(new_definition)
        print("-------------")
        generator_prompt_def = replace_definition(new_definition, generator_prompt_ori[:])
        generator_prompt_def_set.append(generator_prompt_def[:])
        loss_current = Loss(generator_prompt_def[:], discriminator_prompt[:], true_instances, train_instances)
        loss_new_definition_set.append(loss_current)

    minimum_loss = min(loss_new_definition_set)

    if minimum_loss < loss_function:
        generator_prompt_ori = generator_prompt_def_set[loss_new_definition_set.index(minimum_loss)][:]
        definition = [new_definition_set[loss_new_definition_set.index(minimum_loss)]['message']['content'].strip('\n').replace('\n', ' ').replace('\r', ' ')]
        loss_function = minimum_loss
        print((loss_new_definition_set.index(minimum_loss), minimum_loss, generator_prompt_ori))
    else:
        print("Fail Optimization")



    examples = generator_prompt_ori[1:]

    for j in range(int(len(examples)/2)):
        # example = re.findall(r"(Input\:\s.*\nOutput\:\s.*)", examples[j], flags=re.DOTALL)
        print("For example")
        print(j+1)

        input_instance =  'Input: ' + examples[2*j]['content'] + '\nOutput: ' + examples[2*j+1]['content']
        task_instruction = definition[0] + ' Diversify the example to make it more representative. Keep the format as Input: and Output: .' 
        instance_task = '\n\nExample: ' + input_instance + '\n\nImproved example: '
        messages = Construct_Message(task_instruction, instance_task)
        new_example_set = Promting(messages, 0.4, 5)
        loss_new_example_set = []
        generator_prompt_ex_set = []

        for new_example in new_example_set:
            new_example = new_example['message']['content'].strip('\n').replace('\n\n', '\n')  
            print(new_example)
            print("-------------")
            pattern = r"Input: (.*?)\nOutput: (.*?)\n"
            match = re.findall(pattern, new_example+'\n', re.DOTALL)
            # pdb.set_trace()
            new_example = "Input: " + match[0][0] + "\nOutput: " + match[0][1]
            print(new_example)
            print("=============")
            generator_prompt_ex = replace_example(new_example, generator_prompt_ori[:], j)
            loss_current = Loss(generator_prompt_ex[:], discriminator_prompt[:], true_instances, train_instances)
            loss_new_example_set.append(loss_current)
            generator_prompt_ex_set.append(generator_prompt_ex[:])

        minimum_loss= min(loss_new_example_set)

        if minimum_loss < loss_function:
            generator_prompt_ori = generator_prompt_ex_set[loss_new_example_set.index(minimum_loss)][:]
            loss_function = minimum_loss
            print((loss_new_example_set.index(minimum_loss), minimum_loss, generator_prompt_ori))
        else:
            print("Fail Optimization")
    
    return generator_prompt_ori, definition



def Update_discriminator(generator_prompt, discriminator_prompt_ori, definition, true_instances, train_instances, loss_function, negative_data):
    discriminator_prompt_ori = replace_definition_dis(definition, discriminator_prompt_ori)

    definition_dis = discriminator_prompt_ori[0]['content']
    task_instruction = 'Diversify the task instruction to be clearer. Keep the task instruction as declarative.'
    instance_task = '\n\nTask instruction: ' + definition_dis + '\n\nImproved task instruction: '
    messages = Construct_Message(task_instruction, instance_task)
    new_definition_set = Promting(messages, 0.4, number=5)
    loss_new_definition_set = []
    discriminator_prompt_def_set = []

    for new_definition in new_definition_set:
        new_definition = new_definition['message']['content']
        print(new_definition)
        print("-------------")
        discriminator_prompt_def = replace_definition(new_definition, discriminator_prompt_ori[:])
        loss_current = Loss(generator_prompt[:], discriminator_prompt_def[:], true_instances, train_instances)
        discriminator_prompt_def_set.append(discriminator_prompt_def)
        loss_new_definition_set.append(loss_current)

    maximum_loss = max(loss_new_definition_set)
    if maximum_loss > loss_function:
        discriminator_prompt_ori = discriminator_prompt_def_set[loss_new_definition_set.index(maximum_loss)][:]
        loss_function = maximum_loss
        print((loss_new_definition_set.index(maximum_loss), maximum_loss, discriminator_prompt_ori))
    else:
        print("Fail Optimization")
    
    print("===========================")
    print("After Discriminator Instruction")
    
    example = discriminator_prompt_ori[1]['content']
    example = example.splitlines(True)
    example[-4] = example[-4].split('. ')[-1]
    str = ''
    str = str.join(example[-4:])
    str += discriminator_prompt_ori[2]['content']

    task_instruction = 'Diversify the multiple-choice question and the answer to make it more representative. Keep the main content. Keep the format as multiple-choice question and the answer.'
    instance_task = '\n\nMultiple-choice question and the answer: ' + str + '\n\nImproved multiple-choice question and the answer: '
    messages = Construct_Message(task_instruction, instance_task)
    new_example_set = Promting(messages, 0.4, 5)
    loss_new_example_set = []
    discriminator_prompt_ex_set = []

    for new_example in new_example_set:
        new_example = new_example['message']['content']
        print(new_example)
        print("-------------")
        discriminator_prompt_new = replace_discriminator(new_example, discriminator_prompt_ori[:])
        loss_current = Loss(generator_prompt[:], discriminator_prompt_new[:], true_instances, train_instances)
        discriminator_prompt_ex_set.append(discriminator_prompt_new[:])
        loss_new_example_set.append(loss_current)
    
    maximum_loss = max(loss_new_example_set)
    if maximum_loss > loss_function:
        discriminator_prompt_ori = discriminator_prompt_ex_set[loss_new_example_set.index(maximum_loss)][:]
        loss_function = maximum_loss
        print((loss_new_example_set.index(maximum_loss), maximum_loss, discriminator_prompt_ori))
    else:
        print("Fail Optimization")
    
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
        loss_function = Loss(generator_prompt[:], discriminator_prompt[:], true_instances, train_instances)
        print("Before Optimize")
        print(loss_function)


        discriminator_prompt, loss_function = Update_discriminator(generator_prompt[:], discriminator_prompt[:], definition, true_instances, train_instances, loss_function, negative_data)
        print("After Discriminator")
        print(loss_function)
        generator_prompt, definition = Update_generator(generator_prompt[:], discriminator_prompt[:], definition, true_instances, train_instances, loss_function, negative_data)
        
        loss_function = Loss(generator_prompt[:], discriminator_prompt[:], true_instances, train_instances)
        print("After Optimize")
        print(loss_function)
        
        
        print(generator_prompt)

        generator_prompt_set.append(generator_prompt[:])

    return generator_prompt_set


def Attempt(generator_prompt, instance):
    prediction = generator(generator_prompt, instance)[0]['message']['content']

    # print(prediction)
    return prediction


def main(argv):
    localtime = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())
    print(localtime)
    openai.api_key = (argv[0])
    tasks_dir = "/home/bizon/Desktop/Adv-ICL/tasks/"
    num_true_instances = 90
    num_train_instances = 90

    # For translation tasks
    # for track in ["xlingual"]:

    # For other tasks
    for track in ["default"]:
        test_tasks = [l.strip() for l in open(f"/home/bizon/Desktop/Adv-ICL/splits/{track}/test_tasks.txt")]

        for task in test_tasks[int(argv[1]):int(argv[2])]:
            print(task)
            file = os.path.join(tasks_dir, 'testset_'+ task + ".json")
            # file = os.path.join(tasks_dir, task + ".json")
            with open(file) as fin:
                task_data = json.load(fin)

            task_data["Definition"] = [task_data["Definition"][0].strip('\n')]

            GENERATOR_PROMPT = generator_prompt(task_data["Definition"],task_data["Positive Examples"])
            DISCRIMINATOR_PROMPT = discriminator_prompt(task_data["Definition"],task_data["Positive Examples"], task_data["Negative Examples"])
            print(DISCRIMINATOR_PROMPT)
            true_instances = random.sample(task_data["Instances"], num_true_instances)
            train_instances = random.sample(task_data["Instances"], num_train_instances)
            GENERATOR_PROMPT_set = OptimizePrompt(GENERATOR_PROMPT, DISCRIMINATOR_PROMPT, task_data["Definition"], true_instances, train_instances, task_data["Negative Examples"])

            test_instances = task_data["Instances"]
            GENERATOR_PROMPT = GENERATOR_PROMPT_set[-1]
            print(GENERATOR_PROMPT)

            # for ite in range(3):
            name_file = "/home/bizon/Desktop/Adv-ICL/eval/output/" + "[gan-chat-deversify]_" + str(task) + "_" + localtime + ".jsonl"
            print(name_file)
            with open(name_file, "w") as fout:
                print("in create predictions")

                for instance in tqdm(test_instances):
                    prediction = Attempt(GENERATOR_PROMPT,instance["input"])
                    fout.write(json.dumps({
                    "id": instance["id"], 
                    "prediction": prediction},
                    ) + "\n")
                fout.close()

if __name__ == "__main__":
    main(sys.argv[1:])
