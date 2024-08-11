import json
import os
import openai
import json
import re
import sys
import math
from copy import deepcopy
import random
import numpy as np
import time
import pdb

from tqdm import tqdm

from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
)

# @retry(wait=wait_exponential(min=8, max=1000), stop=stop_after_attempt(60))
def Promting(prompt, temperature, number):
    # For Codex (code-davinci-002) Model
    # response = openai.Completion.create(
    #     engine="code-davinci-002", prompt=prompt, temperature=temperature, max_tokens=374, top_p=1, stop=stop, logprobs=1)
    
    # For Text (text-davinci-002) Model
    response = openai.Completion.create(
    engine="text-davinci-002", prompt=prompt, temperature=temperature, max_tokens=1024, n=number, logprobs=1)

    return response['choices']


def replace_definition(new_definition, generator_prompt_temp):
    generator_prompt_temp.strip('\n')
    generator_prompt_split = generator_prompt_temp.splitlines(True)
    generator_prompt_split[0] = new_definition.strip('\n') + '\n'
    str = ''
    generator_prompt_new = str.join(generator_prompt_split)

    return generator_prompt_new


def replace_example(new_example, generator_prompt_temp, example_index):
    try:
        # pdb.set_trace()
        generator_prompt_temp.strip('\n')
        generator_prompt_split = generator_prompt_temp.split('\n\n')
        definition = generator_prompt_split[0]
        # input = re.findall(r"(Input\:\s.*)\nOutput\:\s", new_example, flags=re.DOTALL)
        # output = re.findall(r"(Output\:\s.*)", new_example)
        # new_example = input[0] + '\n' + output[0]
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
    # pdb.set_trace()
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
    # prompt = "Judge the answer is written by human or generated by large language models.\n\n"
    for instance in positive[:3]:
        # prompt += "Input: " + instance['input'] + '\nOutput: ' + instance["output"] + '\n' + definition[0]  + ' Is above output correct ground truth written by human being?' + "\n(A) Yes, it is correct ground truth written by human being.\n(B) No, it is fake output written by language models.\nThe answer is: (A) Yes, it is correct ground truth." + '\n\n'
        # prompt += "Input: " + instance['input'] + '\nOutput: ' + instance["output"] + '\n' + definition[0]  + ' Is above output written by human?' + "\n(A) Yes, it is written by human.\n(B) No, it is generated by large language models.\nThe answer is: (A) Yes, it is written by human." + '\n\n'
        prompt +=  "Input: " + instance['input'] + '\nOutput: ' + instance["output"] + '\n' + definition[0]  + ' Is above output correct ground truth?' + "\n(A) Yes, it is correct ground truth.\n(B) No, it is generated fake output.\nThe answer is: (A) Yes, it is correct ground truth." + '\n\n'

    for instance in negative[:3]:
        # prompt += "Input: " + instance['input'] + '\nOutput: ' + instance["output"] + '\n' + definition[0]  + ' Is above output correct ground truth written by human being?' + "\n(A) Yes, it is correct ground truth written by human being.\n(B) No, it is fake output written by language models.\nThe answer is: (B) No, it is fake output written by language models." + '\n\n'
        # prompt +=  "Input: " + instance['input'] + '\nOutput: ' + instance["output"] + '\n' + definition[0]  + ' Is above output written by human?' + "\n(A) Yes, it is written by human.\n(B) No, it is generated by large language models.\nThe answer is: (B) No, it is generated by large language models." + '\n\n'
        prompt +=  "Input: " + instance['input'] + '\nOutput: ' + instance["output"] + '\n' + definition[0]  + ' Is above output correct ground truth?' + "\n(A) Yes, it is correct ground truth.\n(B) No, it is generated fake output.\nThe answer is: (B) No, it is generated fake output." + '\n\n'


    return prompt


def generator(generator_prompt, instance):

    # prompt = generator_prompt + '\nInput: ' + instance + '\nOutput: '

    # For reasoning tasks
    prompt = generator_prompt + '\nInput: ' + instance + '\nOutput: Let\'s think step by step.\n'

    prediction = Promting(prompt, 0, number=1)[0]['text'].replace('\n\n', '\n').strip('\n')

    return prediction


def discriminator(discriminator_prompt, definition, instance, prediction):
    qu = instance.strip('\n')

    examples = discriminator_prompt.split('\n\n', -1)
    example = examples[1]
    example = example.splitlines(True)
    example[-1] = example[-1].split(':')[0] + ': '
    str1 = ''
    str1 = str1.join(example[-4:])

    # pdb.set_trace()
    # prefix = discriminator_prompt + '\nInput: ' + str(qu) + '\nOutput: ' + str(prediction)  + '\n' + 'The task is: ' + definition[0]  + ' Is above output correct ground truth?' + "\n(A) Yes, it is correct ground truth.\n(B) No, it is generated fake output.\nThe answer is: "

    prefix = discriminator_prompt + '\nInput: ' + str(qu) + '\nOutput: ' + str(prediction)  + str1

    output = Promting(prefix, 0, number=1)[0]['logprobs']

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



def Loss(generator_prompt, discriminator_prompt, definition, true_instances, train_instances, negative_data):
    score = 0
    for instance in true_instances:
        # For GSM8K and BBH
        log_probability = discriminator(discriminator_prompt, definition, instance["input"], instance["output"])
        
        # For other tasks
        # log_probability = discriminator(discriminator_prompt, definition, instance["input"], instance["output"][-1])
        score += log_probability

    # pdb.set_trace()
    for instance in train_instances:
        prediction = 'Let\'s think step by step.\n' + generator(generator_prompt, instance["input"])
        log_probability = discriminator(discriminator_prompt, definition, instance["input"], prediction)
        score += math.log(1 - math.exp(log_probability))

    return score


def Update_generator(generator_prompt_ori, discriminator_prompt, definition, true_instances, train_instances, loss_function, negative_data):
    
    print("===========================")
    print("Before Generator Instruction")
    
    prefix = 'Polish the task instruction to be clearer. Keep the task instruction as declarative.' + '\n\nTask instruction: ' + definition[0] + '\n\nImproved task instruction: '
    new_definition_set = Promting(prefix, 0.4, number=3)
    loss_new_definition_set = []
    generator_prompt_def_set = []
    for new_definition in new_definition_set:
        new_definition = new_definition['text'].strip('\n').replace('\n', ' ')
        generator_prompt_def = replace_definition(new_definition, generator_prompt_ori)
        generator_prompt_def_set.append(generator_prompt_def)
        loss_current = Loss(generator_prompt_def, discriminator_prompt, definition, true_instances, train_instances, negative_data)
        loss_new_definition_set.append(loss_current)
        # print(generator_prompt_def)
        # print(loss_current)
        # print("************")

    minimum_loss = min(loss_new_definition_set)

    if minimum_loss < loss_function:
        generator_prompt_ori = generator_prompt_def_set[loss_new_definition_set.index(minimum_loss)]
        definition = [new_definition_set[loss_new_definition_set.index(minimum_loss)]['text'].strip('\n').replace('\n', ' ').replace('\r', ' ')]
        loss_function = minimum_loss

        # print((loss_new_definition_set.index(minimum_loss), generator_prompt_ori))
        print(generator_prompt_ori)

    else:
        print("Fail Optimization")

    print("===========================")
    print("After Generator Instruction")

    examples = re.split(r"\n\n", generator_prompt_ori)
    for j in range(1,len(examples)):
        # example = re.findall(r"(Input\:\s.*\nOutput\:\s.*)", examples[j], flags=re.DOTALL)
        example = examples[j]
        print("For example")
        print(j)
        if example == '':
            break

        # For other tasks
        # prefix = definition[0] + ' Diversify the example to make it more representative. Keep the format as Input: and Output:.' + '\n\nExample: ' + example + '\n\nImproved example: '
        
        
        # For BBH tasks
        prefix = 'Task is: ' + definition[0] + ' Polish the example to make it more representative. Keep the main content. Keep the format as Input: Option: and Output:. End the Output by So the answer is .' + '\n\nExample: ' + example + '\n\nImproved example: '
        

        # For reasoning tasks
        # prefix = definition[0] + ' Polish the example to make it more representative. Keep the main content. Keep the format as Input: and Output: . End the Output by The answer is .' + '\n\nExample: ' + example + '\n\nImproved example: '
        
        # For summarization tasks
        # prefix = definition[0] + ' Polish the output to make it more representative. Keep the main content. Keep the format as Improved output:.' + '\n\nExample:\n' + example + '\n\nImproved output: '
        
        new_example_set = Promting(prefix, 0.4, number=3)
        # For summarization task
        # input_example = re.findall(r"(Input\:\s.*)\nOutput\:\s", example, flags=re.DOTALL)
        # new_example = input_example[0].strip('\n') + '\n' + 'Output: ' + new_example

        loss_new_example_set = []
        generator_prompt_ex_set = []

        for new_example in new_example_set:
            new_example = new_example['text'].strip('\n').replace('\n\n', '\n')
            generator_prompt_ex = replace_example(new_example, generator_prompt_ori, j)
            generator_prompt_ex_set.append(generator_prompt_ex)
            loss_current = Loss(generator_prompt_ex, discriminator_prompt, definition, true_instances, train_instances, negative_data)
            loss_new_example_set.append(loss_current)
            # print(generator_prompt_ex)
            # print(loss_current)
            # print("************")


        minimum_loss= min(loss_new_example_set)

        if minimum_loss < loss_function:
            generator_prompt_ori = generator_prompt_ex_set[loss_new_example_set.index(minimum_loss)]
            loss_function = minimum_loss
        
            # print((loss_new_example_set.index(minimum_loss), generator_prompt_ori))
            print(generator_prompt_ori)
        else:
            print("Fail Optimization")

    return generator_prompt_ori, definition



def Update_discriminator(generator_prompt, discriminator_prompt_ori, definition, true_instances, train_instances, loss_function, negative_data):
    discriminator_prompt_ori = replace_definition_dis(definition, discriminator_prompt_ori)

    print("===========================")
    print("Before Discriminator Instruction")

    definition_dis = discriminator_prompt_ori.strip('\n').splitlines(True)[0]
    prefix = 'Polish the task instruction to be clearer. Keep the main content. Keep the task instruction as declarative.' + '\n\nTask instruction: ' + definition_dis + '\nImproved task instruction: '
    new_definition_set = Promting(prefix, 0.4, number=3)
    
    loss_new_definition_set = []
    discriminator_prompt_def_set = []
    for new_definition in new_definition_set:
        new_definition = new_definition['text'].strip('\n').replace('\n', ' ').replace('\r', ' ')
        discriminator_prompt_def = replace_definition(new_definition, discriminator_prompt_ori)
        loss_current = Loss(generator_prompt, discriminator_prompt_def, definition, true_instances, train_instances, negative_data)
        discriminator_prompt_def_set.append(discriminator_prompt_def)
        loss_new_definition_set.append(loss_current)

    maximum_loss = max(loss_new_definition_set)
    if maximum_loss > loss_function:
        discriminator_prompt_ori = discriminator_prompt_def_set[loss_new_definition_set.index(maximum_loss)]
        loss_function = maximum_loss
    
        print(discriminator_prompt_ori)
        print(loss_function)
    else:
        print("Fail Optimization")
    
    print("===========================")
    print("After Discriminator Instruction")

    # example = discriminator_prompt_ori.split('\n\n', -1)[1]
    # example = example.splitlines(True)
    # example[-4] = example[-4].split('. ')[-1]
    # str = ''
    # str = str.join(example[-4:])

    # prefix = 'Polish the multiple-choice question and the answer to make it more representative. Keep the main content. Keep the format as multiple-choice question and the answer.\n\nMultiple-choice question and the answer: ' + str + '\n\nImproved multiple-choice question and the answer: '
    # new_example_set = Promting(prefix, 0.4, number=3)
    # loss_new_example_set = []
    # discriminator_prompt_ex_set = []
    # for new_example in new_example_set:
    #     new_example = new_example['text'].strip('\n').replace('\n', ' ')
    #     new_example = re.split(r'(\.\s|\?\s|\.|\?)', new_example)
    #     values = new_example[::2][:-1]
    #     delimiters = new_example[1::2]
    #     new_example = ''
    #     for index in range(len(values)):
    #         new_example += values[index].strip(' ') + delimiters[index] + '\n'
    #     discriminator_prompt_new = replace_discriminator(new_example, discriminator_prompt_ori)
    #     loss_current = Loss(generator_prompt, discriminator_prompt_new, definition, true_instances, train_instances, negative_data)
    #     discriminator_prompt_ex_set.append(discriminator_prompt_new)
    #     loss_new_example_set.append(loss_current)
    
    # maximum_loss = max(loss_new_definition_set)
    # if maximum_loss > loss_function:
    #     discriminator_prompt_ori = discriminator_prompt_ex_set[loss_new_definition_set.index(maximum_loss)]
    #     loss_function = maximum_loss

    #     print(discriminator_prompt_ori)
    #     print(loss_current)
    
    # else:
    #     print("Fail Optimization")
        
    return discriminator_prompt_ori, loss_function



def OptimizePrompt(generator_prompt, discriminator_prompt, definition, true_instances_full, train_instances_full, negative_data):
    num_shots = 3
    num_sample = 3
    num_sample_train = 3

    generator_prompt_set = []

    for i in range(num_shots):
        true_instances = random.sample(true_instances_full, num_sample)
        train_instances = random.sample(train_instances_full, num_sample_train)
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

    # print(instance)

    # print(prediction)

    # For GSM8K and SVAMP
    # try:
    #     prd = re.findall(r"\d+\,?\.?\d*",prediction)[-1]
    #     prd = float(prd.replace(',', '').rstrip('.')) if prd else prd
    # except:
    #     prd = -1
    # prediction = prd

    # For BBH
    try:
        prediction = re.findall(r"So\sthe\sanswer\sis\s(.*)\.", prediction)[0]
    except:
        prediction = re.findall(r"\s(\S*)\.", prediction)[-1]

    return prediction



def main(argv):
    localtime = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())
    print(localtime)
    openai.api_key = (argv[0])
    tasks_dir = "/home/bizon/Desktop/Adv-ICL/tasks/"
    num_true_instances = 100
    num_train_instances = 100
    # For Codex model
    # num_test_instances = 1000

    # For Text model
    # num_test_instances = 100


    print("in create predictions")
    # For translation tasks
    # for track in ["xlingual"]:

    # For other tasks
    for track in ["default"]:
        test_tasks = [l.strip() for l in open(f"/home/bizon/Desktop/Adv-ICL/splits/{track}/test_tasks.txt")]
        for task in test_tasks[int(argv[1]):int(argv[2])]:
            print(task)
            # file = os.path.join(tasks_dir, 'testset_'+ task + ".json")
            file = os.path.join(tasks_dir, task + ".json")
            with open(file) as fin:
                task_data = json.load(fin)

            GENERATOR_PROMPT = generator_prompt(task_data["Definition"],task_data["Positive Examples"],task_data["Negative Examples"])
            DISCRIMINATOR_PROMPT = discriminator_prompt(task_data["Definition"],task_data["Positive Examples"],task_data["Negative Examples"])
            true_instances = random.sample(task_data["Instances"], num_true_instances)
            train_instances = random.sample(task_data["Instances"], num_train_instances)
            print(GENERATOR_PROMPT)
            # GENERATOR_PROMPT_set = OptimizePrompt(GENERATOR_PROMPT, DISCRIMINATOR_PROMPT, task_data["Definition"], true_instances, train_instances, task_data["Negative Examples"])
            
            # For tasks aopted COT, such as GSM8K and BBH
            GENERATOR_PROMPT_set = OptimizePrompt(GENERATOR_PROMPT, DISCRIMINATOR_PROMPT, task_data["Definition"], task_data["Positive Examples"], train_instances, task_data["Negative Examples"])
            

            file = os.path.join(tasks_dir, 'testset_'+ task + ".json")
            with open(file) as fin:
                task_data_test = json.load(fin)
            
            # test_instances = random.sample(task_data["Instances"], num_test_instances)
            test_instances = task_data_test["Instances"]

            for i in range(len(GENERATOR_PROMPT_set)):
                GENERATOR_PROMPT = GENERATOR_PROMPT_set[i]
                print(GENERATOR_PROMPT)
                name_file = "/home/bizon/Desktop/Adv-ICL/eval/output/" + "[gan-text-deversify]_" + str(task) + "_" + str(i) + "_" + localtime + ".jsonl"
                print(name_file)
                with open(name_file, "w") as fout:
                    for instance in tqdm(test_instances):
                        # print("***********************")
                        # print(test_instances.index(instance))
                        prediction = Attempt(GENERATOR_PROMPT,instance["input"])
                        fout.write(json.dumps({
                        "id": instance["id"], 
                        "prediction": prediction},
                        ) + "\n")

if __name__ == "__main__":
    main(sys.argv[1:])
