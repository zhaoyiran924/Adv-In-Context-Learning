import json
import os
import openai
import re
import sys
import math
from copy import deepcopy
import random
import numpy as np


from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
)

@retry(wait=wait_exponential(min=8, max=100000), stop=stop_after_attempt(600))
def Promting(messages):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo", messages=messages)
    
    return response['choices'][0]['message']['content']

@retry(wait=wait_exponential(min=8, max=100000), stop=stop_after_attempt(600))
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

    prediction = Promting(generator_prompt_temp).replace('\n\n', '\n')
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
        log_probability = discriminator(discriminator_prompt, instance["input"], instance["output"])
        score += log_probability
        print(log_probability)

    print("finish true")


    for instance in train_instances:
        prediction = generator(generator_prompt, instance["input"])
        log_probability = discriminator(discriminator_prompt, instance["input"], prediction)
        print(log_probability)
        score += math.log(1 - math.exp(log_probability))

    return score


def Update_generator(generator_prompt_ori, discriminator_prompt, definition, true_instances, train_instances, loss_function, negative_data):

    for i in range(10):
        print("*************")
        print(i)

        definition_dis = generator_prompt_ori[0]['content']
        task_instruction = 'Polish the task instruction to be clearer. Keep the task instruction as declarative.'
        instance_task = 'Task instruction: ' + definition_dis + '\nImproved task instruction: '
        messages = Construct_Message(task_instruction, instance_task)
        new_definition = Promting(messages)
        print(new_definition)
        print('-------------')
        generator_prompt_def = replace_definition(new_definition, generator_prompt_ori[:])
        discriminator_prompt_def = replace_definition_dis([new_definition], discriminator_prompt[:])
        # discriminator_prompt_def = discriminator_prompt[:]
        print(generator_prompt_def)
        for g in generator_prompt_def:
            print(g)
        loss_current = Loss(generator_prompt_def[:], discriminator_prompt_def[:], true_instances, train_instances)
        print(loss_current)
        if loss_current < loss_function:
            generator_prompt_ori = generator_prompt_def[:]
            discriminator_prompt = discriminator_prompt_def[:]
            definition = [new_definition]
            loss_function = loss_current
            break


    examples = generator_prompt_ori[1:]

    for j in range(int(len(examples)/2)):
        print("For example")
        print(j+1)

        for i in range(10):
            print("*************")
            print(i)
            input_instance =  'Input: ' + examples[2*j]['content'] + '\nOutput: ' + examples[2*j+1]['content']
            task_instruction = definition[0] + ' Polish the example to make it more representative. Keep the format as Input: and Output:. Keep Input as a multi-choice question. Keep the Output start with: Let\'s think step by step\n. Keep the last sentence of Output as: So the answer is .'
            instance_task = '\n\nExample: ' + input_instance + '\nImproved example: '
            messages = Construct_Message(task_instruction, instance_task)
            new_example = Promting(messages).replace('\n\n', '\n')
            print(new_example)
            print("-------------")
            generator_prompt_ex = replace_example(new_example, generator_prompt_ori[:], j)
            for g in generator_prompt_ex:
                print(g)
            loss_current = Loss(generator_prompt_ex[:], discriminator_prompt[:], true_instances, train_instances)
            print(loss_current)
            if loss_current < loss_function:
                generator_prompt_ori = generator_prompt_ex[:]
                loss_function = loss_current
                break
    
    return generator_prompt_ori, definition



def Update_discriminator(generator_prompt, discriminator_prompt_ori, definition, true_instances, train_instances, loss_function, negative_data):
    discriminator_prompt_ori = replace_definition_dis(definition, discriminator_prompt_ori)

    for i in range(10):
        print("*************")
        print(i)
        definition_dis = discriminator_prompt_ori[0]['content']
        task_instruction = 'Polish the task instruction to be clearer. Keep the task instruction as declarative.'
        instance_task = 'Task instruction: ' + definition_dis + '\nImproved task instruction: '
        messages = Construct_Message(task_instruction, instance_task)
        new_definition = Promting(messages)
        discriminator_prompt_def = replace_definition(new_definition, discriminator_prompt_ori[:])
        print(discriminator_prompt_def)
        loss_current = Loss(generator_prompt[:], discriminator_prompt_def[:], true_instances, train_instances)
        print(loss_current)   
        if loss_current > loss_function:
            discriminator_prompt_ori = discriminator_prompt_def[:]
            loss_function = loss_current
            break
    
    example = discriminator_prompt_ori[1]['content']
    example = example.splitlines(True)
    example[-4] = example[-4].split('. ')[-1]
    str = ''
    str = str.join(example[-4:])
    str += discriminator_prompt_ori[2]['content']
    for i in range(10):
        print("*************")
        print(i)
        task_instruction = 'Polish the multiple-choice question and the answer to make it more representative. Keep the format as multiple-choice question and the answer.'
        instance_task = 'Multiple-choice question and the answer: ' + str + '\nImproved multiple-choice question and the answer: '
        messages = Construct_Message(task_instruction, instance_task)
        new_example = Promting(messages).replace('\n\n', '\n')
        print(new_example)
        discriminator_prompt_new = replace_discriminator(new_example, discriminator_prompt_ori[:])
        print(discriminator_prompt_new)
        loss_current = Loss(generator_prompt[:], discriminator_prompt_new[:], true_instances, train_instances)
        print(loss_current)
        if loss_current > loss_function:
            discriminator_prompt_ori = discriminator_prompt_new[:] 
            loss_function = loss_current
            break
    
    return discriminator_prompt_ori, loss_function


def OptimizePrompt(generator_prompt, discriminator_prompt, definition, true_instances_full, train_instances_full, negative_data):
    num_shots = 2
    num_sample = 3

    generator_prompt_set = []
    generated_prompt_tt = generator_prompt[:]

    for i in range(num_shots): 
        true_instances = random.sample(true_instances_full, 3)
        train_instances = random.sample(train_instances_full, num_sample)
        print("Optimize Iteration")
        print(i)
        print(generator_prompt)
        loss_function = Loss(generator_prompt[:], discriminator_prompt[:], true_instances, train_instances)
        print("Before Optimize")
        print(loss_function)

        generator_prompt_temp = generator_prompt[:]
        discriminator_prompt_temp = discriminator_prompt[:]
        definition_temp = definition
        loss_function_temp = loss_function


        discriminator_prompt, loss_function = Update_discriminator(generator_prompt[:], discriminator_prompt[:], definition, true_instances, train_instances, loss_function, negative_data)
        print("After Discriminator")
        print(loss_function)
        generator_prompt, definition = Update_generator(generator_prompt[:], discriminator_prompt[:], definition, true_instances, train_instances, loss_function, negative_data)
        
        loss_function = Loss(generator_prompt[:], discriminator_prompt[:], true_instances, train_instances)
        print("After Optimize")
        print(loss_function)
        
        # if loss_function > loss_function_temp:
        #     generator_prompt = generator_prompt_temp[:]
        #     discriminator_prompt = discriminator_prompt_temp[:]
        #     definition = definition_temp
        
        print(generator_prompt)

        generator_prompt_set.append(generator_prompt_temp[:])

    return generator_prompt_set


def Attempt(generator_prompt, instance):
    prediction = generator(generator_prompt, instance)

    try:
        prediction = re.findall(r"So\sthe\sanswer\sis\s(.*)\.", prediction)[0]
    except:
        prediction = re.findall(r"\s(\S*)\.", prediction)[-1]
    print(prediction)

    return prediction


def main(argv):
    openai.api_key = (argv[0])
    tasks_dir = "/home/dltp_yiran/Adversarial-In-context-Learning/tasks/"
    num_train_instances = 100

    for track in ["default"]:
        test_tasks = [l.strip() for l in open(f"/home/dltp_yiran/Adversarial-In-context-Learning/splits/{track}/test_tasks.txt")]


        for task in test_tasks[int(argv[1]):int(argv[2])]:
            print(task)
            file = os.path.join(tasks_dir, task + ".json")
            with open(file) as fin:
                task_data = json.load(fin)

            task_data["Definition"] = [task_data["Definition"][0].strip('\n')]

            GENERATOR_PROMPT = generator_prompt(task_data["Definition"],task_data["Positive Examples"])
            DISCRIMINATOR_PROMPT = discriminator_prompt(task_data["Definition"],task_data["Positive Examples"], task_data["Negative Examples"])

            train_instances = random.sample(task_data["Instances"], num_train_instances)
            GENERATOR_PROMPT_set = OptimizePrompt(GENERATOR_PROMPT, DISCRIMINATOR_PROMPT, task_data["Definition"], task_data["Positive Examples"], train_instances, task_data["Negative Examples"])
        
            test_instances = task_data["Instances"]

            for ite in range(2):
                file_iteration_name = "test_predictions_bbh_" + task + '_' + str(ite) + ".jsonl"
                with open(file_iteration_name, "a") as fout:
                    print("in create predictions")
                    GENERATOR_PROMPT = GENERATOR_PROMPT_set[ite]
                    print(GENERATOR_PROMPT)

                    for instance in test_instances:
                        prediction = Attempt(GENERATOR_PROMPT,instance["input"])
                        fout.write(json.dumps({
                        "id": instance["id"], 
                        "prediction": prediction},
                        ) + "\n")
                    fout.close()

if __name__ == "__main__":
    main(sys.argv[1:])
