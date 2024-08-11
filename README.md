# Prompt Optimization via Adversarial In-Context Learning 

This repository contains codes for the ACL 2024 Oral paper: [Prompt Optimization via Adversarial In-Context Learning](https://arxiv.org/abs/2312.02614)

## Requirements

#### Main Environment

```
openai                             0.27.4
rouge-score                        0.1.2
numpy                              1.22.3
transformers                       4.28.1
```

#### Data Preprocessing

We provide example formats of the input dataset in the folder [`tasks`](tasks).

## Running

1. When testing on some tasks, you can run codes in [`eval/adversarial/[mathod].py`](eval/adversarial/) by

   ```
   python gan_chat_mmlu.py (openai key) (start index of test tasks) (end index of test tasks),
   ```

​		where the index of task is in [`splits`](splits). To align with the format, for flan-t5 model, you also need to input random string to alternate openai key.

2. The results will be recored in the correspoinding `.json` file.

3. To evaluate the results, please run evaluation code in [`eval/automatic/evaluation.py`](eval/automatic/)  and [`eval/leaderboard/create_reference_file.py`](eval/leaderboard/).

   'create_reference_file.py' is used to create labeled test dataset for tasks that you want to test by 
   ```
   python create_reference_file.py (start index of test tasks) (end index of test tasks)
   ```

   'evaluation.py' is used to compare your anwer with labeled test dataset by
   ```
   python evaluation.py --prediction_file=[address of prediction file] --reference_file=[address of reference file got by create_reference_file.py]
   
   ```
