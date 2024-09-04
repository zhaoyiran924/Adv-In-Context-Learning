# [ACL 2024] Prompt Optimization via Adversarial In-Context Learning 

This repository contains codes for the ACL 2024 Oral paper: [Prompt Optimization via Adversarial In-Context Learning](https://arxiv.org/abs/2312.02614).

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

## Reference
- If you have any questions or find any bugs, please feel free to contact Do Xuan Long (xuanlong.do@u.nus.edu) and Zhao Yiran (zhaoyiran@u.nus.edu) and Hannah Brown (hsbrown@comp.nus.edu.sg).
- If you found our work helpful, please cite it:

```
@inproceedings{long-etal-2024-prompt,
    title = "Prompt Optimization via Adversarial In-Context Learning",
    author = "Long, Do  and
      Zhao, Yiran  and
      Brown, Hannah  and
      Xie, Yuxi  and
      Zhao, James  and
      Chen, Nancy  and
      Kawaguchi, Kenji  and
      Shieh, Michael  and
      He, Junxian",
    editor = "Ku, Lun-Wei  and
      Martins, Andre  and
      Srikumar, Vivek",
    booktitle = "Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = aug,
    year = "2024",
    address = "Bangkok, Thailand",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.acl-long.395",
    pages = "7308--7327",
    abstract = "We propose a new method, Adversarial In-Context Learning (adv-ICL), to optimize prompts for in-context learning (ICL). Inspired by adversarial learning, adv-ICL is implemented as a two-player game between a generator and discriminator, with LLMs acting as both. In each round, given an input prefixed by task instructions and several exemplars, the generator produces an output. The discriminator then classifies the generator{'}s input-output pair as model-generated or real data. Based on the discriminator{'}s loss, a prompt modifier LLM proposes possible edits to the generator and discriminator prompts, and the edits that most improve the adversarial loss are selected. We show that applying adv-ICL results in significant improvements over state-of-the-art prompt optimization techniques for both open and closed-source models on 13 generation and classification tasks including summarization, arithmetic reasoning, machine translation, data-to-text generation, and the MMLU and big-bench hard benchmarks. In addition, our method is computationally efficient, easily extensible to other LLMs and tasks, and effective in low-resource settings.",
}
```
