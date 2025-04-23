# Multi Agent Code Development Team with RLHF
This project implements a modular multi-agent system for automated code development. It features specialized language agents for:
- Chain-of-thought reasoning
- Code generation
- Debugging
- Explanation

To improve agent performance, the system uses **Reinforcement Learning with Human Feedback (RLHF)** and **Reinforcement Learning with AI Feedback (RLAIF)**. Feedback is collected via a custom **Streamlit interface** and used to train agent-specific reward models. The training loop uses PPO (Proximal Policy Optimization) for fine-tuning.

---

## Project Structure
```
.
├── CodeCraftModels.ipynb
├── README.md
├── RLHF
│   ├── Reward_Dataset_RLAIF_criteria.ipynb
│   ├── Reward_dataset_with_testcases.ipynb
│   ├── Rewards_RLAIF_criteria.ipynb
│   ├── Rewards_with_testcases.ipynb
│   ├── data
│   │   ├── generator_rewards.jsonl
│   │   ├── preference_feedback.json
│   │   └── scalar_feedback.json
│   ├── json_reader.py
│   ├── policy_trainer.py
│   ├── ppo_trainer.ipynb
│   ├── ppo_trainer.py
│   └── reward_model_trainer.py
├── Student Model Datasets
│   ├── code_student_50_dataset.json
│   ├── code_student_51_dataset.json
│   ├── code_teacher_374_dataset.json
│   ├── debugged_student_50_dataset.json
│   ├── debugged_student_51_dataset.json
│   ├── debugged_teacher_374_dataset.json
│   ├── explanation_student_50_dataset.json
│   ├── explanation_student_51_dataset.json
│   ├── explanation_teacher_374_dataset.json
│   ├── solution_cot_student_50_dataset.json
│   ├── solution_cot_student_51_dataset.json
│   └── solution_cot_teacher_374_dataset.json
├── app.py
├── app_ab.py
├── graph.py

5 directories, 35 files
```
## File Descriptions
```
| File      | Description                                                               |
| --------- | ------------------------------------------------------------------------- |
| app.py    | Scalar feedback UI via Streamlit                                          |
| app_ab.py | Preference based feedback UI via Streamlit                                |
| graph.py  | Implements planner logic for routing prompts and responses between agents |
```

## /RLHF
| File                                | Description                                                         |
| ----------------------------------- | ------------------------------------------------------------------- |
| Reward_Dataset_RLAIF_criteria.ipynb | Implements code for dataset generation using RLAIF                  |
| Reward_dataset_with_testcases.ipynb | Implements code for dataset generation using testcases              |
| json_reader.py                      | Implements code for reading and filtering json data                 |
| policy_triner.py                    | Implements code for retraining the agents using collected feedback  |
| ppo_trainer.py                      | Implements code for PPO training loop                               |
| reward_model_trainer.py             | Implements code for updating reward models using collected feedback |

## /RLHF/data
| File                     | Description                              |
| ------------------------ | ---------------------------------------- |
| generator_rewards.jsonl  | Data generated for training reward model |
| preference_feedback.json | Sample preference feedback               |
| scalar_ffedback.json     | Sample scalar feedback                   |

## Agents
All the agents and models that are trained for the project are in the following drive:
https://drive.google.com/drive/folders/1QA9JWC1a-KPyhpRmJi-rjkaCYp6hTqgz?usp=share_link