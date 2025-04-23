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
├── models
│   ├── generator_reward_model.pth
│   ├── student_cot_model.pt
│   ├── student_debugger_model.pt
│   ├── student_explainer_model.pt
│   └── student_generator_model.pt
└── structure.txt

5 directories, 35 files
```
## File Descriptions
File/Folder | Description
app.py | Launches the Streamlit feedback interface for scalar feedback.
app_ab.py | Launches the Streamlit UI for preference-based feedback between models.
graph.py | Manages the planner agent and the data flow between specialized agents.
CodeCraftModels.ipynb | Notebook that documents and demonstrates how the multi-agent models work together.
README.md | Project overview, setup instructions, and documentation.