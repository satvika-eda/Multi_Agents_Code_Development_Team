from reward_model_trainer import RewardModelTrainer
from ppo_trainer import PPOTrainer

def update_policy_using_feedback(model_name, feedback_type):
    robj = RewardModelTrainer(model_name, feedback_type)
    robj.train()

    mobj = PPOTrainer(model_name, feedback_type)
    mobj.train()

models_to_update = ["developer", "debugger", "explainer"] 
feedback_types = ["scalar", "preference"]

for model in models_to_update:
    for type in feedback_types:
        update_policy_using_feedback(model, type)