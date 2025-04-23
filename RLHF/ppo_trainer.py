from trl import PPOTrainer, PPOConfig
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM
import torch
from json_reader import JSONFilter

class PPOtTrainer():
    def __init__(self, model_name, feedback_type):
        self.model_name = model_name
        self.feedback_type = feedback_type
        self.model_path = "../models/student_" + model_name + "_model.pt"
        self.reward_model = self._load_reward_model("../models/" + model_name + "_reward_model.pth")
        self.train_dataset = self._load_data()
        self.policy_model = self._load_policy_model()
        self.tokenizer = self._load_tokenizer()

    def _load_reward_model(self, name):
        reward_model = AutoModelForSequenceClassification.from_pretrained("gpt2")
        reward_model.load_state_dict(torch.load(name))
        return reward_model
    
    def _load_data(self):
        obj = JSONFilter(self.feedback_type)
        data = obj.data_filter(self.model_name)
        return data
    
    def _load_policy_model(self):
        policy_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-Coder-0.5B")
        policy_model.load_state_dict(torch.load(self.model_path))
        return policy_model

    def _load_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-0.5B")
        return tokenizer

    def train(self):
        config = PPOConfig(
            learning_rate=1e-5,
            batch_size=4,        
            mini_batch_size=2,     
            gradient_accumulation_steps=1,
            num_ppo_epochs=4,       
            gamma=1.0,
            lam=0.95,             
            cliprange=0.2,
            cliprange_value=0.2,
            vf_coef=0.1,
            kl_coef=0.05,          
            seed=42
        )
        ppo_trainer = PPOTrainer(
            args=config,                 
            processing_class = self.tokenizer,   
            model = self.policy_model,           
            ref_model = None,                
            reward_model = self.reward_model,     
            train_dataset = self.train_dataset    
        )

        torch.save(self.policy_model.state_dict(), self.model_path)
