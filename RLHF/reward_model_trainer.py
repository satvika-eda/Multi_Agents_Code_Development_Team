from transformers import AutoModelForSequenceClassification, AdamW
import torch
from json_reader import JSONFilter
from torch.utils.data import DataLoader

class RewardModelTrainer():
    def __init__(self, model_name, feedback_type):
        self.model_name = model_name
        self.feedback_type = feedback_type
        self.model_path = "../models/" + model_name + "_reward_model.pth"
        self.reward_model = self._load_model(self.model_path)
        self.data = self._load_data()

    def _load_model(self):
        reward_model = AutoModelForSequenceClassification.from_pretrained("gpt2", num_labels=1)
        reward_model.load_state_dict(torch.load("../models/generator_reward_model.pth"))

    def _load_data(self):
        obj = JSONFilter(self.feedback_type)
        data = obj.data_filter(self.model_name)
        return data

    def train(self):
        optimizer = AdamW(self.reward_model.parameters(), lr=5e-5)
        train_loader = DataLoader(self.data, batch_size=2, shuffle=True)
        for epoch in range(5):
            total_loss = 0
            for batch in train_loader:
                inputs = {k: v for k, v in batch.items() if k != "labels"}
                labels = batch["labels"].unsqueeze(1)

                outputs = self.reward_model(**inputs, labels=labels)
                loss = outputs.loss

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                total_loss += loss.item()
            print(f"Epoch {epoch+1}: Loss = {total_loss / len(train_loader):.4f}")
        
        torch.save(self.reward_model.state_dict(), self.model_path)


