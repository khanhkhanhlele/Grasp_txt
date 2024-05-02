import torch
import torch.nn as nn
from LLaVA.llava.model.builder import load_pretrained_model
from LLaVA.llava.mm_utils import get_model_name_from_path

class LLaVa(nn.Module):
    def __init__(self, model_path):
        super(LLaVa, self).__init__()
        tokenizer, model, image_processor, context_len = load_pretrained_model(
            model_path=model_path,
            model_base="liuhaotian/llava-v1.5-7b",
            model_name=get_model_name_from_path(model_path),
            device="cuda:0"
        )
        self.tokenizer = tokenizer
        self.llava = model
        self.image_processor = image_processor
        self.context_len = context_len
        self.model_config = model.config
        
        self.adapter = nn.ModuleList([
            nn.Linear(4096, 2048),
            nn.ReLU(),
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
        ])
        
        self.adapter = self.adapter.to("cuda:0")
        
    def forward(self, input_ids, image_tensor, attn_mask):
        outputs = self.llava(
            input_ids=input_ids,
            images=image_tensor,
            attention_mask=attn_mask,
            output_hidden_states=True,
        )
        last_hidden_states = outputs.hidden_states[-1]     
        outputs = last_hidden_states.mean(dim=1)
        
        # gather the vectors from the last hidden layer with indexes
        # corresponding to the context tokens
        stp_range_vectors = []
        STP_indexes = (input_ids == 29911).nonzero(as_tuple=True)[1]
        
        baselines = []
        for i in range(input_ids.shape[0]):
            baselines.append((input_ids[i] != 2).nonzero(as_tuple=True)[0][0].item())
        
        i = 0
        for j in range(input_ids.shape[0]):
            stp_range_vectors.append(last_hidden_states[j, STP_indexes[i]+1-baselines[j]:STP_indexes[i+1]-baselines[j]].mean(dim=0))
            i+=2
        
        outputs = torch.stack(stp_range_vectors, dim=1).T
        for layer in self.adapter:
            outputs = layer(outputs.float().to(self.adapter[0].weight.device))
        
        return outputs