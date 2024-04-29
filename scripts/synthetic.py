from vllm import LLM, SamplingParams
import os
import pickle as pkl
from openai import OpenAI
from tqdm import tqdm

class LocalLLM:
    def __init__(self):
        openai_api_key = "aca586edef32c3f95ed93a70830fe9fb0c38c3f408de8054370c501db0c65268"
        openai_api_base = "https://api.together.xyz"

        self.client = OpenAI(
            api_key=openai_api_key,
            base_url=openai_api_base,
        )

    def __call__(self, prompt: str):
        response = self.client.chat.completions.create(
            temperature=0,
            model="mistralai/Mixtral-8x7B-Instruct-v0.1",
            messages=[
                {"role": "system", "content": "You are an helpful assistant"},
                {"role": "user", "content": prompt},
            ],
            max_tokens=None
        )
        return response.choices[0].message.content

template = """Only output the target grasp object name from the input string (do not include any explanation or note). 
For example: 
input: Hold fork at its tines
Target grasp name: tines of fork
inpur: Grasp mug at its handle 
Target grasp name: handle of the mug
Now do with this following input
input: {prompt}
Target grasp name:
"""

def get_raw_instructions(path):
    files = os.listdir(path)
    instructions = []
    for file in files:
        with open(os.path.join(path, file), "rb") as f:
            instruction = pkl.load(f)
        instructions.append(instruction)
    
    return instructions, files

prompts, files = get_raw_instructions("grasp-anything++/seen/grasp_instructions")
prompts = [template.format(prompt=prompt) for prompt in prompts]

llm = LocalLLM()
# Print the outputs.
for prompt, file in tqdm(zip(prompts, files)):
    output = llm(prompt)
    with open(f"grasp-anything++/seen/txt_label/{file.split('.')[0]+'.txt'}", "w") as f:
        f.write(f"Here is [SPT] {output.strip()} [SPT]")