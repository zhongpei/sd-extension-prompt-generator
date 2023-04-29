import os.path

from transformers import GPT2Tokenizer, GPT2LMHeadModel
from typing import List
import gradio as gr
import random
from huggingface_hub import hf_hub_download
import re
from modules import scripts, script_callbacks, shared
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE_ID = 0 if torch.cuda.is_available() else -1
ui_prompts = []

MODEL_NAME = {
    "normal": 'FredZhang7/distilgpt2-stable-diffusion-v2',
    "anime": 'FredZhang7/anime-anything-promptgen-v2',
}
model = None


def clean_tags(tags):
    # Make tags more human readable
    tags = tags.replace(' ', ', ').replace('_', ' ')

    # Remove "!", "?", ".", "(", ")" from the tags
    tags = re.sub(r"[!.?()]", "", tags)

    # Replace " , " with an empty space
    tags = re.sub(r" , ", " ", tags)

    # Remove any trailing commas
    tags = re.sub(r"^,|,$", "", tags)

    # Strip spaces
    tags = tags.strip()

    # Remove any usernames
    words = tags.split(", ")
    result = []
    for word in words:
        word = word.strip()
        if word == 'v':
            result.append(word)
            continue
        if len(word) < 2:
            continue
        # if any(char.isdigit() for char in word) and word not in ["1girl", "1boy", "1koma", "1other"]:
        #    continue
        result.append(word)

    return ", ".join(result)


class Model(object):
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.cache = {}
        self.model_name = None
        self.model_id = None
        self.last_result = None
        self.last_prompt = None

    def local_dir(self, model_name):
        local_model_dir = model_name.split("/")[-1]
        local_dir = os.path.join(scripts.basedir(), "models", local_model_dir)
        if not os.path.exists(local_dir):
            os.makedirs(local_dir, exist_ok=True)
        return local_dir

    def download_file(self, model_name, filename):
        local_dir = self.local_dir(model_name)

        hf_hub_download(repo_id=model_name, filename=filename, local_dir=local_dir, resume_download=True)

    def download_model(self, model_name):
        local_dir = self.local_dir(model_name)
        # print(f"local dir: {local_dir}")
        self.download_file(model_name=model_name, filename="model.safetensors")
        self.download_file(model_name=model_name, filename="tokenizer.json")
        self.download_file(model_name=model_name, filename="config.json")

    def reset_model(self):
        if self.model is not None:
            del self.model
            self.model = None
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        self.cache = {}
        self.model_name = None
        self.model_id = None
        self.last_result = None
        self.last_prompt = None

    def __add_cache(self, prompt, gen_prompts):
        prompt = prompt.strip()
        self.cache.setdefault(prompt, []).extend(gen_prompts)

    def __get_cache(self, prompt):
        if prompt is None:
            return

        prompt = prompt.strip()
        if len(prompt) == 0:
            return

        gen_prompts = self.cache.get(prompt, None)
        if gen_prompts is not None and len(gen_prompts) > 0:
            return gen_prompts.pop()

    def init_model(self, model_id="anime"):
        self.reset_model()

        model_name = MODEL_NAME.get(model_id)
        self.model_name = model_name
        self.download_model(model_name)

        self.download_model('distilgpt2')
        self.download_file('distilgpt2', filename="generation_config.json")
        self.download_file('distilgpt2', filename="generation_config_for_text_generation.json")
        self.download_file('distilgpt2', filename="merges.txt")
        self.download_file('distilgpt2', filename="vocab.json")

        self.model_id = model_id

        tokenizer = GPT2Tokenizer.from_pretrained(self.local_dir('distilgpt2'), local_files_only=True)
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        model = GPT2LMHeadModel.from_pretrained(self.local_dir(model_name), local_files_only=True).to(DEVICE)

        self.model = model
        self.tokenizer = tokenizer

    def gen_prompt(self, prompt: str) -> str:
        p = self.__get_cache(prompt)
        if p is not None:
            return p

        if self.last_result == prompt and self.last_prompt is not None:
            p = self.__get_cache(self.last_prompt)
            if p is not None:
                return p

        prompts = self.gen_prompts(prompt=prompt)
        if len(prompts) == 0:
            return prompt

        result = prompts.pop()
        self.last_prompt = prompt
        self.last_result = result
        if len(prompts) > 0:
            self.__add_cache(prompt=prompt, gen_prompts=prompts)
        return result

    def gen_prompts(self, prompt: str) -> List[str]:
        temperature = 0.9  # a higher temperature will produce more diverse results, but with a higher risk of less coherent text
        top_k = 8  # the number of tokens to sample from at each step
        max_length = 80  # the maximum number of tokens for the output of the model
        repitition_penalty = 1.2  # the penalty value for each repetition of a token
        num_return_sequences = 5  # the number of results to generate

        # generate the result with contrastive search
        input_ids = self.tokenizer(prompt, return_tensors='pt').input_ids.to(DEVICE)
        output = self.model.generate(
            input_ids,
            do_sample=True,
            temperature=temperature,
            top_k=top_k,
            max_length=max_length,
            num_return_sequences=num_return_sequences,
            repetition_penalty=repitition_penalty,
            penalty_alpha=0.6,
            no_repeat_ngram_size=1,
            early_stopping=True,
        )

        prompt_output = []
        for i in range(len(output)):
            p = self.tokenizer.decode(output[i], skip_special_tokens=True, device=DEVICE_ID)
            prompt_output.append(p)

        prompt_output = list(set(prompt_output))

        if self.model_id == "anime":
            prompt_output = [clean_tags(p) for p in prompt_output]

        print('\nInput:\n' + 100 * '-')
        print('\033[96m' + prompt + '\033[0m')
        print('\nOutput:\n' + 100 * '-')
        for p in prompt_output:
            print('\033[92m' + p + '\033[0m\n')

        return prompt_output


def init_model():
    global model
    model_id = shared.opts.prompt_generator_zh_model_id
    if model is None:
        model = Model()
        model.init_model(model_id=model_id)
    elif model.model_id != model_id:
        print(f"reinit model {model.model_id} ==> {model_id}")
        model.reset_model()
        model.init_model(model_id=model_id)


def gen_prompt(*ui_prompts: list):
    global model
    init_model()
    result_list = []

    for prompt in ui_prompts:

        if not prompt or len(prompt.strip()) == 0:
            result_list.append(prompt)
            print(f"gen prompt(empty): {prompt} --> {prompt}")
            continue

        prompt = prompt.strip()
        result = model.gen_prompt(prompt)
        if result is None:
            result_list.append(prompt)
            print(f"gen prompt(empty): {prompt} --> {prompt}")
            continue

        result = format_prompt_one(result)
        result_list.append(result)

        print(f"gen prompt: {prompt} --> {result}")
    print(f"len ui_prompts: {len(ui_prompts)}  : {len(result_list)}")
    if len(ui_prompts) == 1:
        return result_list[0]
    return result_list


def on_before_component(component: gr.component, **kwargs: dict):
    if 'elem_id' in kwargs:
        # print(f"elem_id: {kwargs['elem_id']}")
        if kwargs['elem_id'] in ['txt2img_prompt', 'img2img_prompt']:
            ui_prompts.append(component)
        elif kwargs['elem_id'] == 'paste':
            with gr.Blocks(analytics_enabled=False) as ui_component:
                button = gr.Button(value='G', elem_classes='tool', elem_id='gen_prompt')
                button.click(
                    fn=gen_prompt,
                    inputs=ui_prompts,
                    outputs=ui_prompts
                )
                return ui_component


SETTINGS_SECTION = ("prompt_generator_zh", "prompt generator ")


def on_ui_settings():
    model_ids = list(MODEL_NAME.keys())
    shared.opts.add_option(
        "prompt_generator_zh_model_id",
        shared.OptionInfo(
            model_ids[0], "model type", gr.Radio,
            lambda: {"choices": model_ids},
            section=SETTINGS_SECTION
        )
    )


script_callbacks.on_ui_settings(on_ui_settings)
script_callbacks.on_before_component(on_before_component)

if __name__ == "__main__":
    m = Model()
    m.download_model(MODEL_NAME.get("anime"))
    m.init_model()
    output = m.gen_prompt("1 girl")
    print(output)
