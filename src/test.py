from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from peft import PeftModel

base_model = "TinyLlama/TinyLlama_v1.1"
adapter_path = "./tinyllama-finetuned/checkpoint-216"

# tokenizer comes from the base (your adapter didn't change the vocab)
tokenizer = AutoTokenizer.from_pretrained(base_model)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# load the base model (the engine)...
model = AutoModelForCausalLM.from_pretrained(base_model)

# ...then stick your trained adapter on top!!
model = PeftModel.from_pretrained(model, adapter_path)

pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

prompt = "Was there any World Cup final that ended because of a fire?"
print(pipe(prompt, max_new_tokens=100)[0]["generated_text"])