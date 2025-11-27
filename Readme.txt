Here's a Readme to load checkpoints for inference.
For both Qwen3-4b and Qwen3-8b we have checkpoints saved in .safetensor format. In order to load them you can either access the Finetuning notebooks and run each cell to fine tune, run the inference notebooks to run the inference for the results or simply load checkpoints by using followinf code ->
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

model_name = "Qwen/Qwen3-4B"
lora_path = "<Path to checkpoint saved>"

bnb_config = BitsAndBytesConfig(load_in_4bit=True)

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    quantization_config=bnb_config,
    torch_dtype="auto",
    trust_remote_code=False,
)

model = PeftModel.from_pretrained(
    model,
    lora_path,
    is_trainable=False
)

model.eval()
