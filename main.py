import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("h2oai/h2o-danube2-1.8b-base")

model = AutoModelForCausalLM.from_pretrained(
    "h2oai/h2o-danube2-1.8b-base",
    torch_dtype=torch.bfloat16,
)

# No .cuda() needed (its nvidia gpu specific)

def generate_npc_response(user_input, persona="friendly"):
    # Add context/personality to the prompt
    prompt = f"The NPC is very {persona}. It responds as follows:\nYou: {user_input}\nNPC:"
    
    # Tokenize and generate response
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=38, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    
    # Extract only the newly generated tokens (after the input tokens)
    response_tokens = outputs[0][inputs['input_ids'].shape[-1]:]
    
    # Decode the output to human-readable text
    response = tokenizer.decode(response_tokens, skip_special_tokens=True)
    return response

# Simple console interaction
while True:
    user_input = input("You: ")
    npc_response = generate_npc_response(user_input)
    print(f"NPC: {npc_response}")
