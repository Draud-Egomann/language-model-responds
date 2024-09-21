import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("h2oai/h2o-danube3-500m-base")

model = AutoModelForCausalLM.from_pretrained(
    "h2oai/h2o-danube3-500m-base",
    torch_dtype=torch.float32,  # Use float32 since you are on CPU
)

# No .cuda() needed (its nvidia gpu specific)

def generate_npc_response(user_input):
    # Tokenize the input and prepare it for the model
    inputs = tokenizer(user_input, return_tensors="pt")  # Stay on CPU
    
    # Generate response (text prediction)
    outputs = model.generate(**inputs, max_new_tokens=32, do_sample=False)
    
    # Decode the output to human-readable text
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Simple console interaction
while True:
    user_input = input("You: ")
    npc_response = generate_npc_response(user_input)
    print(f"NPC: {npc_response}")
