from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load pre-trained model and tokenizer
model_name = 'gpt2'
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

def get_suggestions(prompt):
    inputs = tokenizer.encode(prompt, return_tensors='pt')
    outputs = model.generate(inputs, max_length=50, num_return_sequences=1)
    suggestion = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return suggestion

if __name__ == "__main__":
    user_input = input("Enter your code prompt: ")
    suggestion = get_suggestions(user_input)
    print("\nAI Suggestion:\n", suggestion)
