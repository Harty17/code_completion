from transformers import GPT2LMHeadModel, GPT2Tokenizer
import tkinter as tk
from tkinter import scrolledtext

# Load pre-trained model and tokenizer
model_name = 'gpt2'
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

def get_suggestions(prompt):
    try:
        inputs = tokenizer.encode(prompt, return_tensors='pt')
        outputs = model.generate(inputs, max_length=150, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id)
        suggestion = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return suggestion
    except Exception as e:
        return f"An error occurred: {e}"

def on_submit():
    user_input = input_text.get("1.0", tk.END).strip()
    suggestion = get_suggestions(user_input)
    output_text.config(state=tk.NORMAL)
    output_text.delete("1.0", tk.END)
    output_text.insert(tk.END, suggestion)
    output_text.config(state=tk.DISABLED)

# Set up the GUI
root = tk.Tk()
root.title("AI Code Assistant")

# Input field
input_label = tk.Label(root, text="Enter your code prompt:")
input_label.pack()
input_text = scrolledtext.ScrolledText(root, height=10, width=50)
input_text.pack()

# Submit button
submit_button = tk.Button(root, text="Get Suggestions", command=on_submit)
submit_button.pack()

# Output field
output_label = tk.Label(root, text="AI Suggestions:")
output_label.pack()
output_text = scrolledtext.ScrolledText(root, height=10, width=50, state=tk.DISABLED)
output_text.pack()

# Run the GUI event loop
root.mainloop()
