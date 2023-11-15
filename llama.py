from llama_cpp import Llama

llm = Llama(model_path="./llama-2-7b-chat.Q8_0.gguf")
output = llm("Question: Name the planets in the solar system? Answer: ", max_tokens=1024, stop=['\n'], echo=True)
print(output)