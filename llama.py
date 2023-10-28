from llama_cpp import Llama

llm = Llama(model_path="./llama-2-7b.Q8_0.gguf")
output = llm("Q: Name the planets in the solar system? A: ", max_tokens=32, stop=["Q:", "\n"], echo=True)
print(output)