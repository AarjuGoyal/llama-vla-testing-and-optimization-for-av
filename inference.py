"""
Llama 3.2-3B Inference Profiling and Optimization

Requires llama-models library:
https://github.com/meta-llama/llama-models

Model weights must be downloaded separately from Meta

Author: Aarju Goyal
"""

import torch
import time
from pathlib import Path
from llama_models.llama3.generation import Llama3 as Llama
# from llama_models.datatypes import QuantizationMode
import torch.cuda.profiler as profiler
import numpy as np

def load_model(ckpt_dir, max_batch_size, quantization_mode = None):
    print("Loading model.. ")
    start = time.time()
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        max_seq_len=512,
        max_batch_size=max_batch_size,
        quantization_mode = quantization_mode
    )
    load_time = time.time() - start
    print(f"Model loaded in {load_time:.2f}s")
    if torch.cuda.is_available():
        print(f"\n GPU Memory:")
        print(f"  Allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")
        print(f"  Reserved: {torch.cuda.memory_reserved()/1e9:.2f} GB")
    return generator


def run_inference(generator, prompts, batch_size, temperature, top_p):
    print("\n" + "#"*50)
    print("Running inference")

    list_inference_times = []
    list_tokens_per_sec = []
    list_latency_per_token = []

    for i in range(0, len(prompts), batch_size):
        prompt_batch = prompts[i:i+batch_size]
        print(f"Prompts: {prompt_batch}")
        inference_start = time.time()
        generated_tokens = [0]*batch_size
        full_generation = [""]*batch_size
        # print(f"Prompt: {prompt}")
        torch.cuda.synchronize()
        start_event = torch.cuda.Event(enable_timing = True)
        end_event = torch.cuda.Event(enable_timing = True)
        start_event.record()
        for results in generator.completion(
            prompt_batch,
            max_gen_len=64,
            temperature=temperature,
            top_p=top_p,
        ):
            for b in range(batch_size):
                token_text = results[b].text
                full_generation[b] += token_text
                generated_tokens[b] += 1
            # token_text = results[0].text
            # if results[0].finished:
            #     break
            # print(token_text, end="", flush=True)
            # full_generation += token_text
            # generated_tokens += 1
        
        end_event.record()
        torch.cuda.synchronize()
        for b in range(batch_size):
            print(f"\nFor prompt {b+1}, full text is {full_generation[b]}")
            print(f"Generated tokens {generated_tokens[b]}")

        print()
        elapsed_time_ms = start_event.elapsed_time(end_event)
        print(f"Pure GPU time: {elapsed_time_ms:.2f}ms")
        
        generated_tokens = np.array(generated_tokens)
        inference_time = time.time() - inference_start
        tokens_per_sec = generated_tokens.sum() / inference_time if inference_time > 0 else 0
        latency_per_token = (inference_time/generated_tokens.sum())*1000 ##in ms
        
        print(f"Inference time {inference_time}")
        print(f"Tokens per sec {tokens_per_sec}")
        print(f"Latency per token {latency_per_token}")
    #     list_inference_times.append(inference_time)
    #     list_tokens_per_sec.append(tokens_per_sec)
    #     list_latency_per_token.append(latency_per_token)

    #     # if torch.cuda.is_available():
    #     #     print(f"\n GPU Memory after inference:")
    #     #     print(f"  Allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")
    #     #     print(f"  Reserved: {torch.cuda.memory_reserved()/1e9:.2f} GB")

    # print_metrics(list_inference_times, list_tokens_per_sec, list_latency_per_token)

def print_metrics(list_inference_times, list_tokens_per_sec, list_latency_per_token):

    print("="*50)
    print("BASELINE METRICS:")
    # print(f" Model load time: {load_time}")
    print(f" Inference Timess List: {list_inference_times}")
    print(f" Tokens per sec List: {list_tokens_per_sec}")
    print(f" Latencty per token list: {list_latency_per_token}") 

# Manually quantize after loading
def quantize_model_int8(model):
    """Quantize all Linear layers to INT8"""
    return torch.quantization.quantize_dynamic(
        model,
        {torch.nn.Linear},  # Only quantize Linear layers
        dtype=torch.qint8
    )
def main():

    ckpt_dir = str(Path.home() / ".llama/checkpoints/Llama3.2-3B")
    tokenizer_path = str(Path.home() / ".llama/checkpoints/Llama3.2-3B/tokenizer.model")

    prompts = [
    # Simple completion - baseline speed
    "The future of autonomous vehicles",
    
    # Medium - structured reasoning
    "Explain the benefits of electric vehicles in three points:",

    #Complex driving scenario
    "You are driving an ego vehicle. There is a pedestrian on the left sidewalk, there is a red SUV approaching from the opposite side, I'm at a 4-way stop sign, a bicylist is moving in the same direction behind me.",

    "The light is green and I want to make a left turn."
    ]
    batch_size = 4
    temperature = 0.3 #Between 0-1 control randomness
    top_p = 0.9 #nucleus sampling 
    quantization_mode = None #QuantizationMode.fp8_mixed
    llama3_generator = load_model(ckpt_dir, max_batch_size = batch_size, quantization_mode = quantization_mode)
    run_inference(llama3_generator, prompts, batch_size, temperature, top_p)


if __name__ == "__main__":
    main()