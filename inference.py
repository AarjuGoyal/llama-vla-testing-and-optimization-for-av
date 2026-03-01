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
import pandas as pd

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


def run_inference(generator, prompts, batch_size, max_gen_len, temperature, top_p, generate_quality_results=False):
    print("\n" + "#"*50)

    ##Warm up
    print("\n Warming up")
    for i in range(0,5):
        _ = list(generator.completion(
            prompts[0:batch_size],
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p
        ))

    print("\nRunning inference")

    inference_performance_results = []


    for i in range(0, len(prompts), batch_size):
        prompt_batch = prompts[i:i+batch_size]
        print(f"Prompts: {prompt_batch}")
        inference_start = time.time()
        generated_tokens = [0]*batch_size
        full_generation = [""]*batch_size
        unique_ratio_list = []
        repetition_score_list = []
        quality_score_list = []
        # print(f"Prompt: {prompt}")
        torch.cuda.synchronize()
        start_event = torch.cuda.Event(enable_timing = True)
        end_event = torch.cuda.Event(enable_timing = True)
        start_event.record()
        for results in generator.completion(
            prompt_batch,
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
        ):
            for b in range(batch_size):
                if not results[b].finished:
                    full_generation[b] += results[b].text
                    generated_tokens[b] += 1

            # for b in range(batch_size):
            #     token_text = results[b].text
            #     full_generation[b] += token_text
            #     generated_tokens[b] += 1
            # token_text = results[0].text
            # if results[0].finished:
            #     break
            # print(token_text, end="", flush=True)
            # full_generation += token_text
            # generated_tokens += 1
        
        end_event.record()
        torch.cuda.synchronize()
            

        print()
        elapsed_time_ms = start_event.elapsed_time(end_event)
        total_generated_tokens = (np.array(generated_tokens)).sum()
        inference_time = time.time() - inference_start
        tokens_per_sec = total_generated_tokens / inference_time if inference_time > 0 else 0
        latency_per_token = (inference_time/total_generated_tokens)*1000 ##in ms

        performance_result = {
            "max_gen_len": max_gen_len,
            "total_cpu_measured_time": inference_time,
            "total_gpu_time_ms": elapsed_time_ms,
            "total_tokens": total_generated_tokens,
            "tokens_throughput": tokens_per_sec,
            "latency": latency_per_token,
            "avg_tokens_per_prompt": total_generated_tokens/ batch_size
        }


        if generate_quality_results == True:
            
            for b in range(batch_size):
                print(f"\nFor prompt {b+1}, full text is {full_generation[b]}")
                print(f"Generated tokens {generated_tokens[b]}")
                prompt_quality_result = analyse_quality(full_generation[b])
                unique_ratio_list.append(prompt_quality_result['unique_ratio'])
                repetition_score_list.append(prompt_quality_result['repetition_score'])
                quality_score_list.append(prompt_quality_result['quality_score'])

            print(f"Uniue ratio list {unique_ratio_list}")
            
            quality_result = {
                "unique_ratio": np.average(unique_ratio_list),
                "repetition_score": np.average(repetition_score_list),
                "quality_score": np.average(quality_score_list)
            }
        
            batch_result = performance_result | quality_result
        
        else:
            batch_result = performance_result
        
        
        
        inference_performance_results.append(batch_result)

        print(f"Number of entries in the performance results for gen len of {max_gen_len} is {len(inference_performance_results)}")

    #     # if torch.cuda.is_available():
    #     #     print(f"\n GPU Memory after inference:")
    #     #     print(f"  Allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")
    #     #     print(f"  Reserved: {torch.cuda.memory_reserved()/1e9:.2f} GB")

    # print_metrics(list_inference_times, list_tokens_per_sec, list_latency_per_token)
    return inference_performance_results

def print_results(label, performance_results, quality_results = None):

    print("="*50)
    print(f"{label}: ")
    # # print(f" Model load time: {load_time}")
    # print(f" Inference Timess List: {list_inference_times}")
    # print(f" Tokens per sec List: {list_tokens_per_sec}")
    # print(f" Latencty per token list: {list_latency_per_token}") 

    for idx in range(len(performance_results)):
        r = performance_results[idx]
        print(f"  Generated: {r['total_tokens']} tokens ({r['avg_tokens_per_prompt']:.1f} avg/prompt)")
        print(f"  Total CPU time: {r['total_cpu_measured_time']:.2f}s")
        print(f"  Total GPU time: {r['total_gpu_time_ms']:.2f}ms")
        print(f"  Throughput: {r['tokens_throughput']:.2f} tokens/sec")
        print(f"  Latency: {r['latency']:.2f}ms/token")
        if quality_results:
            q = quality_results[idx]            
            print(f" Unique word ratio: {q['unique_word_ratio']}")
            print(f" Repetition Score: {q['repetition_score']}")
            print(f" Token Count: {q['token_count']}")
            print(f" Loop Detected: {q['loop_detected']}")
            print(f" Quality Score: {q['quality_score']}")    

# Manually quantize after loading
def quantize_model_int8(model):
    """Quantize all Linear layers to INT8"""
    return torch.quantization.quantize_dynamic(
        model,
        {torch.nn.Linear},  # Only quantize Linear layers
        dtype=torch.qint8
    )

#Create the test scenario and test how performance scales with generation length
def test_sequence_length_scaling(llama3_generator, batch_size, results_file, generate_quality_results = False):

    prompt = "You are driving and autonomous vehicle. You are at a 4-way stop sign. A pedestrian is crossing street ahead."

    test_configs = [
        {"max_gen_len": 16, "label": "Short (16 tokens)"},
        {"max_gen_len": 32, "label": "Medium (32 tokens)"},
        {"max_gen_len": 64, "label": "Long (64 tokens)"},
        {"max_gen_len": 128, "label": "Very Long (128 tokens)"},
        {"max_gen_len": 256, "label": "Extended (256 tokens)"},
    ]
    temperature = 0.3 #Between 0-1 control randomness
    top_p = 0.9 #nucleus sampling 
    prompt_sequence =[prompt]*4

    all_results = []
    

    for config in test_configs:
        results = run_inference(generator = llama3_generator, prompts = prompt_sequence,
                            batch_size = batch_size, max_gen_len = config["max_gen_len"],
                            temperature = temperature, top_p = top_p, generate_quality_results = generate_quality_results)
        for r in results:
            r['config_label'] = config['label']
            r['max_gen_len'] = config['max_gen_len']

        all_results.extend(results)
        
        # print_results(config["label"], performance_results)
    df = pd.DataFrame(all_results)
    df.to_csv(results_file, index=False)

def basic_test(llama3_generator,batch_size = 1, generate_quality_results = False):
    prompts = [
    # Simple completion - baseline speed
    "The future of autonomous vehicles",
    
    # Medium - structured reasoning
    "Explain the benefits of electric vehicles in three points:",

    #Complex driving scenario
    "You are driving an ego vehicle. There is a pedestrian on the left sidewalk, there is a red SUV approaching from the opposite side, I'm at a 4-way stop sign, a bicylist is moving in the same direction behind me.",

    "The light is green and I want to make a left turn."
    ]

    temperature = 0.3 #Between 0-1 control randomness
    top_p = 0.9 #nucleus sampling 
    max_gen_len = 64

    performance_results = run_inference(generator = llama3_generator, prompts = prompts, batch_size = batch_size, max_gen_len = max_gen_len, temperature = temperature, top_p = top_p, generate_quality_results = generate_quality_results)
    print_results("BASELINE METRICS", performance_results)

def analyse_quality(generated_text):
    """Simple Quality Metrics"""

    #1. Words repition detection
    words = generated_text.lower().split()
    unique_ratio = len(set(words)) / len(words)

    #2. Detect repetitive n-grams
    trigrams = [' '.join(words[i:i+3]) for i in range(len(words) - 2)]
    repeated_trigrams = len(trigrams) - len(set(trigrams))
    repetition_score = repeated_trigrams / len(trigrams) if trigrams else 0

    #3. Length (too short -> cut off, too long with repetions -> stuck in a loop/ halucinating/ rambling)
    token_count = len(words)

    #4. Check for loops (exact phrase repetiontion)
    sentences = generated_text.split('.')
    loop_detected = (len(sentences) != len(set(sentences)))

    quality_score = unique_ratio * (1 - repetition_score) # Combined metric
    # print(f"For token gneerated of config {label}")
    print(f" Unique word ratio: {unique_ratio}")
    print(f" Repetition Score: {repetition_score}")
    print(f" Token Count: {token_count}")
    print(f" Loop Detected: {loop_detected}")
    print(f" Quality Score: {quality_score}") 
    
    return {
        'unique_ratio': unique_ratio,
        'repetition_score': repetition_score,
        'token_count': token_count,   
        'loop_dtected': loop_detected,
        'quality_score': quality_score
    }



def main():

    ckpt_dir = str(Path.home() / ".llama/checkpoints/Llama3.2-3B")
    tokenizer_path = str(Path.home() / ".llama/checkpoints/Llama3.2-3B/tokenizer.model")    
    quantization_mode = None #QuantizationMode.fp8_mixed
    max_batch_size = 8
    llama3_generator = load_model(ckpt_dir, max_batch_size = max_batch_size, quantization_mode = quantization_mode)
    
    results_file_name = "results_seq_scaling_batch_1.csv"
    test_sequence_length_scaling(llama3_generator, batch_size=1,results_file = results_file_name, generate_quality_results= True)

    # basic_test(llama3_generator, batch_size = 1,generate_quality_results= True)

if __name__ == "__main__":
    main()