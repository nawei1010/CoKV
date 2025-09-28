import sys
project_path='your/project/path'
sys.path.append(project_path)
import tiktoken
import os 
import pdb
import glob
import jieba
import random
from tqdm import trange

import json
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
import argparse
from rouge_score import rouge_scorer

import sys
import os

from datetime import datetime, timezone
import time
import torch

from adaptive_kv.monkeypatch.monkeypatch import replace_mistral_adaptive

scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)

class LLMNeedleHaystackTester:
    """
    This class is used to test the LLM Needle Haystack.
    """
    def __init__(self,
                 needle="\nThe best thing to do in San Francisco is eat a sandwich and sit in Dolores Park on a sunny day.\n",
                 haystack_dir="/your/haystack/path", # PaulGrahamEssays  
                 retrieval_question="The best thing to do in San Francisco is: ", 
                 answer = "The best thing to do in San Francisco is eat a sandwich and sit in Dolores Park on a sunny day.",
                 results_version = 1,
                 context_lengths_min = None,
                 context_lengths_max = None,
                 context_lengths_num_intervals = 5,
                 context_lengths = None,
                 document_depth_percent_min = 0,
                 document_depth_percent_max = 100,
                 document_depth_percent_intervals = 10,
                 document_depth_percents = None,
                 document_depth_percent_interval_type = "linear",
                 model_provider = "OpenAI",
                 openai_api_key=None,
                 anthropic_api_key = None,
                 model_name='',
                 model_name_suffix=None,
                 model_version=None, 
                 num_concurrent_requests = 1,
                 save_results = True,
                 save_contexts = True,
                 final_context_length_buffer = 200,
                 seconds_to_sleep_between_completions = None,
                 print_ongoing_status = True, 
                 step=1000, 
                 attn_implementation='flash_attention_2',
                 mask_heads=None):
        """        
        :param needle: The needle to be found in the haystack. Default is None.
        :param haystack_dir: The directory of text files to use as background context (or a haystack) in which the needle is to be found. Default is Paul Graham Essays.
        :param retrieval_question: The question which with to prompt the model to do the retrieval.
        :param results_version: In case you would like to try the same combination of model, context length, and depth % multiple times, change the results version other than 1
        :param num_concurrent_requests: Due to volume, this object is set up to run concurrent requests, default = 1. Be careful of rate limits.
        :param save_results: Whether or not you would like to save your contexts to file. Warning: These will get long! Default = True
        :param save_contexts: Whether or not you would like to save your contexts to file. Warning: These will get long! Default is True.
        :param final_context_length_buffer: The amount of cushion you'd like to leave off the input context to allow for the output context. Default 200 tokens
        :param context_lengths_min: The minimum length of the context. Default is 0.
        :param context_lengths_max: The maximum length of the context. Default is 200000.
        :param context_lengths_num_intervals: The number of intervals for the context length. Default is 35.
        :param context_lengths: The lengths of the context. Default is None.
        :param document_depth_percent_min: The minimum depth percent of the document. Default is 0.
        :param document_depth_percent_max: The maximum depth percent of the document. Default is 100.
        :param document_depth_percent_intervals: The number of intervals for the document depth percent. Default is 35.
        :param document_depth_percents: The depth percentages of the document. Default is None.
        :param document_depth_percent_interval_type: The type of interval for the document depth percent. Must be either 'linear' or 'sigmoid'. Default is 'linear'.
        :param model_provider: The provider of the model. Must be either 'OpenAI' or 'Anthropic'. Default is 'OpenAI'.
        :param openai_api_key: The API key for OpenAI. Default is None.
        :param anthropic_api_key: The API key for Anthropic. Default is None.
        :param model_name: The name of the model. Default is 'gpt-4-1106-preview'.
        :param seconds_to_sleep_between_completions: The number of seconds to sleep between completions. Default is None.
        :param print_ongoing_status: Whether or not to print the ongoing status. Default is True.
        """
        if not needle or not haystack_dir or not retrieval_question:
            raise ValueError("Needle, haystack, and retrieval_question must be provided.")
        
        self.needle = needle
        self.answer = answer
        self.haystack_dir = haystack_dir
        self.retrieval_question = retrieval_question
        self.results_version = results_version
        self.num_concurrent_requests = num_concurrent_requests
        self.save_results = save_results
        self.final_context_length_buffer = final_context_length_buffer
        self.save_contexts = save_contexts
        self.seconds_to_sleep_between_completions = seconds_to_sleep_between_completions
        self.print_ongoing_status = print_ongoing_status
        self.model_provider = model_provider
        self.testing_results = []
        self.step = step
        self.attn_implementation = attn_implementation


        self.model_version = model_version
        if(model_name_suffix is not None): self.model_version += "_" + model_name_suffix

        if context_lengths is None:
            if context_lengths_min is None or context_lengths_max is None or context_lengths_num_intervals is None:
                raise ValueError("Either context_lengths_min, context_lengths_max, context_lengths_intervals need to be filled out OR the context_lengths_list needs to be supplied.")
            else:
                # self.context_lengths = np.round(np.linspace(context_lengths_min, context_lengths_max, num=context_lengths_num_intervals, endpoint=True)).astype(int)
                self.context_lengths = np.arange(context_lengths_min, context_lengths_max+1, step=self.step)
        else:
            self.context_lengths = context_lengths


        if document_depth_percents is None:
            if document_depth_percent_min is None or document_depth_percent_max is None or document_depth_percent_intervals is None:
                raise ValueError("Either document_depth_percent_min, document_depth_percent_max, document_depth_percent_intervals need to be filled out OR the document_depth_percents needs to be supplied.")
            else:
                if document_depth_percent_interval_type == 'linear':
                    self.document_depth_percents = np.round(np.linspace(document_depth_percent_min, document_depth_percent_max, num=document_depth_percent_intervals, endpoint=True)).astype(int)
                elif document_depth_percent_interval_type == 'sigmoid':
                    self.document_depth_percents = [self.logistic(x) for x in np.linspace(document_depth_percent_min, document_depth_percent_max, document_depth_percent_intervals)]
        else:
            self.document_depth_percents = document_depth_percents

        if document_depth_percent_interval_type not in [None, "linear", "sigmoid"]:
            raise ValueError("document_depth_percent_interval_type must be either None, 'linear' or 'sigmoid'. If you'd like your own distribution give a list of ints in via document_depth_percent_intervals")
        
        self.model_name = model_name

        if(self.model_provider in ["LLaMA3", "Mistral"]):
            self.enc = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            # self.enc.add_special_tokens({'pad_token': '[PAD]'})
            print("loading from %s" % model_name)

            # NOTE: Compress config
            compress_args = {}
            compress_args = json.load(open(os.path.join('config', args.compress_args_path), "r"))
            compress_args['window_size'] = 8
            compress_args['floor_alpha'] = args.floor_alpha
            compress_args['gqa_support'] = args.gqa_support
            compress_args['normalize'] = args.normalize
            compress_args['pyram_mode']= args.pyram
            compress_args['skip'] = args.skip
            compress_args['pyram_beta'] = args.pyram_beta
            compress_args['given_adaptive_size'] = torch.ones((32,8))*120
                

            def config_compress(model, window_size=32, base_capacity=512, kernel_size=7, pooling="maxpool", floor_alpha=0.5, pyram_mode = False, pyram_beta = 20, normalize=True, skip=0, gqa_support=False,given_adaptive_size=None,mask_heads=None):
                model.model.config.window_size = window_size
                model.model.config.base_capacity = base_capacity
                model.model.config.kernel_size = kernel_size

                model.model.config.normalize = normalize
                model.model.config.pooling = pooling
                model.model.config.floor_alpha = floor_alpha

                model.model.config.pyram_mode = pyram_mode
                model.model.config.pyram_beta = pyram_beta
                model.model.config.skip = skip
                model.model.config.gqa_support = gqa_support
                model.model.config.given_adaptive_size = given_adaptive_size
                return model

            replace_mistral_adaptive()

            self.model_to_test=AutoModelForCausalLM.from_pretrained(
                model_name, 
                torch_dtype=torch.float16,
                attn_implementation=self.attn_implementation,
                device_map="auto",
                low_cpu_mem_usage=True, 
                # use_cache=False
                ).eval()
            self.model_to_test = config_compress(self.model_to_test, **compress_args)
                
                # self.model_to_test = tp.tensor_parallel(self.model_to_test, sharded=True)

        else:raise ValueError("model_provider must be either 'LLaMA3' or 'Mistral'")

    
    def change_given_adaptive_size(self,given_adaptive_size):
        self.model_to_test.config.given_adaptive_size = given_adaptive_size
            

    def logistic(self, x, L=100, x0=50, k=.1):
        if x == 0:
            return 0
        if x == 100:
            return 100
        return np.round(L / (1 + np.exp(-k * (x - x0))), 3)
    
    def bound_evaluate_and_log(self, *args):
        score = self.evaluate_and_log(*args)
        return score

    def run_test(self, args):

        # Run through each iteration of context_lengths and depths
        sum = 0
        cnt = 0
        for context_length in self.context_lengths:
            if context_length < args.s_len or context_length > args.e_len: continue
            for depth_percent in self.document_depth_percents:
                score = self.bound_evaluate_and_log(context_length, depth_percent)
                sum += score
                cnt += 1
        return sum/cnt

    def generate_prompt(self, context):
        # Generate the prompt for the Anthropic model
        # Replace the following line with the appropriate prompt structure
        if(self.model_provider not in ["OpenAI", "Anthropic"]):
            test_format=f"<|im_start|> This is a very long story book: <book> {context} </book>.\n Based on the content of the book, Question: {self.retrieval_question}\nAnswer:"
            return test_format
        else: 
            return [
                {
                    "role": "system",
                    "content": "You are a helpful AI bot that answers questions for a user. Keep your response short and direct"
                },
                {
                    "role": "user",
                    "content": context
                    },
                {
                    "role": "user",
                    "content": f"{self.retrieval_question} Don't give information outside the document or repeat your findings. The document definitely contains the answer, and I'm 100% sure. So try your best to find it."
                },
                {
                    "role": "assistant",
                    "content":"",
                },
                
            ]

    def evaluate_and_log(self, context_length, depth_percent):
        # Checks to see if you've already checked a length/percent/version.
        
        context = self.generate_context(context_length, depth_percent)

        print('depth_percent: ', depth_percent)

        # Prepare your message to send to the model you're going to evaluate
        prompt = self.generate_prompt(context)
        test_start_time = time.time()
        
        if(self.model_provider in ["LLaMA3", "Mistral"]):

            prompt = self.enc(prompt, return_tensors="pt")
            input_ids = prompt['input_ids'].to(self.model_to_test.device)
            
            output_ids = self.model_to_test.generate(
                input_ids, 
                output_attentions=False,
                max_new_tokens=30,
                num_beams=1,
                do_sample=False,
                temperature=1.0,
                eos_token_id=[self.enc.eos_token_id, self.enc.encode("\n", add_special_tokens=False)[-1]]
            )
            response = self.enc.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True).strip()

        
        #print(response)
        test_end_time = time.time()
        test_elapsed_time = test_end_time - test_start_time
        if len(response) != 0:
            answer = "eat a sandwich and sit in Dolores Park on a sunny day."
            expected_answer = "eat a sandwich and sit in Dolores Park on a sunny day.".lower().split()
            model_response = response[:len(answer)].lower()
            score = len(set(model_response.split()).intersection(set(expected_answer))) / len(set(expected_answer))
        else:
            score = 0.0

        results = {
            'model' : self.model_name,
            'context_length' : int(context_length),
            'depth_percent' : float(depth_percent),
            'version' : self.results_version,
            'needle' : self.needle,
            'model_response' : response,
            'score' : score,
            'test_duration_seconds' : test_elapsed_time,
            'test_timestamp_utc' : datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S%z'), 
        }

        self.testing_results.append(results)

        return score

    

    def generate_context(self, context_length, depth_percent):
        # Load up tiktoken so we navigate tokens more easily

        # Get your Paul Graham files loaded into a string
        context = self.read_context_files()

        # Truncate the Paul Graham essays to the context length you desire
        context = self.encode_and_trim(context, context_length)

        # Insert your random statement according to your depth percent
        context = self.insert_needle(context, depth_percent, context_length)

        return context
    
    def encode_text_to_tokens(self, text):
        if self.model_provider in ["Mistral", "LLaMA3"]:
            return self.enc.encode(text)
        elif self.model_provider == "Anthropic":
            # Assuming you have a different encoder for Anthropic
            return self.enc.encode(text).ids
        else:
            return self.enc.encode(text)
            raise ValueError("model_provider must be either 'OpenAI' or 'Anthropic'")
    
    def insert_needle(self, context, depth_percent, context_length):
        tokens_needle = self.encode_text_to_tokens(self.needle)
        tokens_context = self.encode_text_to_tokens(context)

        # Reducing the context length by 150 buffer. This is to account for system message, the user question, and response.
        context_length -= self.final_context_length_buffer

        # If your context + needle are longer than the context length (which it will be), then reduce tokens from the context by the needle length
        if len(tokens_context) + len(tokens_needle) > context_length:
            tokens_context = tokens_context[:context_length - len(tokens_needle)]

        if depth_percent == 100:
            # If your depth percent is 100 (which means your needle is the last thing in the doc), throw it at the end
            tokens_new_context = tokens_context + tokens_needle
        else:
            # Go get the position (in terms of tokens) to insert your needle
            insertion_point = int(len(tokens_context) * (depth_percent / 100))

            # tokens_new_context represents the tokens before the needle
            tokens_new_context = tokens_context[:insertion_point]

            # We want to make sure that we place our needle at a sentence break so we first see what token a '.' is
            if(self.model_provider in ["LLaMA", "LongLLaMA"]): period_tokens = [29889, 869]
            elif(self.model_provider == "LLaMA3"): period_tokens = [13]
            elif(self.model_provider == "Mistral"): period_tokens = [842, 28723]
            elif(self.model_provider == "GLM"): period_tokens = [918, 30930]
            else: period_tokens = self.encode_text_to_tokens('.')
            
            # Then we iteration backwards until we find the first period
            while tokens_new_context and tokens_new_context[-1] not in period_tokens:
                insertion_point -= 1
                tokens_new_context = tokens_context[:insertion_point]

            print("insertion at %d" % insertion_point)
            # Once we get there, then add in your needle, and stick the rest of your context in on the other end.
            # Now we have a needle in a haystack
            tokens_new_context += tokens_needle + tokens_context[insertion_point:]

        # Convert back to a string and return it
        new_context = self.decode_tokens(tokens_new_context)
        return new_context

    def get_context_length_in_tokens(self, context):
        if self.model_provider in ["Mistral", "LLaMA3"]:
            return len(self.enc.encode(context))
        else:
            return len(self.enc.encode(context))
            raise ValueError("model_provider must be either 'OpenAI' or 'Anthropic'")

    def read_context_files(self):
        context = ""
        max_context_length = max(self.context_lengths)

        while self.get_context_length_in_tokens(context) < max_context_length:
            for file in glob.glob(f"{self.haystack_dir}/*.txt"):
                with open(file, 'r') as f:
                    context += f.read()
        return context

    def get_tokens_from_context(self, context):
        if self.model_provider in ["Mistral", "LLaMA3"]:
            return self.enc.encode(context)
        else:
            return self.enc.encode(context)
            # raise ValueError("model_provider must be either 'OpenAI' or 'Anthropic'")
        
    def decode_tokens(self, tokens, context_length=None):
        if self.model_provider in ["Mistral", "LLaMA3"]:
            return self.enc.decode(tokens[:context_length])
        else:
            return self.enc.decode(tokens[:context_length])
            # raise ValueError("model_provider must be either 'OpenAI' or 'Anthropic'")

    def encode_and_trim(self, context, context_length):
        tokens = self.get_tokens_from_context(context)
        if len(tokens) > context_length:
            context = self.decode_tokens(tokens, context_length)
        return context
    
    def get_results(self):
        return self.testing_results
    
    def print_start_test_summary(self):
        print ("\n")
        print ("Starting Needle In A Haystack Testing...")
        print (f"- Model: {self.model_name}")
        print (f"- Context Lengths: {len(self.context_lengths)}, Min: {min(self.context_lengths)}, Max: {max(self.context_lengths)}")
        print (f"- Document Depths: {len(self.document_depth_percents)}, Min: {min(self.document_depth_percents)}%, Max: {max(self.document_depth_percents)}%")
        print (f"- Needle: {self.needle.strip()}")
        print ("\n\n")

    def start_test(self, args):
        if self.print_ongoing_status:
            self.print_start_test_summary()
        #asyncio.run(self.run_test())
        avg_score = self.run_test(args)
        return avg_score


if __name__ == "__main__":
    # Tons of defaults set, check out the LLMNeedleHaystackTester's init for more info
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--s_len', metavar='N', type=int, help='a number')
    parser.add_argument('-e', '--e_len', metavar='N', type=int, help='a number')
    parser.add_argument('--model_name', type=str, default=None, help='name of model')
    parser.add_argument("--attn_implementation", type=str,  default="flash_attention_2", choices=["flash_attention_2", "sdpa", "None"])
    parser.add_argument('--model_version', type=str, default=None, help='provider of model')
    parser.add_argument('--model_name_suffix', type=str, default=None, help='name of model')
    parser.add_argument('--model_provider', type=str, default="LLaMA", help='which model to use')
    parser.add_argument('--api_key', type=str, default="", help='OpenAI API Key')
    parser.add_argument('--step', type=int, default=1000)
    parser.add_argument('--sampling_number', type=int, default=0, help='a number')
    parser.add_argument('--mode', type=str, choices=['ada', 'fix', 'base','head_mask','cokv'], help="Ada mode, fix mode or normal")
    parser.add_argument('--compress_args_path', type=str, default=None, help="Path to the compress args")
    parser.add_argument('--gpu_idx', type=int, default=1, help='a number') 
    parser.add_argument('--set_size', type=int, default=1, help='a number')
    parser.add_argument("--skip",type=int, default=0, help="skip layer number")
    parser.add_argument('--gqa_support',action='store_true', default=False, help="init gqa_support")
    parser.add_argument('--floor_alpha',type=float,default=0.2,help="floor_alpha budgets for each head")
    parser.add_argument('--normalize',action='store_true')
    parser.add_argument('--pyram',action='store_true',help="using pyram mode")
    parser.add_argument('--pyram_beta',default=20,type=int, help="hyper parameter for pyram")
    args = parser.parse_args()

    mask_heads = torch.ones(32,32)
    ht = LLMNeedleHaystackTester(model_name=args.model_name, 
                            model_name_suffix=args.model_name_suffix,
                            model_provider=args.model_provider,
                            model_version=args.model_version, 
                            context_lengths_min=args.s_len,
                            save_contexts=True,
                            save_results=True,
                            openai_api_key=args.api_key, 
                            context_lengths_max=args.e_len, 
                            step=args.step, 
                            attn_implementation=args.attn_implementation,
                            mask_heads=mask_heads
                            )

    output_dir = f"{project_path}/experiments/Needle/complementary_contributions"
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, "cc_{}_new.npy".format(args.set_size,args.gpu_idx))
    

    n = 256 
    local_state = np.random.RandomState(None)
    cc_left = np.zeros(n+1)
    cc_right = np.zeros(n+1)
    count_left = np.zeros(n+1)
    count_right = np.zeros(n+1)
    idxs = np.arange(n)
    for _ in trange(args.sampling_number):
        local_state.shuffle(idxs)
        start_idx = args.set_size
        masked_size = torch.zeros((32,8))
        for i in range(start_idx):
            masked_size[int(idxs[i]/8)][idxs[i]%8] = 32000
        ht.change_given_adaptive_size(masked_size)
        u_left = ht.start_test(args)

        masked_size = torch.zeros((32,8))
        for i in range(start_idx,n):
            masked_size[int(idxs[i]/8)][idxs[i]%8] = 32000
        ht.change_given_adaptive_size(masked_size)
        u_right = ht.start_test(args)

        for i in range(n):
            if i < start_idx:
                cc_left[idxs[i]] += u_left-u_right
                count_left[idxs[i]] += 1
            else:
                cc_right[idxs[i]] += u_right-u_left
                count_right[idxs[i]] += 1

        
        if (_ + 1) % 5 == 0:
            data_to_save = {
                'cc_left': cc_left,
                'cc_right': cc_right,
                'count_left': count_left,
                'count_right': count_right,
                'sampling_count': _ + 1  
            }
            np.save(file_path, data_to_save)

             

