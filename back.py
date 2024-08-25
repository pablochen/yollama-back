import subprocess
import pty
import os
import sys
import time
import numpy as np
import pandas as pd

from fastapi import FastAPI
from fastapi.responses import StreamingResponse

import torch
from FlagEmbedding import BGEM3FlagModel
from pymilvus import connections, Collection, FieldSchema, DataType, CollectionSchema, utility

app = FastAPI()

embed_model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)

connections.connect(
    alias="default",
    host="milvus",
    port="19530"
)

collection_name = "test_insurance_003"
collection = Collection(name=collection_name)
collection.load()

alpaca_prompt = """
### Instruction:
{}

### Context:
{}

### Input:
{}

### Response:
{}
"""

instruction = '''
다음은 작업을 설명하는 지시사항과 추가적인 맥락을 제공하는 입력이 쌍을 이루는 형태입니다.
단계별로 차근 차근 생각해서 결론 위주로 자세히 대답해줘.
한줄에 띄어쓰기 포함해서 60글자 이내로 써주고 그 이상일 때는, 줄바꿈을 해줘.
'''


script_path = os.path.abspath(__file__)

def vectorize(text):
    with torch.no_grad():
        encoded = embed_model.encode(text, batch_size=60, max_length=8192)
    return encoded['dense_vecs']

def get_answer_from_vectordb(text, search_len):
    search_vectors = vectorize(text)
    output_fields = ["file_name", "div", "kwan", "jo", "text"]
    search_params = {"metric_type": "COSINE"} 
    results = collection.search([search_vectors], 
        "embedding", 
        search_params, 
        output_fields=output_fields, 
        limit=search_len)[0]


    context = ''
    for res in results:
        context += res.entity.div + " : " + res.entity.kwan + " : " + res.entity.jo + " : " + res.entity.text + '\n\n'
        
    context = context.replace(' \n', '')
    print(content)
    
    return context
    
def runner(model_path, token_len, thread_len, prompt):
    start_time = time.time()
    
    command = [
        '/data/runner',
        '--model', model_path,
        '--prompt', prompt,
        '--n-predict', token_len,
        '--threads', thread_len,
    ]

    master_fd, slave_fd = pty.openpty()
    process = subprocess.Popen(command, 
                               stdin=slave_fd, 
                               stdout=slave_fd, 
                               stderr=subprocess.PIPE, 
                               text=False,
                               bufsize=0)
    os.close(slave_fd)

    start_marker = '### Response:'
    output_text = ""
    response_started = False

    while True:
        time.sleep(0.01)
        try:
            output = os.read(master_fd, 1024)
            try:
                output = output.decode('utf-8')
            except UnicodeDecodeError:
                output = ""
        
            if output == "" and process.poll() is not None:
                break
                            
            if output:
                if not response_started:
                    if start_marker in output_text:
                        response_started = True
                        output_text = output_text.split(start_marker, 1)[1]
                    else:
                        output_text += output
                
                if response_started:
                    if '###' in output_text:
                        break
                    else:
                        output_text += output
                        yield output

        except OSError as e:
            print(f"OSError: {e}")
            break

    end_time = time.time()
    elapsed_time = end_time - start_time
    yield f"<br>Execution time: {elapsed_time:.2f} seconds"

    os.close(master_fd)
    process.wait()

@app.get("/run_model")
def run_model(input_text: str, token_len: int, thread_len:int, search_len:int):
    model_option = '/data/model.gguf'
    context = ''
    prompt = alpaca_prompt.format(instruction, context, input_text, "")
    return StreamingResponse(
        runner(model_option, str(token_len), str(thread_len), prompt), 
        media_type="text/plain"
    )

def main():
    input_text = "심장 관련된 내용 알려줘"
    search_len = 5
    context = get_answer_from_vectordb(input_text, search_len)
    
    model_option = '/root/model/model.gguf'
    token_len = 256
    thread_len = 8
    prompt = alpaca_prompt.format(instruction, context, input_text, "")
    
    runner(model_option, str(token_len), str(thread_len), prompt)

if __name__ == "__main__":
    main()