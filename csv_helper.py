import os
import glob
from collections import Counter

import torch
import fire
import numpy as np
import pandas as pd
from tqdm import tqdm
from loguru import logger

OPCODE = {
    "abs": 1,
    "add": 2,
    "add-dependency": 3,
    "after-all": 4,
    "all-reduce": 5,
    "all-to-all": 6,
    "atan2": 7,
    "batch-norm-grad": 8,
    "batch-norm-inference": 9,
    "batch-norm-training": 10,
    "bitcast": 11,
    "bitcast-convert": 12,
    "broadcast": 13,
    "call": 14,
    "ceil": 15,
    "cholesky": 16,
    "clamp": 17,
    "collective-permute": 18,
    "count-leading-zeros": 19,
    "compare": 20,
    "complex": 21,
    "concatenate": 22,
    "conditional": 23,
    "constant": 24,
    "convert": 25,
    "convolution": 26,
    "copy": 27,
    "copy-done": 28,
    "copy-start": 29,
    "cosine": 30,
    "custom-call": 31,
    "divide": 32,
    "domain": 33,
    "dot": 34,
    "dynamic-slice": 35,
    "dynamic-update-slice": 36,
    "exponential": 37,
    "exponential-minus-one": 38,
    "fft": 39,
    "floor": 40,
    "fusion": 41,
    "gather": 42,
    "get-dimension-size": 43,
    "set-dimension-size": 44,
    "get-tuple-element": 45,
    "imag": 46,
    "infeed": 47,
    "iota": 48,
    "is-finite": 49,
    "log": 50,
    "log-plus-one": 51,
    "and": 52,
    "not": 53,                       
    "or": 54,                        
    "xor": 55,                       
    "map": 56,                       
    "maximum": 57,                   
    "minimum": 58,                   
    "multiply": 59,                  
    "negate": 60,                    
    "outfeed": 61,                   
    "pad": 62,                       
    "parameter": 63,                 
    "partition-id": 64,              
    "popcnt": 65,                    
    "power": 66,                     
    "real": 67,                      
    "recv": 68,                      
    "recv-done": 69,                 
    "reduce": 70,                    
    "reduce-precision": 71,          
    "reduce-window": 72,             
    "remainder": 73,                 
    "replica-id": 74,                
    "reshape": 75,                   
    "reverse": 76,                   
    "rng": 77,                       
    "rng-get-and-update-state": 78,  
    "rng-bit-generator": 79,         
    "round-nearest-afz": 80,         
    "rsqrt": 81,                     
    "scatter": 82,                   
    "select": 83,                    
    "select-and-scatter": 84,        
    "send": 85,                      
    "send-done": 86,                 
    "shift-left": 87,                
    "shift-right-arithmetic": 88,    
    "shift-right-logical": 89,       
    "sign": 90,                      
    "sine": 91,                      
    "slice": 92,                     
    "sort": 93,                      
    "sqrt": 94,                      
    "subtract": 95,                  
    "tanh": 96,                      
    "transpose": 98,                 
    "triangular-solve": 99,          
    "tuple": 100,                    
    "while": 102,                    
    "cbrt": 103,                     
    "all-gather": 104,               
    "collective-permute-start": 105, 
    "collective-permute-done": 106,  
    "logistic": 107,                 
    "dynamic-reshape": 108,          
    "all-reduce-start": 109,         
    "all-reduce-done": 110,          
    "reduce-scatter": 111,           
    "all-gather-start": 112,         
    "all-gather-done": 113,          
    "opt-barrier": 114,              
    "async-start": 115,              
    "async-update": 116,             
    "async-done": 117,               
    "round-nearest-even": 118,       
    "stochastic-convert": 119,       
    "tan": 120,
}


@logger.catch
def draw_graph(graph_file):
    from graphviz import Digraph
    # ignores = ['parameter']
    ignores = []
    code2op = {v: k for k, v in OPCODE.items()}
    np_graph = np.load(graph_file)
    
    num_nodes = np_graph["node_feat"].shape[0]
    print('num_nodes: ', num_nodes)
    # breakpoint()
    graph = {
        "edges": np_graph['edge_index'].tolist(),
        "node_name": [''] * num_nodes
    }
    node_config_ids = set(np_graph['node_config_ids'].tolist())
    op_cnts = Counter()
    for i, op in enumerate(np_graph['node_opcode']):
        op_cnts[code2op[op]] += 1
        if code2op[op] in ignores:
            continue
        graph['node_name'][i] = f"[{i}] {code2op[op]}"
    print(op_cnts)
    # Create a Digraph object
    dot = Digraph(
        comment='Graph Visualization', 
        engine='dot', 
        graph_attr={
            'label': os.path.basename(graph_file),
            'splines': 'line',
            # 'nodesep': '0.8',
            'overlap': 'scale',
        },
    )

    # Add nodes and edges, setting the shape to "rectangle" and coloring node-2-a blue
    for i, node_name in enumerate(graph["node_name"]):
        if not node_name:
            continue
        if i in node_config_ids:
            dot.node(node_name, shape="rectangle", color="blue")
        else:
            dot.node(node_name, shape="rectangle")

    # print(graph["node_name"])
    for edge in graph["edges"]:
        from_node, to_node = edge
        a = graph["node_name"][from_node]
        b = graph["node_name"][to_node]
        if a and b:
            dot.edge(a, b)

    # Render the graph to a file (e.g., in PNG format)
    dot.render('graph', format='svg')


def strip_table(ckpt):
    checkpoint = torch.load(ckpt)
    # print(checkpoint.keys())
    # breakpoint()
    checkpoint['model_state'].pop('history.emb')
    output = ckpt.replace('.ckpt', '.strip.ckpt')
    torch.save(checkpoint, output)


def int8_config_feat(dir):
    files = glob.glob(os.path.join(dir, '*.pt'))
    for f in tqdm(files):
        if "_data" in f:
            data = torch.load(f)
            if -2**7 <= data.config_feats.min() and data.config_feats.min() <= 2**7:
                data.config_feats = data.config_feats.to(torch.int8)
                torch.save(data, f)


def overwrite(csv_a, csv_b, out_path):
    # Step 1: Read both CSV files into DataFrames
    a_df = pd.read_csv(csv_a)
    b_df = pd.read_csv(csv_b)

    # Step 2: Identify the rows to be overwritten (e.g., based on a common column)
    # For example, if you have a common column 'ID' to match rows:
    common_column = 'ID'
    rows_to_overwrite = a_df[common_column].isin(b_df[common_column])
    print(rows_to_overwrite)
    print(b_df)

    # Step 3: Overwrite the selected rows in a_df with corresponding rows from b_df
    a_df.loc[rows_to_overwrite] = b_df.values
    print(f'Overwrite {sum(rows_to_overwrite)} rows')
    print(a_df.loc[rows_to_overwrite])

    # Step 4: Save the modified a_df to 'a.csv'
    a_df.to_csv(out_path, index=False)


if __name__ == '__main__':
    fire.Fire({
        overwrite.__name__: overwrite,
        strip_table.__name__: strip_table,
        int8_config_feat.__name__: int8_config_feat,
        draw_graph.__name__: draw_graph,
    })