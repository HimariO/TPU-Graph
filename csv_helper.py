import os
import glob
from pprint import pprint
from collections import Counter, defaultdict
from itertools import combinations, permutations
from typing import *

import fire
import ranky
import torch
import scipy
import numpy as np
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
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
    ignores = ['parameter', 'get-tuple-element', 'tuple', 'constant']
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
        graph['node_name'][i] = f"{code2op[op]}"
        # graph['node_name'][i] = f"[{i}] {code2op[op]}"

    # Create a directed graph
    G = nx.DiGraph()

    # Add nodes and edges from your graph data
    for node_name in graph["node_name"]:
        G.add_node(node_name)

    parent_types = defaultdict(Counter)
    chd_types = defaultdict(Counter)
    for edge in graph["edges"]:
        from_node, to_node = edge
        G.add_edge(graph["node_name"][from_node], graph["node_name"][to_node])
        parent_types[graph["node_name"][to_node]][graph["node_name"][from_node]] += 1
        chd_types[graph["node_name"][from_node]][graph["node_name"][to_node]] += 1
    
    logger.info('op-parents')
    pprint(dict(parent_types))
    logger.info('op-childrens')
    pprint(dict(chd_types))

    # Create a layout for the nodes
    pos = nx.spring_layout(G)

    # Draw the graph
    nx.draw(G, pos, with_labels=True, node_size=5000, node_color="skyblue", font_size=10, font_color="black", font_weight="bold")

    # Display node names
    labels = {node_name: node_name for node_name in graph["node_name"]}
    nx.draw_networkx_labels(G, pos, labels, font_size=8, font_color="black", font_weight="bold")

    # # Show the graph
    # plt.axis("off")
    # plt.show()
    # nx.write_gexf(G, graph_file.replace(".npz", ".gexf"))


@logger.catch
def _draw_graph(graph_file):
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
    files = glob.glob(os.path.join(dir, 'xla_tile*.pt'))
    int16 = 0
    int8 = 0
    for f in tqdm(files):
        if "_data" in f:
            data = torch.load(f)
            if -2**7 <= data.config_feats.min() and data.config_feats.max() <= 2**7:
                data.config_feats = data.config_feats.to(torch.int8)
                torch.save(data, f)
                int8 += 1
            elif -2**15 <= data.config_feats.min() and data.config_feats.max() <= 2**15:
                data.config_feats = data.config_feats.to(torch.int16)
                torch.save(data, f)
                int16 += 1
    print(int8, int16)


def overwrite(csv_a, csv_b, out_path):
    # Step 1: Read both CSV files into DataFrames
    a_df = pd.read_csv(csv_a)
    b_df = pd.read_csv(csv_b)

    # Step 2: Identify the rows to be overwritten (e.g., based on a common column)
    # For example, if you have a common column 'ID' to match rows:
    common_column = 'ID'
    rows_to_overwrite = a_df[common_column].isin(b_df[common_column])
    print(rows_to_overwrite)
    print(a_df.loc[rows_to_overwrite])
    print(b_df)

    # Step 3: Overwrite the selected rows in a_df with corresponding rows from b_df
    a_df.loc[rows_to_overwrite] = b_df.values
    print(f'Overwrite {sum(rows_to_overwrite)} rows')
    print(a_df.loc[rows_to_overwrite])

    # Step 4: Save the modified a_df to 'a.csv'
    a_df.to_csv(out_path, index=False)


def rankaggr_lp(ranks):

    def _build_graph(ranks: np.ndarray):
        n_voters, n_candidates = ranks.shape
        edge_weights = np.zeros((n_candidates, n_candidates))
        for i, j in combinations(range(n_candidates), 2):
            preference = ranks[:, i] - ranks[:, j]
            h_ij = np.sum(preference < 0)  # prefers i to j
            h_ji = np.sum(preference > 0)  # prefers j to i
            if h_ij > h_ji:
                edge_weights[i, j] = h_ij - h_ji
            elif h_ij < h_ji:
                edge_weights[j, i] = h_ji - h_ij
        return edge_weights

    """Kemeny-Young optimal rank aggregation"""

    n_voters, n_candidates = ranks.shape
    
    # maximize c.T * x
    edge_weights = _build_graph(ranks)
    c = -1 * edge_weights.ravel()  

    idx = lambda i, j: n_candidates * i + j

    # constraints for every pair
    pairwise_constraints = np.zeros([
        (n_candidates * (n_candidates - 1)) // 2,
        n_candidates ** 2
    ])
    for row, (i, j) in zip(pairwise_constraints,
                           combinations(range(n_candidates), 2)):
        row[[idx(i, j), idx(j, i)]] = 1

    # and for every cycle of length 3
    triangle_constraints = np.zeros([
        (n_candidates * (n_candidates - 1) * (n_candidates - 2)),
        n_candidates ** 2
    ])
    for row, (i, j, k) in zip(triangle_constraints,
                              permutations(range(n_candidates), 3)):
        row[[idx(i, j), idx(j, k), idx(k, i)]] = 1

    constraints = np.vstack([pairwise_constraints, triangle_constraints])
    constraint_rhs = np.hstack([np.ones(len(pairwise_constraints)),
                                np.ones(len(triangle_constraints))])
    constraint_signs = np.hstack([np.zeros(len(pairwise_constraints)),  # ==
                                  np.ones(len(triangle_constraints))])  # >=

    # obj, x, duals = lp_solve(c, constraints, constraint_rhs, constraint_signs,
    #                          xint=range(1, 1 + n_candidates ** 2))
    res = scipy.optimize.linprog(
        c, 
        A_ub=triangle_constraints, 
        b_ub=np.ones(len(triangle_constraints)),
        A_eq=pairwise_constraints,
        b_eq=np.ones(len(pairwise_constraints)),
    )
    breakpoint()

    # x = np.array(x).reshape((n_candidates, n_candidates))
    # aggr_rank = x.sum(axis=1)

    # return obj, aggr_rank


def ensemble_csv(csvs: List[str], out_path: str):
    print('Ensemble CSVs: ', csvs)

    def merge(rank_strs: List[str]) -> str:
        print('merging: ', rank_strs)
        score_mtx = []
        for judge, s in enumerate(rank_strs):
            config_ids = [int(v) for v in s.split(';')]
            score_mtx.append([0] * len(config_ids))
            for rank, id in enumerate(config_ids):
                score_mtx[judge][id] = float(rank)

            # print(score_mtx[-1])
        score_mtx = np.asarray(score_mtx)
        # merge_score = ranky.pairwise(score_mtx.T)
        # merge_score = ranky.kemeny_young(score_mtx.T, workers=16)
        rankaggr_lp(score_mtx)
        merge_score = score_mtx.mean(axis=0)
        new_rank = sorted([(score, i) for i, score in enumerate(merge_score)])
        return ';'.join(str(r[1]) for r in new_rank)
    

    dfs = [
        pd.read_csv(file)
        for file in csvs
    ]

    all_id = [set(df['ID']) for df in dfs]
    all_id = all_id[0].union(*all_id[1:])

    updated_ranks = {}
    for row_id in tqdm(all_id):
        predicts = []
        for df in dfs:
            row = df[df['ID'] == row_id]
            rank_str = row['TopConfigs'].values[0]
            predicts.append(rank_str)
        if len(predicts) > 1:
            updated_ranks[row_id] = merge(predicts)
        else:
            updated_ranks[row_id] = predicts[0]

    result = {'ID': [], 'TopConfigs': []}
    for k, v in updated_ranks.items():
        result['ID'].append(k)
        result['TopConfigs'].append(v)
    
    pd.DataFrame.from_dict(result).to_csv(out_path, index=False)


def ensemble_raw(pt_files: List[str], out_path: str):
    id2scores = defaultdict(list)
    for file in pt_files:
        ranks: Dict[str, List] = torch.load(file)
        for graph_id, predicts in ranks.items():
            score_vec = [0] * len(predicts)
            for runtime, item_id in predicts:
                score_vec[item_id] = runtime
            id2scores[graph_id].append(score_vec)
    
    def merge(score_mtx: List[List[float]]) -> str:
        score_mtx = np.asarray(score_mtx)
        score_mtx -= np.min(score_mtx, axis=1, keepdims=True)
        score_mtx /= np.max(score_mtx, axis=1, keepdims=True)
        # merge_score = ranky.pairwise(score_mtx.T)
        # merge_score = ranky.kemeny_young(score_mtx.T, workers=16)
        merge_score = score_mtx.mean(axis=0)
        print('-' * 100)
        print(score_mtx)
        print(merge_score)
        new_rank = sorted([(score, i) for i, score in enumerate(merge_score)])
        return ';'.join(str(r[1]) for r in new_rank)
    
    updated_ranks = {}
    for graph_id, score_mtx in tqdm(id2scores.items()):
        if len(score_mtx) == 1:
            raise ValueError('nothing to merge?')
        updated_ranks[graph_id] = merge(score_mtx)
    
    result = {'ID': [], 'TopConfigs': []}
    for k, v in updated_ranks.items():
        result['ID'].append(k)
        result['TopConfigs'].append(v)
    pd.DataFrame.from_dict(result).to_csv(out_path, index=False)
        

def inpsect_dataset(root_dir, field='runtime'):
    from graphgps.loader.dataset.tpu_graphs import TPUGraphsNpz
    dataset = TPUGraphsNpz(root_dir, source='nlp', search='default', task='layout')
    dataset._norm_op_feat = False
    config_feats = []
    code2name = {v: k for k, v in OPCODE.items()}
    
    for i in tqdm(range(len(dataset))):
        graph = dataset[i]
        logger.info(f'Graph: {graph.graph_name}')
        if (graph.y < 1e-6).all():
            logger.warning(f"Skip test-set data that don't have runtime labels.")
            continue
        
        if field == 'runtime':
            panels = 4
            plt.subplot(panels, 1, 1)
            plt.title(graph.graph_name)
            plt.hist(graph.y, bins=100)
            
            m = graph.y.size(0)
            ordered = graph.y.sort().values
            # breakpoint()
            plt.subplot(panels, 1, 2)
            plt.title(f"{graph.graph_name} first half")
            plt.hist(ordered[:m // 2], bins=100)
            plt.subplot(panels, 1, 3)
            plt.title(f"{graph.graph_name} second half")
            plt.hist(ordered[m // 2:], bins=100)
            
            plt.subplot(panels, 1, 4)
            plt.title(f"{graph.graph_name} first querter")
            plt.hist(ordered[:m // 4], bins=100)
            
            plt.show()
        elif field == 'config':

            def color_map(layout_ind):
                colors = torch.tensor([
                    [227, 51, 39],   # #e32727
                    [227, 167, 39],  # #e3a727
                    [136, 227, 39],  # #88e327
                    [39, 227, 193],  # #27e3c1
                    [39, 70, 227],   # #273de3
                    [174, 39, 227],  # #ae27e3
                    [0, 0, 0],
                ], dtype=torch.uint8)
                return colors[layout_ind]
            
            num_config = graph.num_config
            config_nodes = graph.num_config_idx
            config_feats = graph.config_feats.view(num_config, config_nodes, -1)
            
            m = min(10, num_config)
            for j, single_config in enumerate(config_feats[20: 20+m]):
                plt.subplot(m, 1, j + 1)
                plt.imshow(color_map(single_config.int().T))
            # plt.tight_layout()
            plt.show()
        elif field == 'dup_config':
            num_config = graph.num_config
            config_nodes = graph.num_config_idx
            config_feats = graph.config_feats.view(num_config, config_nodes, -1)
            config_feats_2d = config_feats.view(num_config, -1).int().cuda()

            handled = set()
            dup_from = defaultdict(lambda: 0)
            breakpoint()
            for r, row in enumerate(config_feats_2d):
                if r in handled:
                    continue
                handled.add(r)
                delta = abs(config_feats_2d - row).sum(dim=1)
                result = torch.sort(delta)
                for i, v in zip(result.indices.cpu(), result.values.cpu()):
                    if v > 0:
                        break
                    if i != r:
                        handled.add(int(i))
                        dup_from[r] += 1
                pprint(dict(dup_from))
        elif field == 'op_feat':
            op2feat = defaultdict(list)
            configurables = graph.config_idx.int().tolist()
            op_cnter = Counter()
            prev = defaultdict(lambda: None)

            num_config = graph.num_config
            config_nodes = graph.num_config_idx
            config_feats = graph.config_feats.view(num_config, config_nodes, -1)

            for i, (opcode, feat) in enumerate(zip(graph.op_code.int(), graph.op_feats)):
                op_name = code2name[int(opcode)]
                op_cnter[op_name] += 1
                if i not in configurables:
                    continue
                stuff = feat.int()
                cfeat = config_feats[:10, configurables.index(i)]
                if len(op2feat[op_name]) < 4 and (prev[op_name] is None or (stuff != prev[op_name]).any()):
                    # if prev[op_name] is not None and op_name == 'reshape':
                    #     print(stuff != prev[op_name])
                    prev[op_name] = stuff
                    stuff = {
                        'shape_dimensions': stuff[21:29],
                        'dimensions': stuff[31:37],
                        'window_size': stuff[37:43],
                        'convolution_dim_numbers': stuff[93:107],
                        'slice_dims': stuff[109:121],
                        'layout_minor_to_major': stuff[134:],
                        # 'all': stuff,
                        'config': cfeat.int(),
                    }
                    """
                    NOTE: 
                    The layout configuration of a convolution/dot node specifies only the "first two dimensions" 
                    because of the 2D nature of registers on a TPU. 
                    The other dimensions are essentially outer loops that do not matter to the speed of the program.
                    """
                    stuff = {k: v.tolist() if k != 'all' else v for k, v in stuff.items()}
                    op2feat[op_name].append(stuff)
            # op2feat = {k: torch.stack(v) for k, v in op2feat.items()}
            # pprint(op_cnter)
            pprint(dict(op2feat), width=100)
            print(f"configurabels: {config_nodes}/{graph.op_code.size(0)}")
            input()


if __name__ == '__main__':
    fire.Fire({
        overwrite.__name__: overwrite,
        strip_table.__name__: strip_table,
        int8_config_feat.__name__: int8_config_feat,
        draw_graph.__name__: draw_graph,
        ensemble_csv.__name__: ensemble_csv,
        ensemble_raw.__name__: ensemble_raw,
        inpsect_dataset.__name__: inpsect_dataset,
    })