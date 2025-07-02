
import pandas as pd
from tqdm import tqdm

def add_unique_dd_edges_rk(rk_list:pd.DataFrame) -> pd.DataFrame:
    rk_edge_set = []
    
    for idx, row in tqdm(rk_list.iterrows(), total=len(rk_list), desc="extracting robokop unique d-d edges"):
        rk_edge_set.append("|".join([row['Treatment'], row['Disease']]))

    print(f"{len(rk_edge_set)} edges found in robokop")
    print(f"{len(set(rk_edge_set))} unique")
    rk_list['drug|disease']=rk_edge_set
    return rk_list



def compare_medi_robokop(medi_list: pd.DataFrame, robokop_list:pd.DataFrame) -> pd.DataFrame:
    rk_edge_set = set(list(robokop_list['drug|disease']))
    medi_edge_set = list(medi_list['drug|disease'])
    print(len(medi_edge_set))

    medi_original_edges = set(medi_edge_set)-set(rk_edge_set)
    print(f"{len((set(medi_edge_set)-set(rk_edge_set)))} edges in medi not found in rk")

    source_list = []
    target_list = []
    for i in medi_original_edges:
        item = i.split("|")
        source_list.append(item[0])
        target_list.append(item[1])
    
    return pd.DataFrame({
        "source":source_list,
        "target":target_list
    })
        
    