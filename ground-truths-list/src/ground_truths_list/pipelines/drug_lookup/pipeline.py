"""
This is a boilerplate pipeline 'drug_lookup'
generated using Kedro 0.19.9
"""

from kedro.pipeline import Pipeline, pipeline, node
from . import nodes


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func = nodes.merge_lists,
            inputs = [
                "fda_indications_list_with_ids",
                "ema_indications_list_with_ids",
                "pmda_indications_list_with_ids",
            ],
            outputs = "indication_list_merged",
            name = "merge-lists"
        ),
    ])
