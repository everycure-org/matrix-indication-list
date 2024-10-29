"""
This is a boilerplate pipeline 'ema_indications'
generated using Kedro 0.19.9
"""

from kedro.pipeline import Pipeline, pipeline, node

from . import nodes


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
                func = nodes.extract_ema_indications,
                inputs = "epar_table_4",
                outputs = "ema_structured_indications_list",
                name = "extract-named-diseases"
            ),
        node(
                func = nodes.build_list_ema,
                inputs = "ema_structured_indications_list",
                outputs = "ema_indications_list_with_ids",
                name = "add-identifiers-ema"
            ),

    ])
