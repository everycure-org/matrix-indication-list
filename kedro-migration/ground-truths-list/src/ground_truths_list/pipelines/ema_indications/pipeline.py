"""
This is a boilerplate pipeline 'ema_indications'
generated using Kedro 0.19.9
"""
from kedro.pipeline import Pipeline, node, pipeline

from . import nodes

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            # node(
            #     func=ingest_label_indication_sections_fda,
            #     inputs="params:labels_location",
            #     outputs="indications_fda",
            # ),
            node(
                func = nodes.indications_to_structured_disease_lists_fda,
                inputs = "epar_table_4",
                outputs = "structured_indications_lists_ema",
                name = "indications_to_structured_lists_ema"
            ),
            node(
                func=nodes.structured_disease_lists_to_edges_with_IDs_fda,
                inputs="structured_indications_lists_ema",
                outputs="indications_with_ids_ema",
                name = "lists_to_ids_ema"
            ),
            # node(
            #     func=nodes.downfill_mondo,
            #     inputs="indications_with_ids_fda",
            #     outputs="downfilled_mondo_list_fda",
            # ),
        ]
    )