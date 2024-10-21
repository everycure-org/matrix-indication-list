from kedro.pipeline import Pipeline, node, pipeline

from . import nodes

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=ingest_label_indication_sections_fda,
                inputs="params:labels_location",
                outputs="indications_fda",
            ),
            node(
                func=indications_to_structured_disease_lists_fda,
                inputs="indications_fda",
                outputs="structured_indications_lists_fda",
            ),
            node(
                func=structured_disease_lists_to_edges_with_IDs_fda,
                inputs="structured_indications_lists_fda",
                outputs="indications_with_ids_fda",
            ),
            node(
                func=downfill_mondo,
                inputs="indications_with_ids_fda",
                outputs="downfilled_mondo_list_fda",
            ),
        ]
    )
