"""
This is a boilerplate pipeline 'fda_indications'
generated using Kedro 0.19.9
"""

from kedro.pipeline import Pipeline, pipeline, node

from . import nodes


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func = nodes.extract_fda_indications,
            inputs = "dailymed_labels",
            outputs = "fda_structured_indications_list",
            name = "extract-named-diseases-fda"
        ),

        node(
            func = nodes.build_list_fda,
            inputs = "fda_structured_indications_list",
            outputs = "fda_indications_list_with_ids",
            name = "add-identifiers-fda"
        ),

    ])
