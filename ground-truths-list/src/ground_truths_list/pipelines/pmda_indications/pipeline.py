"""
This is a boilerplate pipeline 'pmda_indications'
generated using Kedro 0.19.9
"""

from kedro.pipeline import Pipeline, pipeline, node

from . import nodes


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func = nodes.extract_pmda_indications,
            inputs = "pmda_approvals",
            outputs = "pmda_structured_indications_list",
            name = "extract-named-diseases-pmda"
        ),

        node(
            func = nodes.build_list_pmda,
            inputs = "pmda_structured_indications_list",
            outputs = "pmda_indications_list_with_ids",
            name = "add-identifiers-pmda"
        ),

    ])
