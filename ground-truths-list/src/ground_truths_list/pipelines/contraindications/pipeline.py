"""
This is a boilerplate pipeline 'contraindications'
generated using Kedro 0.19.9
"""

from kedro.pipeline import Pipeline, pipeline, node
from . import nodes


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
                func = nodes.extract_structured_lists_contraindications_dailymed,
                inputs = "dailymed_label_contraindications",
                outputs = "structured_contraindications_lists_fda",
                name = "contraindications-to-structured-lists-fda"
            ),
        node(
                func = nodes.generate_contraindications_list,
                inputs = "structured_contraindications_lists_fda",
                outputs = "contraindications_list_with_ids_fda",
                name = "build_fda_contraindications_list"
            ),
         node(
                func = nodes.merge_contraindications_and_indications,
                inputs = [
                    "contraindications_list_with_ids_fda",
                    "indication_list_downfilled",

                ],
                outputs = "ground_truths_list",
                name = "merge-contraindications-and-indications"
            ),    

    ])
