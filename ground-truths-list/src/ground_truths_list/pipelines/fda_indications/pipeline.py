from kedro.pipeline import Pipeline, pipeline, node
from . import nodes

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([

        # INDICATIONS
        node(
            func = nodes.mine_labels,
            inputs = "params:path_to_fda_labels",
            outputs = "dailymed_labels",
            name = "mine-dailymed-labels",
        ),


        # FDA INDICATIONS
        node(
            func = nodes.extract_fda_indications,
            inputs =[
                "dailymed_labels",
                "params:structured_list_prompt"
            ],
            outputs = "fda_structured_indications_list",
            name = "extract-named-diseases-fda"
        ),

        node(
            func = nodes.build_list_fda,
            inputs = "fda_structured_indications_list",
            outputs = "fda_indications_list_with_ids",
            name = "add-identifiers-fda"
        ),

        node(
            func= nodes.check_nameres_accuracy,
            inputs=[
                "fda_indications_list_with_ids",
                "params:id_correct_incorrect_tag",
            ],
            outputs="fda_test_list_llm_qc_1",
            name="check-nameres-accuracy",
        ),

        node(
            func=nodes.add_llm_selected_best_ids,
            inputs = [
                "fda_test_list_llm_qc_1",
                "params:llm_best_id_tag",
            ],
            outputs = "fda_test_list_llm_qc_2",
            name = "llm-choose-best-id",
        ),
        # node(
        #     func = nodes.enrich_list_llm_ids,
        #     inputs = [
        #         "fda_indications_list_with_ids",
        #         "params:id_tagging_params",
        #     ],
        #     outputs = "fda_indications_list_with_llm_ids",
        #     name = "add-llm-identifiers-fda",
        # ),
        node(
            func= nodes.add_normalized_llm_tag_ids,
            inputs=[
                "fda_test_list_llm_qc_2",
            ],
            outputs="fda_test_list_llm_ID",
            name="normalize-llm-ids",
        ),


    ])
