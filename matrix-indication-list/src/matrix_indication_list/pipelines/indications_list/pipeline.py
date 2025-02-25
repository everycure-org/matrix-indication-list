from kedro.pipeline import Pipeline, pipeline, node
from . import nodes


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([


#########################################
### FDA CONTRAINDICATIONS ###############
#########################################
        node(
            func=nodes.mine_contraindications,
            inputs = [
               "params:path_to_fda_labels",
            ],
            outputs = "dailymed_contraindications_raw",
            name = "mine-contraindications-fda",
        ),
        node(
            func=nodes.extract_named_diseases,
            inputs = [
                "dailymed_contraindications_raw",
                "params:column_names.contraindications_active_ingredients",
                "params:column_names.contraindications_text_column",
                "params:column_names.contraindications_structured_list_column",
                "params:contraindications_structured_list_prompt",
            ],
            outputs = "dailymed_contraindications_1",
            name = "extract-contraindications-lists-fda",
        ),
        node(
            func=nodes.flatten_list,
            inputs = [
                "dailymed_contraindications_1",
                "params:column_names.contraindications_structured_list_column",
                "params:column_names.contraindications_active_ingredients",
                "params:column_names.contraindications_text_column",
                "params:column_names.new_contraindications_disease_name_column",
            ],
            outputs = "dailymed_contraindications_2",
            name = "flatten-contraindications-list-fda"
        ),

        node(
            func=nodes.clean_list,
            inputs = [
                "dailymed_contraindications_2",
                "params:column_names.new_contraindications_disease_name_column",
                "params:strings_to_clean_from_disease_list",
                "params:cleaning_regex_sub_pattern"
            ],
            outputs = "dailymed_contraindications_3",
            name = "clean-contraindications-list-fda"
        ),
        
        node(
            func=nodes.clean_list,
            inputs = [
                "dailymed_contraindications_3",
                "params:column_names.contraindications_active_ingredients",
                "params:strings_to_clean_from_disease_list",
                "params:cleaning_regex_sub_pattern"
            ],
            outputs = "dailymed_contraindications_4",
            name = "clean-contraindications-active-ingredients-fda"
        ),
        
        node(
            func=nodes.resolve_concepts,
            inputs = [
                "dailymed_contraindications_4",
                "params:column_names.new_contraindications_disease_name_column",
                "params:column_names.disease_id_column",
                "params:column_names.disease_label_column",
                "params:biolink_type_disease"
            ],
            outputs = "dailymed_contraindications_5",
            name = "nameres-fda-contraindications-diseases"
        ),

         node(
            func=nodes.check_nameres_llm,
            inputs = [
                "dailymed_contraindications_5",
                "params:column_names.new_contraindications_disease_name_column",
                "params:column_names.disease_label_column",
                "params:id_correct_incorrect_tag_disease",
                "params:column_names.llm_true_false_column_disease"
            ],
            outputs = "dailymed_contraindications_6",
            name = "nameres-auto-qc-contraindications-disease"
        ),
        node(
            func=nodes.llm_improve_ids,
            inputs = [
                "dailymed_contraindications_6",
                "params:column_names.new_contraindications_disease_name_column",
                "params:llm_best_id_tag",
                "params:biolink_type_disease",
                "params:column_names.disease_id_column",
                "params:column_names.llm_true_false_column_disease",
                "params:column_names.llm_improved_id_column",
            ],
            outputs = "dailymed_contraindications_7",
            name = "llm-id-improvement-contraindications-diseases"
        ),      
        node(
            func=nodes.resolve_concepts,
            inputs = [
                "dailymed_contraindications_7",
                "params:column_names.contraindications_active_ingredients",
                "params:column_names.drug_id_column",
                "params:column_names.drug_label_column",
                "params:biolink_type_drug"
            ],
            outputs = "dailymed_contraindications_8",
            name = "nameres-fda-contraindications-drugs",
        ),
        node(
            func=nodes.add_normalized_llm_tag_ids,
            inputs= [
                "dailymed_contraindications_8",
                "params:column_names.llm_improved_id_column",
                "params:column_names.llm_normalized_id_column_disease",
                "params:column_names.llm_normalized_label_column_disease",
            ],
            outputs="dailymed_contraindications_9",
            name = "normalize-diseases-fda-contraindications"
        ),
        node(
            func=nodes.check_nameres_llm,
            inputs = [
                "dailymed_contraindications_9",
                "params:column_names.contraindications_active_ingredients",
                "params:column_names.drug_label_column",
                "params:id_correct_incorrect_tag_drug",
                "params:column_names.llm_true_false_column_drug"
            ],
            outputs = "dailymed_contraindications_10",
            name = "nameres-auto-qc-drug-fda-contraindications"
        ),

        node(
            func=nodes.llm_improve_ids,
            inputs = [
                "dailymed_contraindications_10",
                "params:column_names.contraindications_active_ingredients",
                "params:llm_best_id_tag_drug",
                "params:biolink_type_drug",
                "params:column_names.drug_id_column",
                "params:column_names.llm_true_false_column_drug",
                "params:column_names.llm_improved_id_column_drug",
            ],
            outputs = "dailymed_contraindications_11",
            name = "llm-id-improvement-drug-fda-contraindications"
        ),

        node(
            func=nodes.add_normalized_llm_tag_ids,
            inputs= [
                "dailymed_contraindications_11",
                "params:column_names.llm_improved_id_column_drug",
                "params:column_names.llm_normalized_id_column_drug",
                "params:column_names.llm_normalized_label_column_drug",
            ],
            outputs="dailymed_contraindications_12",
            name = "normalize-drug-contraindications-fda",
        ),

        node(
            func=nodes.deduplicate_entities,
            inputs=[
                "dailymed_contraindications_12",
                "params:column_names.llm_normalized_id_column_drug",
                "params:column_names.llm_normalized_id_column_disease",
                "params:column_names.deduplication_column",
            ],
            outputs="dailymed_contraindications_13",
            name="deduplicate-fda-contraindications"
        ),

        node(
            func=nodes.downfill_list_mondo,
            inputs=[
                "dailymed_contraindications_13",
                "mondo_edges",
                "mondo_nodes",
                "params:column_names",
            ],
            outputs="matrix_contraindications_list_downfilled",
            name="downfill-list-contraindications"
        ),



#########################################
######## FDA LIST #######################
#########################################
        node(
            func=nodes.extract_named_diseases,
            inputs = [
                "dailymed_raw",
                "params:column_names.drug_name_column",
                "params:column_names.indications_text_column",
                "params:column_names.disease_list_column",
                "params:structured_list_prompt",
            ],
            outputs = "dailymed_1",
            name = "extract-disease-lists-fda"
        ),
        node(
            func=nodes.flatten_list,
            inputs = [
                "dailymed_1",
                "params:column_names.disease_list_column",
                "params:column_names.drug_name_column",
                "params:column_names.indications_text_column",
                "params:column_names.disease_name_column",
            ],
            outputs = "dailymed_2",
            name = "flatten-list-fda"
        ),
        
        node(
            func=nodes.clean_list,
            inputs = [
                "dailymed_2",
                "params:column_names.disease_name_column",
                "params:strings_to_clean_from_disease_list",
                "params:cleaning_regex_sub_pattern"
            ],
            outputs = "dailymed_3",
            name = "clean-list-fda"
        ),

        node(
            func=nodes.resolve_concepts,
            inputs = [
                "dailymed_3",
                "params:column_names.disease_name_column",
                "params:column_names.disease_id_column",
                "params:column_names.disease_label_column",
                "params:biolink_type_disease"
            ],
            outputs = "dailymed_4",
            name = "nameres-fda-diseases"
        ),

        node(
            func=nodes.check_nameres_llm,
            inputs = [
                "dailymed_4",
                "params:column_names.disease_name_column",
                "params:column_names.disease_label_column",
                "params:id_correct_incorrect_tag_disease",
                "params:column_names.llm_true_false_column_disease"
            ],
            outputs = "dailymed_5",
            name = "nameres-auto-qc"
        ),

        node(
            func=nodes.llm_improve_ids,
            inputs = [
                "dailymed_5",
                "params:column_names.disease_name_column",
                "params:llm_best_id_tag",
                "params:biolink_type_disease",
                "params:column_names.disease_id_column",
                "params:column_names.llm_true_false_column_disease",
                "params:column_names.llm_improved_id_column",
            ],
            outputs = "dailymed_6",
            name = "llm-id-improvement"
        ),
        
        node(
            func=nodes.resolve_concepts,
            inputs = [
                "dailymed_6",
                "params:column_names.drug_name_column",
                "params:column_names.drug_id_column",
                "params:column_names.drug_label_column",
                "params:biolink_type_drug"
            ],
            outputs = "dailymed_7",
            name = "nameres-fda-drugs",
        ),

        node(
            func=nodes.add_normalized_llm_tag_ids,
            inputs= [
                "dailymed_7",
                "params:column_names.llm_improved_id_column",
                "params:column_names.llm_normalized_id_column_disease",
                "params:column_names.llm_normalized_label_column_disease",
            ],
            outputs="dailymed_8",
            name = "normalize-diseases-fda"
        ),

        node(
            func=nodes.check_nameres_llm,
            inputs = [
                "dailymed_8",
                "params:column_names.drug_name_column",
                "params:column_names.drug_label_column",
                "params:id_correct_incorrect_tag_drug",
                "params:column_names.llm_true_false_column_drug"
            ],
            outputs = "dailymed_9",
            name = "nameres-auto-qc-drug-fda"
        ),

        node(
            func=nodes.llm_improve_ids,
            inputs = [
                "dailymed_9",
                "params:column_names.drug_name_column",
                "params:llm_best_id_tag_drug",
                "params:biolink_type_drug",
                "params:column_names.drug_id_column",
                "params:column_names.llm_true_false_column_drug",
                "params:column_names.llm_improved_id_column_drug",
            ],
            outputs = "dailymed_10",
            name = "llm-id-improvement-drug-fda"
        ),

        node(
            func=nodes.add_normalized_llm_tag_ids,
            inputs= [
                "dailymed_10",
                "params:column_names.llm_improved_id_column_drug",
                "params:column_names.llm_normalized_id_column_drug",
                "params:column_names.llm_normalized_label_column_drug",
            ],
            outputs="dailymed_11",
            name = "normalize-drugs-fda",
        ),

        node(
            func=nodes.deduplicate_entities,
            inputs=[
                "dailymed_11",
                "params:column_names.llm_improved_id_column_drug",
                "params:column_names.llm_improved_id_column",
                "params:column_names.deduplication_column",
            ],
            outputs="dailymed_12",
            name="deduplicate-fda"
        ),



#########################################
######## EMA LIST #######################
#########################################


        node(
            func=nodes.standardize_ema_rows,
            inputs=[
                "epar_table_4",
                "params:column_names",

            ],
            outputs="ema_preprocessed",
            name="standardize-ema"

        ),

        node(
            func=nodes.extract_named_diseases,
            inputs = [
                "ema_preprocessed",
                "params:column_names.drug_name_column",
                "params:column_names.indications_text_column",
                "params:column_names.disease_list_column",
                "params:structured_list_prompt",
            ],
            outputs = "ema_1",
            name = "extract-disease-lists-ema"
        ),

        node(
            func=nodes.flatten_list,
            inputs = [
                "ema_1",
                "params:column_names.disease_list_column",
                "params:column_names.drug_name_column",
                "params:column_names.indications_text_column",
                "params:column_names.disease_name_column",
            ],
            outputs = "ema_2",
            name = "flatten-list-ema"
        ),

        node(
            func=nodes.clean_list,
            inputs = [
                "ema_2",
                "params:column_names.disease_name_column",
                "params:strings_to_clean_from_disease_list",
                "params:cleaning_regex_sub_pattern"
            ],
            outputs = "ema_3",
            name = "clean-list-ema"
        ),

        node(
            func=nodes.resolve_concepts,
            inputs = [
                "ema_3",
                "params:column_names.disease_name_column",
                "params:column_names.disease_id_column",
                "params:column_names.disease_label_column",
                "params:biolink_type_disease"
            ],
            outputs = "ema_4",
            name = "nameres-ema-diseases"
        ),

        node(
            func=nodes.check_nameres_llm,
            inputs = [
                "ema_4",
                "params:column_names.disease_name_column",
                "params:column_names.disease_label_column",
                "params:id_correct_incorrect_tag_disease",
                "params:column_names.llm_true_false_column_disease"
            ],
            outputs = "ema_5",
            name = "nameres-auto-qc-ema"
        ),

        node(
            func=nodes.llm_improve_ids,
            inputs = [
                "ema_5",
                "params:column_names.disease_name_column",
                "params:llm_best_id_tag",
                "params:biolink_type_disease",
                "params:column_names.disease_id_column",
                "params:column_names.llm_true_false_column_disease",
                "params:column_names.llm_improved_id_column",
            ],
            outputs = "ema_6",
            name = "llm-id-improvement-ema"
        ),
        
        node(
            func=nodes.resolve_concepts,
            inputs = [
                "ema_6",
                "params:column_names.drug_name_column",
                "params:column_names.drug_id_column",
                "params:column_names.drug_label_column",
                "params:biolink_type_drug"
            ],
            outputs = "ema_7",
            name = "nameres-ema-drugs",
        ),

        node(
            func=nodes.add_normalized_llm_tag_ids,
            inputs= [
                "ema_7",
                "params:column_names.llm_improved_id_column",
                "params:column_names.llm_normalized_id_column_disease",
                "params:column_names.llm_normalized_label_column_disease",
            ],
            outputs="ema_8",
            name = "normalize-diseases-ema"
        ),

        node(
            func=nodes.check_nameres_llm,
            inputs = [
                "ema_8",
                "params:column_names.drug_name_column",
                "params:column_names.drug_label_column",
                "params:id_correct_incorrect_tag_drug",
                "params:column_names.llm_true_false_column_drug"
            ],
            outputs = "ema_9",
            name = "nameres-auto-qc-drug-ema"
        ),

        node(
            func=nodes.llm_improve_ids,
            inputs = [
                "ema_9",
                "params:column_names.drug_name_column",
                "params:llm_best_id_tag_drug",
                "params:biolink_type_drug",
                "params:column_names.drug_id_column",
                "params:column_names.llm_true_false_column_drug",
                "params:column_names.llm_improved_id_column_drug",
            ],
            outputs = "ema_10",
            name = "llm-id-improvement-drug-ema"
        ),

        node(
            func=nodes.add_normalized_llm_tag_ids,
            inputs= [
                "ema_10",
                "params:column_names.llm_improved_id_column_drug",
                "params:column_names.llm_normalized_id_column_drug",
                "params:column_names.llm_normalized_label_column_drug",
            ],
            outputs="ema_11",
            name = "normalize-drug-ema",
        ),

        node(
            func=nodes.deduplicate_entities,
            inputs=[
                "ema_11",
                "params:column_names.llm_improved_id_column_drug",
                "params:column_names.llm_improved_id_column",
                "params:column_names.deduplication_column",
            ],
            outputs="ema_12",
            name="deduplicate-ema"
        ),





#########################################
######## PMDA LIST #######################
#########################################


        node(
            func=nodes.standardize_pmda_rows,
            inputs=[
                "pmda_approvals",
                "params:column_names",
            ],
            outputs="pmda_preprocessed",
            name="standardize-pmda"

        ),

        node(
            func=nodes.extract_named_diseases,
            inputs = [
                "pmda_preprocessed",
                "params:column_names.drug_name_column",
                "params:column_names.indications_text_column",
                "params:column_names.disease_list_column",
                "params:structured_list_prompt",
            ],
            outputs = "pmda_1",
            name = "extract-disease-lists-pmda"
        ),

        node(
            func=nodes.flatten_list,
            inputs = [
                "pmda_1",
                "params:column_names.disease_list_column",
                "params:column_names.drug_name_column",
                "params:column_names.indications_text_column",
                "params:column_names.disease_name_column",
            ],
            outputs = "pmda_2",
            name = "flatten-list-pmda"
        ),

        node(
            func=nodes.clean_list,
            inputs = [
                "pmda_2",
                "params:column_names.disease_name_column",
                "params:strings_to_clean_from_disease_list",
                "params:cleaning_regex_sub_pattern"
            ],
            outputs = "pmda_3",
            name = "clean-list-pmda"
        ),

        node(
            func=nodes.resolve_concepts,
            inputs = [
                "pmda_3",
                "params:column_names.disease_name_column",
                "params:column_names.disease_id_column",
                "params:column_names.disease_label_column",
                "params:biolink_type_disease"
            ],
            outputs = "pmda_4",
            name = "nameres-pmda-diseases"
        ),

        node(
            func=nodes.check_nameres_llm,
            inputs = [
                "pmda_4",
                "params:column_names.disease_name_column",
                "params:column_names.disease_label_column",
                "params:id_correct_incorrect_tag_disease",
                "params:column_names.llm_true_false_column_disease"
            ],
            outputs = "pmda_5",
            name = "nameres-auto-qc-pmda"
        ),

        node(
            func=nodes.llm_improve_ids,
            inputs = [
                "pmda_5",
                "params:column_names.disease_name_column",
                "params:llm_best_id_tag",
                "params:biolink_type_disease",
                "params:column_names.disease_id_column",
                "params:column_names.llm_true_false_column_disease",
                "params:column_names.llm_improved_id_column",
            ],
            outputs = "pmda_6",
            name = "llm-id-improvement-pmda"
        ),
        
        node(
            func=nodes.resolve_concepts,
            inputs = [
                "pmda_6",
                "params:column_names.drug_name_column",
                "params:column_names.drug_id_column",
                "params:column_names.drug_label_column",
                "params:biolink_type_drug"
            ],
            outputs = "pmda_7",
            name = "nameres-pmda-drugs",
        ),

        node(
            func=nodes.add_normalized_llm_tag_ids,
            inputs= [
                "pmda_7",
                "params:column_names.llm_improved_id_column",
                "params:column_names.llm_normalized_id_column_disease",
                "params:column_names.llm_normalized_label_column_disease",
            ],
            outputs="pmda_8",
            name = "normalize-diseases-pmda"
        ),

        node(
            func=nodes.check_nameres_llm,
            inputs = [
                "pmda_8",
                "params:column_names.drug_name_column",
                "params:column_names.drug_label_column",
                "params:id_correct_incorrect_tag_drug",
                "params:column_names.llm_true_false_column_drug"
            ],
            outputs = "pmda_9",
            name = "nameres-auto-qc-drug-pmda"
        ),

        node(
            func=nodes.llm_improve_ids,
            inputs = [
                "pmda_9",
                "params:column_names.drug_name_column",
                "params:llm_best_id_tag_drug",
                "params:biolink_type_drug",
                "params:column_names.drug_id_column",
                "params:column_names.llm_true_false_column_drug",
                "params:column_names.llm_improved_id_column_drug",
            ],
            outputs = "pmda_10",
            name = "llm-id-improvement-drug-pmda"
        ),

        node(
            func=nodes.add_normalized_llm_tag_ids,
            inputs= [
                "pmda_10",
                "params:column_names.llm_improved_id_column_drug",
                "params:column_names.llm_normalized_id_column_drug",
                "params:column_names.llm_normalized_label_column_drug",
            ],
            outputs="pmda_11",
            name = "normalize-drug-pmda",
        ),

        node(
            func=nodes.deduplicate_entities,
            inputs=[
                "pmda_11",
                "params:column_names.llm_improved_id_column_drug",
                "params:column_names.llm_improved_id_column",
                "params:column_names.deduplication_column",
            ],
            outputs="pmda_12",
            name="deduplicate-pmda"
        ),
        
        node(
            func=nodes.join_lists,
            inputs=[
                "dailymed_12",
                "ema_12",
                "pmda_12",
                "params:column_names",
            ],
            outputs="matrix_indication_list",
            name="join_lists"
        ),

        node(
            func=nodes.downfill_list_mondo,
            inputs=[
                "matrix_indication_list",
                "mondo_edges",
                "mondo_nodes",
                "params:column_names",
            ],
            outputs="matrix_indication_list_downfilled",
            name="downfill-list"
        ),

    ])
