dataset=$1
predict_path=$2

python3 /data/home/jacampos/project/updated_vl/VL-T5/VL-T5/evaluation/create_results_json_memory.py \
        --memory_test_json $1 \
        --model_output_json $2/predictions.json
        
# python3 /data/home/jacampos/project/updated_vl/VL-T5/VL-T5/evaluation/response_evaluation_memory.py \
#         --data_json_path $1 \
#         --model_response_path $2/predictions_response_results.json

python3 /data/home/jacampos/project/updated_vl/VL-T5/VL-T5/evaluation/evaluate_dst_memory.py \
        --input_path_target $1 \
        --input_path_predicted $2/predictions_dst_results.json\
        --output_path_report $2/report.out \
