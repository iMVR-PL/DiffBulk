export CUDA_VISIBLE_DEVICES=0

OUTPUT_CSV="./results.csv"

if [[ ! -f "$OUTPUT_CSV" ]]; then
    echo "Config,MAE,MSE,Pearson_Correlation,P_value" > $OUTPUT_CSV
fi

CONFIG_PATH="./test_config.yaml"

if [[ -f "$CONFIG_PATH" ]]; then
    echo "Testing model with config: $CONFIG_PATH"
    python test.py --config_paths "$CONFIG_PATH" --output_csv "$OUTPUT_CSV"
else
    echo "Config file not found: $CONFIG_PATH"
fi

echo "All tests completed. Results saved to $OUTPUT_CSV"
