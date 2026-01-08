# Reconstruct a new EMA profile with std=0.150
export CUDA_VISIBLE_DEVICES=1

python reconstruct_phema.py --indir="/path/to/your/inputs" \
    --outdir="/path/to/your/outputs" \
    --outkimg=<OUTPUT_KIMG> \
    --outstd=0.10,0.15,0.20