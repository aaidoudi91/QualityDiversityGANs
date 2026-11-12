#!/bin/bash
# Adapted from the course starter.
name="generate_images"
outdir="outputs"
mkdir -p "$outdir"

sbatch <<EOT
#!/bin/bash
#SBATCH -p mesonet
#SBATCH -N 1
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH --mem=16G
#SBATCH --account=${ACCOUNT:?set ACCOUNT}
#SBATCH --job-name=${name}
#SBATCH --output=${outdir}/%x_%j.out
#SBATCH --error=${outdir}/%x_%j.err

cd "\$SLURM_SUBMIT_DIR"
source venv/bin/activate
python -m gan.generate --batch_size 256
EOT