#!/bin/bash
# Adapted from the course starter.
name="gan_train"
outdir="outputs"
mkdir -p "$outdir"
export DATA="${DATA:-data}"

sbatch <<EOT
#!/bin/bash
#SBATCH -p mesonet
#SBATCH -N 1
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --mem=16G
#SBATCH --account=${ACCOUNT:?set ACCOUNT}
#SBATCH --job-name=${name}
#SBATCH --output=${outdir}/%x_%j.out
#SBATCH --error=${outdir}/%x_%j.err

cd "\$SLURM_SUBMIT_DIR"
source venv/bin/activate
python -m gan.train_gan --name juliet_smoke --epochs 10 --no_eval
EOT