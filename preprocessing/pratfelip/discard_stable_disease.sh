sbatch -n 1 -c 1 --mem=32G -t 120 -o discard_stable_disease.log --wrap="python discard_stable_disease.py"
