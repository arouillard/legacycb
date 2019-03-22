sbatch -n 1 -c 1 --mem=32G -t 120 -o apply_rgep_filters.log --wrap="python apply_rgep_filters.py"
