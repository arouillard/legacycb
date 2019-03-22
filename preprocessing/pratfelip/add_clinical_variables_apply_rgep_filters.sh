sbatch -n 1 -c 1 --mem=32G -t 120 -o add_clinical_variables_apply_rgep_filters.log --wrap="python add_clinical_variables_apply_rgep_filters.py"
