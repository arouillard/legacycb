sbatch -n 1 -c 1 --mem=32G -t 120 -o swap_in_deconv_variables.log --wrap="python swap_in_deconv_variables.py"
