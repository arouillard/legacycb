sbatch -n 1 -c 1 --mem=32G -t 60 -o train_logistic_regression_example.log --wrap="python train_logistic_regression.py train_logistic_regression_input_example.json"
