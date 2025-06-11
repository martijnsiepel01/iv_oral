# main.py

from cli import parse_args, override_config_from_args
from pipeline.run_training import run

if __name__ == "__main__":
    args = parse_args()
    config = override_config_from_args(args)  # merged config
    run(config=config)  # pass this into get_model_name, wandb.init, etc.