import argparse
import pathlib
from time import strftime, localtime
from omegaconf import OmegaConf
from dmqc.pipeline.train import main


if __name__ == "__main__":
    OmegaConf.register_resolver("now", lambda pattern: strftime(pattern, localtime()))
    cwd = pathlib.Path().cwd()

    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/home/mustafa/Documents/DMQC/conf/")
    parser.add_argument("--exp_root", default="experiment/*.yaml")
    parser.add_argument(
        "--conf_path", default="/home/mustafa/Documents/DMQC/conf/config.yaml"
    )
    args = parser.parse_args()

    experiment_paths = pathlib.Path(args.root).glob(args.exp_root)
    experiment_sorted = sorted([x for x in experiment_paths])
    for experiment in experiment_sorted:

        config = OmegaConf.load(str(args.conf_path))
        conf_cli = OmegaConf.from_cli()
        tmp = str(experiment).split("/")
        config.defaults.experiment = str(tmp[-1])[:-5]

        entry_conf = OmegaConf.load(str(experiment))
        config = OmegaConf.merge(config, entry_conf)
        config = OmegaConf.merge(config, conf_cli)

        main(config)
