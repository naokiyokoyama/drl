from drl.utils.common import construct_config, get_default_parser
from drl.utils.registry import drl_registry


def main():
    parser = get_default_parser()
    args = parser.parse_args()
    config = construct_config(args.opts)
    runner_cls = drl_registry.get_runner(config.RUNNER.name)
    runner = runner_cls(config)


if __name__ == "__main__":
    main()
