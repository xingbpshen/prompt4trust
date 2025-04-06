import sys
from engine.agent import Agent
from util import parse_args_and_config


def main():
    args, config = parse_args_and_config()
    agent = Agent(args, config)
    if args.train:
        agent.train()


if __name__ == "__main__":
    sys.exit(main())
