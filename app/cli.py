"""Command-line entry point: `python -m app.cli --help`."""
import argparse


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="sports-play-by-play-generator", description="Sports fans get a command-line assistant that generates short, dynamic commentary snippets for fictional game scenarios, trained on real sports news articles.")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("hello", help="check that everything is wired up")
    # TODO: add your own commands here as you work through the milestones.
    args = parser.parse_args(argv)
    if args.command == "hello":
        print("Sports Play-by-Play Generator is ready. Open README.md for your milestones.")


if __name__ == "__main__":
    main()
