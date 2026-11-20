"""Command-line entry point: `python -m app.cli --help`."""
import argparse


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="sports-play-by-play-generator", description="Sports fans get a command-line assistant that generates short, dynamic commentary snippets for fictional game scenarios, trained on real sports news articles.")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("hello", help="check that everything is wired up")
    prepare_parser = sub.add_parser("prepare", help="build the local corpus splits")
    prepare_parser.add_argument("--seed", type=int, default=42)
    train_parser = sub.add_parser("train", help="train the tokenizer and a small CPU GPT")
    train_parser.add_argument("--steps", type=int, default=700)
    train_parser.add_argument("--seed", type=int, default=42)
    train_parser.add_argument("--width", type=int, default=96)
    train_parser.add_argument("--layers", type=int, default=2)
    train_parser.add_argument("--context", type=int, default=64)
    train_parser.add_argument("--batch-size", type=int, default=16)
    train_parser.add_argument("--vocab-size", type=int, default=512)
    args = parser.parse_args(argv)
    if args.command == "prepare":
        from app.storage import prepare

        corpus = prepare(seed=args.seed)
        print(f"Prepared {len(corpus['train'])} training lines and {len(corpus['val'])} validation lines.")
    if args.command == "train":
        from app.storage import train

        train(args.steps, args.seed, args.width, args.layers, args.context, args.batch_size, args.vocab_size)
    if args.command == "hello":
        print("Sports Play-by-Play Generator is ready. Open README.md for your milestones.")


if __name__ == "__main__":
    main()
