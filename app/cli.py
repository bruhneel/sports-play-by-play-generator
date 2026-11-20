"""Command-line entry point for the local sports commentary model."""
import argparse
import math

from app.storage import load_checkpoint, prepare, train


def _positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def _temperature(value: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("must be positive and finite")
    return number


def _sampling_options(command: argparse.ArgumentParser) -> None:
    command.add_argument("--max-tokens", type=_positive_int, default=32, help="maximum new BPE tokens")
    command.add_argument("--temperature", type=_temperature, default=0.8, help="sampling temperature")
    command.add_argument("--top-k", type=_positive_int, default=12, help="maximum candidates per step")
    command.add_argument("--seed", type=int, default=42, help="sampling seed")
    command.add_argument("--num-snippets", type=_positive_int, default=1, help="number of sampled alternatives")


def _run_generation(args: argparse.Namespace) -> None:
    import torch

    from app.core import commentary

    torch.set_num_threads(2)
    try:
        model, tokenizer = load_checkpoint()
    except FileNotFoundError as exc:
        raise ValueError("No trained model found. Run train first.") from exc

    def show(prompt: str) -> None:
        for i in range(args.num_snippets):
            result = commentary(
                model, tokenizer, prompt, args.max_tokens,
                args.temperature, args.top_k, args.seed + i,
            )
            print(result)

    if args.command == "commentate":
        show(args.prompt)
        return

    print("Enter a fictional game scenario. Type /quit to exit.")
    while True:
        try:
            prompt = input("Game> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if prompt == "/quit":
            return
        if prompt:
            show(prompt)


def main(argv: list[str] | None = None) -> None:
    """Dispatch corpus preparation, training, and local generation."""
    parser = argparse.ArgumentParser(
        prog="sports-play-by-play-generator",
        description="Generate short fictional sports commentary with a local GPT.",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("hello", help="check that everything is wired up")
    prep = sub.add_parser("prepare", help="build the local corpus splits")
    prep.add_argument("--seed", type=int, default=42)
    trainer = sub.add_parser("train", help="train the byte-pair tokenizer and CPU GPT")
    trainer.add_argument("--steps", type=_positive_int, default=700)
    trainer.add_argument("--seed", type=int, default=42)
    trainer.add_argument("--width", type=_positive_int, default=96)
    trainer.add_argument("--layers", type=_positive_int, default=2)
    trainer.add_argument("--context", type=_positive_int, default=64)
    trainer.add_argument("--batch-size", type=_positive_int, default=16)
    trainer.add_argument("--vocab-size", type=_positive_int, default=512)
    generator = sub.add_parser("commentate", help="generate commentary for a fictional prompt")
    generator.add_argument("prompt", help="game scenario, such as Penalty in the 89th minute")
    _sampling_options(generator)
    _sampling_options(sub.add_parser("interactive", help="generate commentary in a prompt loop"))
    args = parser.parse_args(argv)

    try:
        match args.command:
            case "hello":
                print("Sports Play-by-Play Generator is ready. Open README.md for your milestones.")
            case "prepare":
                corpus = prepare(seed=args.seed)
                print(f"Prepared {len(corpus['train'])} training lines and {len(corpus['val'])} validation lines.")
            case "train":
                train(args.steps, args.seed, args.width, args.layers, args.context, args.batch_size, args.vocab_size)
            case "commentate" | "interactive":
                _run_generation(args)
    except (ValueError, OSError, RuntimeError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
