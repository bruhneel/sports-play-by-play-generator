"""Write original fictional commentary examples for four sports."""
from itertools import product
from pathlib import Path

SCENARIOS = (
    (
        [f"Last over, {runs} needed off 6 balls" for runs in range(6, 21)]
        + [f"Final ball, {runs} needed to win" for runs in (2, 3, 4, 6)],
        [
            "The bowler charges in, and the batter sends it racing through cover.",
            "The batter waits deep in the crease and pulls it into the stands!",
            "A full toss on the pads, and the ball flies past square leg.",
            "The bowler nails the yorker and the batter can only dig it out.",
            "The slow ball fools the batter, but they sneak through for a single.",
            "The batter clears the front leg and launches it over long-on!",
            "It drops just short of the fielder and they sprint back for two.",
            "The edge flies wide of the keeper and runs away to the rope!",
        ],
    ),
    (
        [f"Penalty in the {minute}th minute" for minute in range(76, 96)]
        + ["Free kick just outside the box", "Corner in stoppage time"],
        [
            "The keeper dives left, but the ball is already in the opposite corner!",
            "The striker takes a breath, steps up, and fires it low into the net!",
            "The keeper guesses right and pushes the shot around the post.",
            "The ball clips the underside of the bar and bounces over the line!",
            "The run-up is short and the shot whistles just wide.",
            "The striker sends the ball low and hard, and the net ripples.",
            "The keeper gets a glove to it but cannot keep it out!",
            "The shot hits the post and rolls along the goal line before it is cleared.",
        ],
    ),
    (
        [f"Basketball, {seconds} seconds left on the clock" for seconds in range(1, 21)],
        [
            "The guard steps back beyond the arc and buries the three!",
            "The pass finds the cutter and the layup drops off the glass.",
            "The defender gets a fingertip to the shot and the buzzer sounds.",
            "The point guard drives, draws two defenders, and kicks it to the corner.",
            "The shot rattles twice on the rim before falling through.",
            "The ball goes inside and the center finishes with a two-handed dunk!",
            "They swing the ball to the wing for a clean look at the buzzer.",
            "The defense switches and forces a fallaway jumper from the baseline.",
        ],
    ),
    (
        [f"Tennis, {score} in the final set" for score in ("15-all", "30-all", "40-all", "deuce",
                                                          "5-4", "4-5", "6-5", "5-6")]
        + ["Break point in the final set", "Match point on the second serve"],
        [
            "The return lands deep and the server floats a backhand long.",
            "The serve kicks high and the return skids beyond the baseline.",
            "A long rally ends with a forehand winner down the line!",
            "The drop shot dies just over the net and the crowd erupts.",
            "The server finds the corner and follows it with a clean volley.",
            "The lob sails over the player's reach and lands inside the line!",
            "The rally swings from corner to corner until the backhand catches the tape.",
            "The second serve comes back hard, and the server counters crosscourt.",
        ],
    ),
)


def commentary_lines() -> list[str]:
    """Return one line for each scenario and possible play."""
    return [f"{prompt}, {play[0].lower() + play[1:]}" for prompts, plays in SCENARIOS
            for prompt, play in product(prompts, plays)]


if __name__ == "__main__":
    Path(__file__).with_name("commentary.txt").write_text(
        "\n".join(commentary_lines()) + "\n", encoding="utf-8"
    )
