"""Multi-frame vote: N reads of one track -> one verdict. Pure function.

STUDENT DECISIONS (2026-07-19, Stage B spec):
- a text with >= 2 votes wins;
- no_plate reads and technical failures abstain (carry no text);
- no majority but exactly ONE distinct text seen -> accept it (a single
  usable read degrades gracefully to v1's single-read behavior);
- no majority and >= 2 distinct texts -> failed (can't break the tie);
- zero texts: all usable reads said no_plate -> no_plate; anything with
  a technical failure in the mix (or an empty list) -> failed."""

from car_logger.services.plate_result import PlateResult


def vote_on_reads(reads):
    """Return (PlateResult, winner_index).

    winner_index points at the read whose confidence/region the verdict
    carries — the caller saves that crop as the event's visual evidence.
    It is 0 when there is no winning read."""
    votes = {}
    saw_no_plate = False
    saw_technical = False
    for i, read in enumerate(reads):
        if read.status == "success" and read.plate_text:
            votes.setdefault(read.plate_text, []).append(i)
        elif read.status == "no_plate":
            saw_no_plate = True
        else:
            saw_technical = True

    if votes:
        best_text = max(votes, key=lambda text: len(votes[text]))
        indexes = votes[best_text]
        if len(indexes) >= 2 or len(votes) == 1:
            winner = max(indexes,
                         key=lambda i: reads[i].confidence or 0.0)
            best = reads[winner]
            return (PlateResult(best_text, best.confidence, "success",
                                best.region), winner)
        return (PlateResult(None, None, "failed", None), 0)

    if saw_no_plate and not saw_technical:
        return (PlateResult(None, None, "no_plate", None), 0)
    return (PlateResult(None, None, "failed", None), 0)
