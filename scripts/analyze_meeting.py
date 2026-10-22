#!/usr/bin/env python3
"""Analyze a meeting transcript with a local LLM: Ollama, or an OpenAI-compatible
server such as LM Studio.

Long transcripts are cut into chunks; each chunk gets its own notes file, then a
final pass merges them into summary.md following the (optional) template.

    python scripts/analyze_meeting.py transcripts/Tuesday-fr.json -m nemotron-3-nano:4b
    python scripts/analyze_meeting.py notes.txt --template my-structure.md -o out/
    python scripts/analyze_meeting.py rec.json --api openai -m google/gemma-4-26b-a4b-qat
"""

import argparse
import json
import os
import pathlib
import re
import sys
import urllib.request

DEFAULT_TEMPLATE = """# Meeting notes

## Summary
## Attendees
## Topics discussed
## Decisions
## Action items (owner - task - deadline)
## Open questions
"""

TIMESTAMP = re.compile(r"^(\d+$|\d\d:\d\d:\d\d[.,]\d+ --> |WEBVTT)")


def read_transcript(path: pathlib.Path, names: dict[str, str] | None = None) -> str:
    """Plain text from whisperX .json (speaker-labelled) or .txt/.srt/.vtt.

    `names` renames diarization labels, so the notes say "Alice" and not "SPEAKER_02".
    """
    names = names or {}
    if path.suffix == ".json":
        segments = json.loads(path.read_text())["segments"]
        return "\n".join(
            f"{names.get(s.get('speaker'), s.get('speaker', 'UNKNOWN'))}: {s['text'].strip()}"
            for s in segments
        )
    lines = (l.strip() for l in path.read_text().splitlines())
    return "\n".join(l for l in lines if l and not TIMESTAMP.match(l))


def chunk(text: str, size: int) -> list[str]:
    """Split on line boundaries into pieces of at most `size` chars."""
    chunks, current = [], ""
    for line in text.splitlines(keepends=True):
        if current and len(current) + len(line) > size:
            chunks.append(current)
            current = ""
        current += line
    return chunks + [current] if current else chunks


def request(prompt: str, args) -> urllib.request.Request:
    # Low temperature because this is extraction, not writing: at the default the
    # model invents attendees who were never there, and spells one name three ways.
    # The output cap because nothing else stops a loop: a merge once ran 61 minutes on
    # a 4 KB input. 6000, not 2048, because a thinking trace shares the budget: at
    # 2048 a 9-section answer came back empty.
    if args.api == "openai":
        # LM Studio and anything else speaking /v1/chat/completions. The context
        # window is fixed when the model is loaded, so --ctx has nothing to set here:
        # load the model with a window that fits --chunk-chars plus the output, or the
        # server truncates the prompt exactly the way ollama does below.
        url = f"{args.host}/chat/completions"
        body = {
            "model": args.model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "temperature": 0.2,
            "max_tokens": 6000,
        }
    else:
        # num_ctx matters: ollama defaults to a small window and truncates the FRONT
        # of the prompt, which silently eats the instructions.
        # think must be set explicitly. Left unset the model still reasons, but ollama
        # does not split the trace out: it arrives as plain untagged prose at the top of
        # response, spending the num_predict budget and truncating the real answer.
        # Measured on one prompt: unset 78 tokens with the trace inline, false 24 clean,
        # true 78 with the trace in a separate `thinking` field. True because false cost
        # the merge its judgement - it deleted a real decision and wrote comma splices -
        # and the trace stays out of `response` either way, at ~3x the tokens.
        url = f"{args.host}/api/generate"
        body = {
            "model": args.model,
            "prompt": prompt,
            "stream": False,
            "think": True,
            "options": {"num_ctx": args.ctx, "temperature": 0.2, "num_predict": 6000},
        }
    headers = {"Content-Type": "application/json"}
    if args.api == "openai" and args.token_file:
        # hai-context rejects an unauthenticated call. Read here, per request, so the
        # token never sits on a command line or in an environment variable.
        headers["Authorization"] = f"Bearer {args.token_file.expanduser().read_text().strip()}"
    return urllib.request.Request(url, json.dumps(body).encode(), headers)


def generate(prompt: str, args) -> str:
    with urllib.request.urlopen(request(prompt, args)) as resp:
        data = json.load(resp)
    # ollama's `thinking` field, and openai's `reasoning_content`, are both deliberately
    # ignored: only what the model committed to as the answer is kept.
    answer = (
        data["choices"][0]["message"]["content"]
        if args.api == "openai"
        else data["response"]
    )
    return strip_think(answer)


def strip_think(answer: str) -> str:
    """Drop a reasoning trace some models inline in the response anyway.

    An unclosed <think> means generation was cut off mid-trace: without the second
    branch rsplit finds no delimiter and hands back the whole scratchpad.
    """
    if "</think>" in answer:
        answer = answer.rsplit("</think>", 1)[-1]
    elif "<think>" in answer:
        answer = answer.split("<think>", 1)[0]
    return answer.strip()


QUOTE = re.compile(r'["“«]\s*([^"”»]+?)\s*["”»]')
# Sections whose bullets claim something was said; each must prove it with a quote.
CHECKED = ("decisions", "action items", "open questions")
# An action item opens with its owner as a tag: `- #Alice - send the draft - "..."`.
OWNER = re.compile(r"^\s*[-*]\s+#([\w-]+)")


def words(text: str) -> str:
    return " ".join(re.findall(r"\w+", text.lower()))


def tag(name: str) -> str:
    """A speaker name as an owner tag spells it: no spaces."""
    return re.sub(r"\s+", "-", name.strip())


def spoken(source: str) -> dict[str, str]:
    """{lowercased tag: everything that speaker said, as words()}, padded for
    whole-word search.

    Empty unless every line is `Name: text`, as read_transcript writes a .json: a
    plain .txt has no speakers to hold an owner to.
    """
    lines = [l for l in source.splitlines() if l.strip()]
    found = [m for l in lines if (m := re.match(r"^([^:\n]{1,40}): (.*)$", l))]
    if not lines or len(found) < len(lines):
        return {}
    out: dict[str, str] = {}
    for m in found:
        key = tag(m[1]).lower()
        out[key] = f"{out.get(key, ' ')}{words(m[2])} "
    return out


def keep_quoted(md: str, source: str) -> tuple[str, list[str]]:
    """Drop every Decisions / Action items / Open questions bullet whose quote is not
    in `source`.

    The model files suggestions, opinions and its own inferences as decisions, and no
    prompt wording stopped it. A bullet survives only if it carries words of at least
    three tokens, in quotes, that appear verbatim (case and punctuation aside) in the
    transcript it came from, plus at least two words of its own saying what was
    settled. An action item must also open with a `#Owner` tag, and
    when the transcript names its speakers the quote must be the owner's own words:
    the model pinned other people's shrugs ("Je ne sais pas.") on owners as tasks
    (2026-10-21). A task someone was handed and only answered "ok" to is lost; a wrong
    owner in the notes is worse. A heading left with no bullet goes too.
    """
    said, voices = f" {words(source)} ", spoken(source)
    kept, dropped, section, seen = [], [], None, set()
    for line in md.splitlines():
        m = re.match(r"^#+\s+(.+?)\s*$", line)
        if m:
            section = m.group(1).strip().lower()
        elif re.match(r"^\s*[-*]\s", line) and section and section.startswith(CHECKED):
            where: str | None = said
            if section.startswith("action items"):
                owner = OWNER.match(line)
                where = (voices.get(owner[1].lower()) if voices else said) if owner else None
            quotes = [words(q) for q in QUOTE.findall(line)]
            # The claim itself, outside the quotes: a bare quote ("c'est bien") is a
            # remark, not a decision (2026-10-22).
            claim = words(QUOTE.sub(" ", line)).split()
            # A small model copies the template's `<placeholder>`, and loops: one task
            # came back 73 times, each with another real line as its quote (2026-10-22).
            placeholder = re.search(r"<[^>]*>", QUOTE.sub(" ", line))
            if (len(claim) < 2 or placeholder or (section, *claim) in seen or not where
                    or not any(len(q.split()) >= 3 and f" {q} " in where for q in quotes)):
                dropped.append(line.strip())
                continue
            seen.add((section, *claim))
        kept.append(line)
    out = "\n".join(kept)
    # A checked heading followed only by blank lines up to the next heading or the end.
    names = "|".join(re.escape(c) for c in CHECKED)
    out = re.sub(rf"^#+\s+(?:{names})[^\n]*(?:\n\s*)*(?=#|\Z)", "", out, flags=re.M | re.I)
    return out.strip(), dropped


# Words that say something was agreed, in the languages these meetings are held in.
# shortcut: French and English only, and a fixed list; make it a setting when a meeting
# in another language comes through.
AGREED = re.compile(
    r"\b(on est d'accord|je suis d'accord|mis d'accord|c'est décidé|on a décidé|we agreed|agreed)\b", re.I
)


def agreements(text: str, before: int = 2) -> str:
    """A `## Said agreements` section: every transcript line voicing an agreement,
    with the lines that led to it, verbatim.

    Plain code, so it invents nothing and loses nothing: the model dropped the
    meeting's one explicit agreement in one draft of two (2026-10-22). A question
    ("on est d'accord ?") is not an agreement. Empty when nothing matched.
    """
    lines = text.splitlines()
    hits = [i for i, l in enumerate(lines) if AGREED.search(l) and "?" not in l]
    blocks = ["\n".join(f"> {l}" for l in lines[max(0, i - before): i + 1]) for i in hits]
    return "## Said agreements\n\n" + "\n\n".join(blocks) if blocks else ""


def checked_bullets(notes: list[str]) -> dict[str, list[str]]:
    """{"Decisions": [...], "Action items": [...]} from every part, deduped by quote.

    Merged by code, not by the model: the merge pass reworded a verbatim quote and
    keep_quoted then deleted the meeting's one real decision (2026-10-21).
    """
    out: dict[str, list[str]] = {}
    seen: set[str] = set()
    for note in notes:
        section = None
        for line in note.splitlines():
            m = re.match(r"^#+\s+(.+?)\s*$", line)
            if m:
                name = m.group(1).strip()
                section = name if name.lower().startswith(CHECKED) else None
            elif section and re.match(r"^\s*[-*]\s", line):
                key = " | ".join(words(q) for q in QUOTE.findall(line)) or words(line)
                if key not in seen:
                    seen.add(key)
                    out.setdefault(section, []).append(line.rstrip())
    return out


def with_checked(md: str, bullets: dict[str, list[str]]) -> str:
    """`md` with its Decisions / Action items replaced by `bullets`, placed before
    Open questions (or at the end)."""
    kept, skip = [], False
    for line in md.splitlines():
        m = re.match(r"^#+\s+(.+?)\s*$", line)
        if m:
            skip = m.group(1).strip().lower().startswith(CHECKED)
        if not skip:
            kept.append(line)
    body = "\n".join(kept).strip()
    order = sorted(bullets, key=lambda h: next(i for i, c in enumerate(CHECKED) if h.lower().startswith(c)))
    block = "\n\n".join(f"## {h}\n" + "\n".join(bullets[h]) for h in order if bullets[h])
    if not block:
        return body
    head, found, tail = body.partition("## Open questions")
    return f"{head.rstrip()}\n\n{block}\n\n{found}{tail}".strip() if found else f"{body}\n\n{block}"


def first_heading(template: str) -> str:
    """The template's own first `## ` heading; what the prose pass fills in.

    Hardcoding "## Summary" here silently collapsed every template that does not
    have one: no split point, so the whole document counted as prose and the other
    headings never survived the merge.
    """
    return next((l for l in template.splitlines() if l.startswith("## ")), "## Summary")


def split_summary(md: str, head: str = "## Summary") -> tuple[str, str]:
    """One notes document -> (prose under `head`, the bullet sections below it).

    Tolerant on purpose. A document with no headings at all is all summary, and any
    heading this script does not know about travels with the bullets, so a model that
    invents a section still cannot make content disappear.
    """
    _, found, rest = md.partition(head)
    if not found:
        return ("", md) if md.lstrip().startswith("## ") else (md.strip(), "")
    body, found, tail = rest.partition("\n## ")
    return body.strip(), ("## " + tail).strip() if found else ""


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("transcript", type=pathlib.Path)
    p.add_argument("-m", "--model", help="default: nemotron-3-nano:4b; required with --api openai")
    p.add_argument("-t", "--template", type=pathlib.Path, help="markdown structure to follow")
    p.add_argument("-o", "--out", type=pathlib.Path, help="output dir (default: <transcript>-notes/)")
    p.add_argument(
        "-s",
        "--speakers",
        default="",
        help="rename diarization labels: 'SPEAKER_01=Alice,SPEAKER_02=Bob'",
    )
    p.add_argument("-c", "--chunk-chars", type=int, default=12000)
    p.add_argument("--ctx", type=int, default=16384, help="model context window (ollama only)")
    p.add_argument(
        "--api",
        choices=("ollama", "openai"),
        default="ollama",
        help="ollama's /api/generate, or an OpenAI-compatible /v1 server (LM Studio)",
    )
    p.add_argument(
        "--host",
        help="default: http://localhost:11434 (ollama) or http://localhost:1234/v1 (openai)",
    )
    p.add_argument(
        "--token-file",
        type=pathlib.Path,
        default=os.environ.get("LLM_TOKEN_FILE") or None,
        help="file holding a bearer token for --api openai (default: $LLM_TOKEN_FILE)",
    )
    args = p.parse_args()

    if args.api == "openai":
        args.host = args.host or "http://localhost:1234/v1"
        # No guessing from /v1/models: it lists everything downloaded, not what is
        # loaded, so picking one silently runs the wrong model.
        if not args.model:
            p.error(f"--api openai needs -m/--model (list them: curl {args.host}/models)")
    else:
        args.host = args.host or "http://localhost:11434"
        args.model = args.model or "nemotron-3-nano:4b"

    template = args.template.read_text() if args.template else DEFAULT_TEMPLATE
    out = args.out or args.transcript.with_name(args.transcript.stem + "-notes")
    out.mkdir(parents=True, exist_ok=True)

    names = dict(pair.split("=", 1) for pair in args.speakers.split(",") if pair)
    # Naming the attendees outright is the one thing that stops the model inventing an
    # extra person or spelling one name three different ways.
    roster = (
        f"Only people here: {', '.join(names.values())}. No other speaker names. "
        f"No spelling variants. "
        if names
        else ""
    )
    text = read_transcript(args.transcript, names)
    parts = chunk(text, args.chunk_chars)
    # The tags keep_quoted accepts, so the model writes #Alice and not "Alice:".
    owners = ", ".join(sorted({f"#{tag(l.split(': ', 1)[0])}" for l in text.splitlines() if spoken(l)}))
    owners = owners or "#Name"
    print(f"{len(parts)} chunk(s) -> {out}/", file=sys.stderr)

    notes = []
    for i, part in enumerate(parts, 1):
        # A killed run (closed terminal, OOM) shouldn't cost the chunks already paid
        # for. Delete the part file, or the whole -notes dir, to force a redo.
        done = out / f"part-{i:02d}.md"
        # .strip(), not st_size: a model that returns nothing still writes a newline,
        # and a 1-byte part was being reused as if the chunk had been done.
        if done.exists() and done.read_text().strip():
            print(f"  chunk {i}/{len(parts)} already done, reusing", file=sys.stderr)
            notes.append(done.read_text().strip())
            continue

        print(f"  chunk {i}/{len(parts)}...", file=sys.stderr)
        # Instructions repeated after the transcript: last tokens are the ones the
        # model actually follows, and front-truncation can't eat them.
        answer = generate(
            f"You take notes for a meeting. Below is part {i} of {len(parts)} of the "
            f"transcript, then the note structure to fill in.\n\n"
            f"--- TRANSCRIPT (part {i}/{len(parts)}) ---\n{part}\n"
            f"--- END TRANSCRIPT ---\n\n"
            f"Write the notes for part {i} using this structure, keeping the headings "
            f"you use in the same order:\n\n{template}\n\n"
            # Short imperative lines, not prose: a 4b model drops clauses out of long
            # flowing rules, and these are the last tokens before it starts writing.
            f"Rules. {roster}Only what transcript says. Unsure - leave out. Short and "
            f"right beats long and invented. Transcript is machine-made and garbles "
            f"words: passage makes no sense - drop it, never repeat nonsense. Every "
            f"decision, action item and open question ends with the speaker's exact words "
            f"in double quotes, copied from the transcript in its own language. No exact "
            f"words - no bullet. Action item starts with the tag of the person who "
            f"committed ({owners}) and quotes that person. No owner - no bullet. "
            f"Third person. Never 'you' or "
            f"'I'. {first_heading(template).lstrip('# ')} always. OPTIONAL section with nothing in it - delete that "
            f"heading. Never write 'none'. Never leave an empty heading. Output the "
            f"structure only.",
            args,
        )
        answer, dropped = keep_quoted(answer, part)
        for line in dropped:
            print(f"    dropped, quote not found: {line}", file=sys.stderr)
        done.write_text(answer + "\n")
        notes.append(answer)

    merged = "\n\n---\n\n".join(notes)
    if len(parts) > 1:
        if first_heading(template).lstrip("# ").lower().startswith("summary"):
            merged = merge(notes, template, roster, args)
        else:
            merged = merge_sections(notes, template, roster, args) or merged
        merged = with_checked(merged, checked_bullets(notes))
    # Again on the whole transcript: a merge pass can reword a quote it was given.
    merged, dropped = keep_quoted(merged, "\n".join(parts))
    for line in dropped:
        print(f"  merge: dropped, quote no longer matched: {line}", file=sys.stderr)
    merged = "\n\n".join(filter(None, [merged, agreements(text)]))
    (out / "summary.md").write_text(merged + "\n")
    print(out / "summary.md")


def template_headings(template: str) -> list[str]:
    """The template's `## ` headings, in order."""
    return [l.rstrip() for l in template.splitlines() if l.startswith("## ")]


def sections(md: str, headings: list[str]) -> dict[str, str]:
    """A notes document -> {template heading: body}.

    Matched on heading text, not level: a model writes `# Check-in (5')` about as
    often as `##`. A heading the template does not have (the `# Meeting notes` title,
    an invented `### sub-topic`) stays inside the section it appears in, so nothing
    is silently dropped.
    """
    known = {h.lstrip("# ").strip().lower(): h for h in headings}
    out: dict[str, list[str]] = {h: [] for h in headings}
    current = None
    for line in md.splitlines():
        m = re.match(r"^#+\s+(.+?)\s*$", line)
        hit = known.get(m.group(1).strip().lower()) if m else None
        if hit:
            current = hit
        elif current:
            out[current].append(line)
    return {h: "\n".join(body).strip() for h, body in out.items()}


def merge_sections(notes: list[str], template: str, roster: str, args) -> str:
    """Merge part notes one section at a time, keeping the template's own structure.

    One small call per section beats one big call: the 4b model loops on a whole
    transcript, but merging a single section is a few hundred characters of input.
    """
    headings = template_headings(template)
    per_part = [sections(note, headings) for note in notes]
    comments = sections(template, headings)

    merged = []
    for h in headings:
        pieces = [f"PART {i}\n{p[h]}" for i, p in enumerate(per_part, 1) if p.get(h)]
        if not pieces:
            continue
        if len(pieces) == 1:
            merged.append(f"{h}\n{pieces[0].split(chr(10), 1)[1].strip()}")
            continue
        body = generate(
            f"--- SECTION \"{h.lstrip('# ')}\" FROM {len(pieces)} PARTS OF ONE MEETING ---\n"
            + "\n\n".join(pieces)
            + f"\n--- END ---\n\nMerge these into one section.\n\n{comments[h]}\n\n"
            f"Rules. {roster}Keep every fact. Never invent one. One bullet per real "
            f"thing, even when parts word it differently. Drop a bullet that reads as "
            f"nonsense. Third person. Never 'you' or 'I'. Never write 'none'. Output "
            f"the section body only, with no heading.",
            args,
        )
        body = re.sub(rf"^#+\s*{re.escape(h.lstrip('# '))}\s*", "", body).strip()
        if body:
            merged.append(f"{h}\n{body}")
    return "\n\n".join(merged).strip()


def merge(notes: list[str], template: str, roster: str, args) -> str:
    """Merge the part notes in two passes: prose first, then the bullets.

    One call cannot restructure prose by thread, dedupe bullets and re-apply the
    "is this really a decision" filters at the same time. The single-pass version
    kept part boundaries as paragraphs, dropped a whole part, and passed every
    rhetorical question straight through to Open questions.
    """
    n = len(notes)
    head = first_heading(template)
    split = [split_summary(note, head) for note in notes]
    summaries = "\n\n".join(f"PART {i}\n{s}" for i, (s, _) in enumerate(split, 1) if s)
    bullets = "\n\n".join(b for _, b in split if b)

    # The preamble (language instruction) is above the first heading and matters to
    # both passes; the section comments are what each pass is actually shaped by.
    preamble = template.split("## ", 1)[0].strip()
    tpl_summary, tpl_bullets = split_summary(template, head)

    narrative = generate(
        f"--- SUMMARIES FROM {n} PARTS OF ONE MEETING ---\n{summaries}\n--- END ---\n\n"
        f"Write the summary of the whole meeting.\n\n{preamble}\n\n"
        f"{head}\n{tpl_summary}\n\n"
        # No rule the model can argue with. Told to weigh "two parts on one thread vs
        # one part with two threads" it spent the whole token budget reasoning out
        # loud and never wrote the summary.
        f"Rules. {roster}Write the finished summary. Never explain your work, never "
        f"list the parts, never think out loud. Every fact comes from the summaries "
        f"above. Never add a fact. Cover all {n} parts. Organise by topic, not by "
        f"part. A part that covered four topics becomes four paragraphs. Two parts on "
        f"one topic become one paragraph. Never copy a part summary as a paragraph. "
        f"Blank line between paragraphs. Never one block. Never open a paragraph with "
        f"'The meeting'. Third person. Never 'you' or 'I'. Never pad. No heading.\n\n"
        f"The summary starts on the next line.\n",
        args,
    )
    narrative = re.sub(rf"^#+ *{re.escape(head.lstrip('# '))}\s*", "", narrative).strip()

    facts = ""
    if bullets:
        facts = generate(
            f"--- DECISIONS, ACTIONS AND QUESTIONS FROM {n} PARTS OF ONE MEETING ---\n"
            f"{bullets}\n--- END ---\n\n"
            f"Merge these into one list.\n\n{preamble}\n\n{tpl_bullets}\n\n"
            # Delete first, merge second. With "merge only, adding a bullet is the
            # worst error" up front the model read the whole job as preservation and
            # filtered nothing - the output was the parts' bullets verbatim.
            f"Rules. {roster}Keep every decision and every action item, each with its "
            f"quoted words copied unchanged. Deleting one is the worst error here. Open questions are different - most of that "
            f"list does not belong, so cut it hard: question answered during the "
            f"meeting - delete; rhetorical question - delete; procedural chatter - "
            f"delete; question nobody has to act on - delete; bullet that reads as "
            f"nonsense - delete. Then merge what is left: one bullet per real thing, "
            f"even when parts word it differently. Never invent a bullet. Third "
            f"person. Never 'you' or 'I'. Section with nothing left - delete that "
            f"heading. Never write 'none'. Output headings and bullets only. No "
            f"summary paragraph.",
            args,
        )
        # A pass that summarises anyway would duplicate the prose above it.
        if head in facts:
            facts = split_summary(facts, head)[1]
        # The model drops the ## about half the time; a bare "Decisions" line renders
        # as body text and the section stops being a section.
        facts = re.sub(
            r"^(Decisions|Action items|Open questions)\s*$",
            r"## \1",
            facts,
            flags=re.M,
        )

    merged = f"{head}\n{narrative}\n\n{facts}".strip()
    # Both passes empty means the model failed; the raw parts beat an empty file.
    return merged if narrative else "\n\n---\n\n".join(notes)


def selftest() -> None:
    assert chunk("a\nb\nc\n", 4) == ["a\nb\n", "c\n"]
    assert chunk("", 10) == []
    assert "".join(chunk("x" * 50, 7)) == "x" * 50
    tmp = pathlib.Path("/tmp/_m.srt")
    tmp.write_text("1\n00:00:01,000 --> 00:00:02,000\nhello\n\n")
    assert read_transcript(tmp) == "hello"
    tmp = tmp.with_suffix(".json")
    tmp.write_text('{"segments":[{"speaker":"S1","text":" hi "},{"text":"ho"}]}')
    assert read_transcript(tmp) == "S1: hi\nUNKNOWN: ho"
    assert read_transcript(tmp, {"S1": "Alice"}) == "Alice: hi\nUNKNOWN: ho"
    # The two backends disagree on url, on where the prompt goes and on where the
    # answer comes back; getting either half crossed sends a valid-looking request
    # nowhere useful.
    ns = argparse.Namespace(api="ollama", host="http://h", model="m", ctx=99, token_file=None)
    req = request("p", ns)
    assert req.full_url == "http://h/api/generate"
    assert json.loads(req.data)["prompt"] == "p"
    assert json.loads(req.data)["options"]["num_ctx"] == 99
    ns.api = "openai"
    req = request("p", ns)
    assert req.full_url == "http://h/chat/completions"
    assert json.loads(req.data)["messages"] == [{"role": "user", "content": "p"}]
    assert json.loads(req.data)["max_tokens"] == 6000
    assert not req.has_header("Authorization")
    tok = pathlib.Path("/tmp/_m.token")
    tok.write_text("s3cret\n")
    ns.token_file = tok
    assert request("p", ns).get_header("Authorization") == "Bearer s3cret"
    ns.api = "ollama"  # ollama never gets the token
    assert not request("p", ns).has_header("Authorization")
    assert strip_think("<think>hmm</think>\n## Summary") == "## Summary"
    assert strip_think("## Summary\n<think>cut off mid-tho") == "## Summary"
    assert strip_think(" plain ") == "plain"
    assert split_summary("## Summary\nprose\n\n## Decisions\n- a\n") == (
        "prose",
        "## Decisions\n- a",
    )
    assert split_summary("## Summary\nonly prose\n") == ("only prose", "")
    # A model that skips or invents headings must not cost us the content.
    assert split_summary("loose text") == ("loose text", "")
    assert split_summary("## Notes\n- a\n") == ("", "## Notes\n- a\n")
    assert first_heading("# T\n\n## Check-in (5')\n<!-- x -->\n\n## Events\n") == "## Check-in (5')"
    assert first_heading("no headings here") == "## Summary"
    tpl = "# T\n\n## Check-in (5')\n<!-- c -->\n\n## Events (10')\n<!-- e -->\n"
    assert template_headings(tpl) == ["## Check-in (5')", "## Events (10')"]
    # Level-tolerant, and an unknown heading stays inside the section it sits in.
    got = sections("# Meeting notes\n# Check-in (5')\n- a\n### Sub\n- b\n## Events (10')\n- c\n",
                   ["## Check-in (5')", "## Events (10')"])
    assert got == {"## Check-in (5')": "- a\n### Sub\n- b", "## Events (10')": "- c"}, got
    assert split_summary("## Check-in\nprose\n\n## Events\n- a\n", "## Check-in") == (
        "prose",
        "## Events\n- a",
    )
    headed = re.sub(
        r"^(Decisions|Action items|Open questions)\s*$", r"## \1", "Decisions\n- x", flags=re.M
    )
    assert headed == "## Decisions\n- x"
    src = "SPEAKER_00: Bon, on fait des copies pour le collectif.\nSPEAKER_01: Peut-être. Je ne sais pas."
    notes = (
        "## Summary\nprose \"not checked here\"\n\n## Decisions\n"
        "- Copies go to the collective - \"On fait des copies\"\n"
        "- Legal counsel per partner - \"we will not assign counsel\"\n"
        "- Unquoted claim\n- Too short - \"Peut-être\"\n\n"
        "## Action items\n- Alex - ships it - \"I will ship\"\n\n## Open questions\n- why?"
    )
    got, n = keep_quoted(notes, src)
    assert len(n) == 5, n
    assert got == (
        "## Summary\nprose \"not checked here\"\n\n## Decisions\n"
        "- Copies go to the collective - \"On fait des copies\""
    ), got
    # An action item needs a #Owner, and the quote must be that owner's own words.
    acts = ("## Action items\n- #SPEAKER_00 - make copies - \"on fait des copies\"\n"
            "- #SPEAKER_01 - make copies - \"on fait des copies\"\n- make copies - \"on fait des copies\"\n"
            "- #Nobody - make copies - \"on fait des copies\"\n- #speaker_01 - check it - \"je ne sais pas\"")
    assert keep_quoted(acts, src) == (
        "## Action items\n- #SPEAKER_00 - make copies - \"on fait des copies\"\n"
        "- #speaker_01 - check it - \"je ne sais pas\"", ["- #SPEAKER_01 - make copies - \"on fait des copies\"",
        "- make copies - \"on fait des copies\"", "- #Nobody - make copies - \"on fait des copies\""]), keep_quoted(acts, src)
    # No speakers in a plain .txt: the tag is still required, any line may hold the quote.
    assert len(keep_quoted(acts, "Note: on fait des copies\nfin")[1]) == 2  # untagged, unsaid
    assert spoken("Jean Paul: Salut toi.") == {"jean-paul": " salut toi "}
    assert spoken("a: b\nno speaker") == {}
    assert len(keep_quoted("## Open questions\n- Copies? - \"on fait des copies\"\n- why?", src)[1]) == 2
    assert keep_quoted("## Open questions\n- Copies or links? - \"on fait des copies\"", src)[1] == []
    # A quote with no claim of its own beside it says nothing was settled.
    assert len(keep_quoted("## Decisions\n- \"on fait des copies\"\n- Copies go - \"on fait des copies\"", src)[1]) == 1
    # Curly and French quotes count; a quote spanning a word boundary does not.
    assert len(keep_quoted("## Decisions\n- copies agreed «on fait des»", src)[1]) == 0
    assert len(keep_quoted("## Decisions\n- copies agreed “ait des copies”", src)[1]) == 1
    parts = ['## Summary\na\n## Open questions\n- Q - "q three four"\n## Decisions\n- Copies - "on fait des copies"\n'
             '## Action items\n- A - x - "q one two"',
             '## Summary\nb\n## Decisions\n- Copies again - "On fait des copies."\n- Record - "always do recording"']
    got = checked_bullets(parts)
    assert got == {"Open questions": ['- Q - "q three four"'],
                   "Decisions": ['- Copies - "on fait des copies"', '- Record - "always do recording"'],
                   "Action items": ['- A - x - "q one two"']}, got
    # The model's merged lists are replaced, whatever it reworded or dropped, and come
    # back in CHECKED order.
    merged = '## Summary\nab\n\n## Decisions\n- Copies - "we copy"\n\n## Open questions\n- why?'
    assert with_checked(merged, got) == (
        '## Summary\nab\n\n## Decisions\n- Copies - "on fait des copies"\n- Record - "always do recording"'
        '\n\n## Action items\n- A - x - "q one two"\n\n## Open questions\n- Q - "q three four"'
    ), with_checked(merged, got)
    assert with_checked("## Summary\nab", {}) == "## Summary\nab"
    assert with_checked("## Summary\nab", got).endswith('## Open questions\n- Q - "q three four"')
    # A looping model's repeats and its copied placeholders go; the first stays.
    loop = ("## Action items\n- #SPEAKER_00 - make copies - \"on fait des copies\"\n"
            "- #SPEAKER_00 - make copies - \"bon on fait des copies\"\n- #SPEAKER_00 - <task> - \"on fait des copies\"")
    assert keep_quoted(loop, src)[0] == "## Action items\n- #SPEAKER_00 - make copies - \"on fait des copies\""
    talk = "A: on copie ?\nB: oui\nA: Je suis d'accord.\nB: On est d'accord ?\nA: fin"
    assert agreements(talk) == "## Said agreements\n\n> A: on copie ?\n> B: oui\n> A: Je suis d'accord.", agreements(talk)
    assert agreements("A: rien") == "" and agreements("A: ok\nB: Agreed.", 5) == "## Said agreements\n\n> A: ok\n> B: Agreed."
    print("ok")


if __name__ == "__main__":
    selftest() if "--selftest" in sys.argv else main()
