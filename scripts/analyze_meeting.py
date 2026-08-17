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
    # a 4 KB input. Notes are ~600 words, so 2048 tokens is slack, not a squeeze.
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
            "max_tokens": 2048,
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
            "options": {"num_ctx": args.ctx, "temperature": 0.2, "num_predict": 2048},
        }
    return urllib.request.Request(
        url, json.dumps(body).encode(), {"Content-Type": "application/json"}
    )


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


def split_summary(md: str) -> tuple[str, str]:
    """One notes document -> (summary prose, the bullet sections below it).

    Tolerant on purpose. A document with no headings at all is all summary, and any
    heading this script does not know about travels with the bullets, so a model that
    invents a section still cannot make content disappear.
    """
    _, found, rest = md.partition("## Summary")
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
    parts = chunk(read_transcript(args.transcript, names), args.chunk_chars)
    print(f"{len(parts)} chunk(s) -> {out}/", file=sys.stderr)

    notes = []
    for i, part in enumerate(parts, 1):
        # A killed run (closed terminal, OOM) shouldn't cost the chunks already paid
        # for. Delete the part file, or the whole -notes dir, to force a redo.
        done = out / f"part-{i:02d}.md"
        if done.exists() and done.stat().st_size:
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
            f"words: passage makes no sense - drop it, never repeat nonsense. Quote "
            f"decisions and action items word for word. Third person. Never 'you' or "
            f"'I'. Summary always. OPTIONAL section with nothing in it - delete that "
            f"heading. Never write 'none'. Never leave an empty heading. Output the "
            f"structure only.",
            args,
        )
        done.write_text(answer + "\n")
        notes.append(answer)

    merged = "\n\n---\n\n".join(notes)
    if len(parts) > 1:
        merged = merge(notes, template, roster, args)
    (out / "summary.md").write_text(merged + "\n")
    print(out / "summary.md")


def merge(notes: list[str], template: str, roster: str, args) -> str:
    """Merge the part notes in two passes: prose first, then the bullets.

    One call cannot restructure prose by thread, dedupe bullets and re-apply the
    "is this really a decision" filters at the same time. The single-pass version
    kept part boundaries as paragraphs, dropped a whole part, and passed every
    rhetorical question straight through to Open questions.
    """
    n = len(notes)
    split = [split_summary(note) for note in notes]
    summaries = "\n\n".join(f"PART {i}\n{s}" for i, (s, _) in enumerate(split, 1) if s)
    bullets = "\n\n".join(b for _, b in split if b)

    # The preamble (language instruction) is above the first heading and matters to
    # both passes; the section comments are what each pass is actually shaped by.
    preamble = template.split("## ", 1)[0].strip()
    tpl_summary, tpl_bullets = split_summary(template)

    narrative = generate(
        f"--- SUMMARIES FROM {n} PARTS OF ONE MEETING ---\n{summaries}\n--- END ---\n\n"
        f"Write the summary of the whole meeting.\n\n{preamble}\n\n"
        f"## Summary\n{tpl_summary}\n\n"
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
    narrative = re.sub(r"^#+ *Summary\s*", "", narrative).strip()

    facts = ""
    if bullets:
        facts = generate(
            f"--- DECISIONS, ACTIONS AND QUESTIONS FROM {n} PARTS OF ONE MEETING ---\n"
            f"{bullets}\n--- END ---\n\n"
            f"Merge these into one list.\n\n{preamble}\n\n{tpl_bullets}\n\n"
            # Delete first, merge second. With "merge only, adding a bullet is the
            # worst error" up front the model read the whole job as preservation and
            # filtered nothing - the output was the parts' bullets verbatim.
            f"Rules. {roster}Keep every decision. Keep every action item. Deleting one "
            f"is the worst error here. Open questions are different - most of that "
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
        if "## Summary" in facts:
            facts = split_summary(facts)[1]
        # The model drops the ## about half the time; a bare "Decisions" line renders
        # as body text and the section stops being a section.
        facts = re.sub(
            r"^(Decisions|Action items|Open questions)\s*$",
            r"## \1",
            facts,
            flags=re.M,
        )

    merged = f"## Summary\n{narrative}\n\n{facts}".strip()
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
    ns = argparse.Namespace(api="ollama", host="http://h", model="m", ctx=99)
    req = request("p", ns)
    assert req.full_url == "http://h/api/generate"
    assert json.loads(req.data)["prompt"] == "p"
    assert json.loads(req.data)["options"]["num_ctx"] == 99
    ns.api = "openai"
    req = request("p", ns)
    assert req.full_url == "http://h/chat/completions"
    assert json.loads(req.data)["messages"] == [{"role": "user", "content": "p"}]
    assert json.loads(req.data)["max_tokens"] == 2048
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
    headed = re.sub(
        r"^(Decisions|Action items|Open questions)\s*$", r"## \1", "Decisions\n- x", flags=re.M
    )
    assert headed == "## Decisions\n- x"
    print("ok")


if __name__ == "__main__":
    selftest() if "--selftest" in sys.argv else main()
