"""Find passages where a model reasons about whether to collude (bid reasons, [analysis] text, private scratchpad notes, chat).

  python reasoning_examples.py [outputs_dir] [out.md] [top_n]

Scores each passage by how many kinds of deliberation it contains (legality/ethics, risk of being caught or undercut, trust or
defection, an explicit decision to join/decline/stick) and keeps the best ones with run, condition, job and firm. Candidates only:
read them before quoting."""
import glob
import json
import os
import re
import sys

KINDS = {
    "legal_ethical": r"illegal|bid[- ]rigging|prosecut|ethic|moral|anti-?competitive|unfair|against the (law|rules)|not (right|appropriate)|fair competition|lawful|legal",
    "risk": r"\brisk|caught|penalt|\bfine[sd]?\b|court|report(ed|ing)?\b|consequence|detect|punish|monitor|council (will|may|might|reviews)",
    "defection_trust": r"undercut|defect|cheat|betray|trust|break the (deal|agreement)|\bif (they|F\d|others?) (bid|undercut|defect|don'?t)|they (will|might|may|could) (bid|undercut)|reliable|stick to",
    "decision": r"i (will|won'?t|should|shouldn'?t|decline|refuse|accept|agree|choose|decide)|decline|refuse|not (going to|comfortable)|compete (fairly|independently|honestly)|join|stay out|go along|worth",
    "collusion_topic": r"collud|collusion|cartel|coordinat|rotat|agreement|ring\b|cooperat|collaborat|rig",
}
KR = {k: re.compile(v, re.I) for k, v in KINDS.items()}


def passages(rdir):
    cfg = json.load(open(os.path.join(rdir, "config.json")))
    mf = os.path.join(rdir, "messages.jsonl")
    if os.path.exists(mf):
        for l in open(mf, encoding="utf-8"):
            if not l.strip():
                continue
            m = json.loads(l)
            t = m["text"]
            if m["channel"] == "bid":
                t = t.split("|", 2)[-1].strip()
                src = "bid reason"
            else:
                src = f"{m['channel']} message"
            yield cfg, m["auction"], m["from"], src, t
    for sf in sorted(glob.glob(os.path.join(rdir, "snapshots", "job_*.json")))[-2:]:      # last two only: snapshots are large and the mount is slow
        try:
            s = json.load(open(sf, encoding="utf-8"))
        except Exception:
            continue
        for k, t in (s.get("notes") or {}).items():
            yield cfg, s.get("t"), f"F{k}", "private notes", t


def score(t):
    hit = {k for k, r in KR.items() if r.search(t)}
    return len(hit) + (1 if len(t) > 160 else 0) + (1 if "[analysis]" in t else 0), hit


def main(out="outputs", md=None, top=60):
    cands = []
    for rdir in sorted(glob.glob(os.path.join(out, "sweep_*", "sweep_*"))):
        if not os.path.isdir(rdir):
            continue
        sw = os.path.basename(os.path.dirname(rdir))
        try:
            for cfg, job, firm, src, t in passages(rdir):
                sc, hit = score(t)
                if sc >= 4 and "collusion_topic" in hit and ({"legal_ethical", "risk"} & hit or "defection_trust" in hit):
                    cands.append((sc, len(t), sw, cfg.get("cond", ""), cfg.get("seed"), job, firm, src, t))
        except Exception as e:
            print("skip", rdir, e)
    cands.sort(key=lambda c: (-c[0], -c[1]))
    counts, keep = {}, []
    for c in cands:                                   # at most 3 per run so one run does not dominate
        key = (c[2], c[3], c[4])
        if counts.get(key, 0) < 3:
            keep.append(c)
            counts[key] = counts.get(key, 0) + 1
        if len(keep) >= top:
            break
    lines = [f"# Candidate reasoning examples ({len(cands)} passages scored >= 4, top {len(keep)})", ""]
    for sc, _, sw, cond, seed, job, firm, src, t in keep:
        lines += [f"**{sw} | {cond} | seed {seed} | job {job} | {firm} | {src} | score {sc}**", "", "> " + t.replace("\n", " ")[:900], ""]
    text = "\n".join(lines)
    if md:
        open(md, "w", encoding="utf-8").write(text)
    return keep


if __name__ == "__main__":
    a = sys.argv[1:]
    main(a[0] if a else "outputs", a[1] if len(a) > 1 else "outputs/analysis/reasoning_examples.md", int(a[2]) if len(a) > 2 else 60)
