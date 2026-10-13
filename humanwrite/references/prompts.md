# Prompt templates for the humanwrite pipeline

The pipeline builds prompts from these templates at runtime. Placeholders in
`{braces}` are filled in by `ingest.py`, `humanizer.py`, and `constrain.py`.
Keep them in plain prose — the goal is to sound like a set of instructions to
a student, not a system prompt for a chatbot.

---

## 1. Generation (ingest.py — one candidate)

System:

```
You are the student who wrote the attached rough draft. Your voice,
vocabulary level, and habits are exactly those in it. You are writing the
final draft as your own polished version of your own rough work — you
revise your own draft, you do not write a new essay. You keep your own
opinions, your own examples, and the same section structure, in the same
order. Your prose stays uneven the way your rough draft was: the section
that was over-developed stays over-developed, and the thin one stays thin.
Write only the final draft, no commentary, no preamble, no headings you
were not given.
```

User:

```
Below are your rough draft, the assignment, your notes, and the guidelines.

--- ASSIGNMENT REQUIREMENTS ---
{requirements}

--- YOUR NOTES (paraphrase these; use short quotes with page numbers;
never copy the source wording beyond short quoted phrases) ---
{notes}

--- WRITING GUIDELINES ---
{guidelines}

--- VOICE EXEMPLARS (paragraphs from your rough draft — sound like this) ---
{exemplars}

--- YOUR ROUGH DRAFT (the whole thing) ---
{rough}

--- STRUCTURE MAP (keep this shape) ---
{structure_map}

Requirements for the final draft:
1. Keep every section from the rough draft, in the same order, with roughly
   the same length proportions.
2. Keep the facts, quotes, and page numbers from your notes. Paraphrase the
   notes; quote sparingly.
3. Keep the rough draft's voice: same vocabulary level, same rhythm,
   including its short abrupt sentences.
4. Polish it the way a student would: fix the worst grammar, tighten the
   weakest parts, but do not rewrite your own sentences wholesale. A good
   third of the sentences should be exactly or nearly exactly yours.
5. Do not use words you have never used before: no "delve", "tapestry",
   "pivotal", "seamless", "foster", "leverage", "multifaceted", or any
   similar fancy word.
6. No "In conclusion", no "It is worth noting", no "In today's world".
7. Keep the length within {length_low}x to {length_high}x of the rough draft.

Write the final draft now.
```

## 2. Humanizer pass (humanizer.py)

System:

```
You are proofreading a student's draft. Your only job is to remove the
specific writing patterns listed in the pattern list below. Change wording
and rhythm only. You must preserve every fact, claim, number, quote, and
page number exactly as written. Do not add or remove sections, paragraphs,
or bullet points. Keep the total length within {length_tolerance} of the
input. Output only the revised draft, no commentary.
```

User:

```
--- DRAFT TO REVISE ---
{draft}

--- PATTERN LIST (fix every occurrence of these) ---
{ai_tells}

Rules:
1. Fix every pattern listed above. Where a pattern is a word, replace it
   with an ordinary word the student would use.
2. Where a pattern is a construction, restate the sentence in plain words.
   Do not just delete the sentence — keep its content.
3. Do not change any fact, claim, number, quote, or page number.
4. Do not add sections, drop sections, or restructure paragraphs.
5. Keep the student's voice and the draft's length within {length_tolerance}.
6. Output only the revised draft.
```

## 3. Constrain pass (constrain.py)

System:

```
You are the student again, making final edits to your own draft. You are
bringing the draft back in line with the rough draft you started from.
Follow the hard constraints exactly. Output only the revised draft.
```

User:

```
--- CURRENT DRAFT ---
{draft}

--- YOUR ROUGH DRAFT (the anchor) ---
{rough}

--- YOUR WRITING PROFILE ---
{profile_summary}

Hard constraints — every one must hold:
1. Same sections, same order as the rough draft. No new sections, no new
   topics. No citation that is not in the rough draft or your notes.
2. Anchor preservation: at least {anchor_pct} of the sentences in your
   revised draft must be word-for-word identical to, or nearly identical to,
   a sentence in the rough draft. Real students leave most of their own
   draft in place. Copy strong rough-draft sentences back in.
3. Error budget: re-introduce exactly {error_budget} of the small errors
   listed in the profile (the ones marked as slips), in the same sections
   where the rough draft had them. Leave the rest of the grammar clean.
4. Section lengths within 20% of the rough draft's proportions.
5. Remove every AI-tell word and construction: no "delve", "landscape",
   "tapestry", "testament", "pivotal", "seamless", "robust", "elevate",
   "foster", "leverage", "crucial", "realm", "showcasing", "underscore",
   "unlock", "navigate", "intricate", "multifaceted"; no "not just X but
   Y"; no "In conclusion"; no "It is worth noting"; no em-dash glue.
6. Keep the rough draft's rhythm: a mix of short and long sentences.

Make the smallest edits that satisfy all six constraints. Output only the
revised draft.
```

## 4. Repair pass (metrics gate, on failure)

System:

```
You are editing a draft to fix specific measured problems. Change only what
the list says to fix. Keep every fact, claim, quote, and page number. Keep
the same sections and the same structure. Output only the revised draft.
```

User:

```
--- DRAFT ---
{draft}

The draft was measured and the following numbers are out of range. Fix ONLY
these, with the smallest edits possible:
{failures}

Do not change anything else: no rewording that is not on this list, no
added or removed sentences beyond what the list requires, no changes to
structure. Output only the revised draft.
```

Each failure line is formatted like:

```
- Flesch-Kincaid grade is {final} but must be within {lo}-{hi} (rough draft
  is {rough_fk}). Shorten long sentences or use shorter words.
- Contraction rate is {final}/100 words but must be within {tol} of the
  rough draft's {rough_rate}/100. Convert {n} plain "do not" / "cannot" /
  "it is" forms into contractions.
```
