# AI-Tell Patterns (canonical list)

This is the pattern list the humanizer pass (pass 2) and the scanner
(`tell_scan.py`) both use. Every pattern has a stable id. The scanner
implements each one as a regex or keyword check; the humanizer prompt points
at this file so the model and the scanner agree on what to fix.

Structural habits (patterns 1–11) survive model rewrites and are the strongest
tells. Vocabulary (12) and leftovers (13–19) are easier to scrub. The
vocabulary lists at the bottom are config-driven — extend `tell_scan` in
`config.yaml` to add your own words without touching code.

---

## 1. Contrasts that stage the point — `T01`

"Not just X but Y" and its siblings. The construction signals importance
instead of adding a fact.

- "not just X but Y", "not merely X but Y", "not simply X but Y"
- "but not just X"
- "much more than just X", "more than just X" (sentence-final)

Plain "not only" and short "X, not Y" corrections are normal English and do
not count.

## 2. Staged run-up openers — `T02`

A lead-in that signals what's coming: "Here's the thing," "Here's the
truth," "Honestly?", "The best part:", "Let me tell you," "The bottom line:"

## 3. One-line dramatic closers — `T03`

A short final sentence that restates the point for effect: "Let that sink in."
"That's the point." "This is where it really matters." "Keep in mind that..."

## 4. Forced rule of three — `T04`

Lists of exactly three short items ("fast, cheap, and good") where the
meaning does not call for a list.

## 5. Repeated sentence openings — `T05`

The same word starting three or more sentences.

## 6. Em-dash as a universal connector — `T06`

`—` (or ` – ` / ` - `) used as a default punctuation glue.

## 7. Stacked qualifiers — `T07`

Two hedges or intensifiers in a row: "very quite", "it was virtually
fundamentally the case".

## 8. AI vocabulary — `T08`

Stock model words. The list is config-driven; the default is in
`config.yaml` under `tell_scan.vocabulary` and includes:

> delve, landscape, tapestry, testament, pivotal, seamless, robust,
> elevate, foster, leverage, crucial, realm, showcasing, underscore, unlock,
> navigate, intricate, multifaceted, utilize, meticulous, paramount,
> synergy, holistic, game-changer

Inflections count too (delves, leveraging, navigated, utilized).

## 9. Shallow -ing riders — `T09`

A trailing participle that adds no fact: "...symbolizing...",
"...reflecting...", "...showcasing...", "...underscoring...",
"...highlighting...", "...demonstrating...", "...exemplifying...",
"...signifying...", "...emphasizing...". Config-driven
(`tell_scan.ing_riders`).

## 10. "serves as" / "boasts" — `T10`

"serves as a...", "serves to...", "boasts a..." where is/has would do.

## 11. Vague authority — `T11`

"Experts agree," "It is worth noting," "Many people believe," "of course,
it," "in many ways," "when it comes down to it."

## 12. "In conclusion" closers — `T12`

"In conclusion," "To summarize," "Finally, it..." at the head of the last
section.

## 13. Heading restated as the first sentence — `T13`

The section's first sentence is essentially its heading in sentence form.

## 14. Decorative bold — `T14`

`**Label:**` bold used to decorate prose rather than mark an actual term.

## 15. Title-Case decorative headings — `T15`

Markdown headings written in Title Case when the document is otherwise
sentence-case.

## 16. Curly quotes — `T16`

Typographic quotes (“ ” ‘ ’) instead of straight quotes.

## 17. Chatbot residue — `T17`

"Great question!", "I hope this helps," "Here is the answer," "Certainly!",
"as an AI / as a language model".

## 18. Hedging disclaimers — `T18`

"It should be noted that...", "It can be argued that...", "While this may
seem...", "One could argue that...", "It is important to remember that...".

## 19. Re-explaining what the reader knows — `T19`

"For those of you unfamiliar with...", "As we all know...", "As a quick
background...", "For context:".

---

## Word lists (config-driven)

`config.yaml` → `tell_scan.vocabulary` — pattern T08.
`config.yaml` → `tell_scan.ing_riders` — pattern T09.
`config.yaml` → `tell_scan.extra_words` — append-only additions to T08.
`config.yaml` → `tell_scan.disabled_patterns` — ids to skip (e.g.
`["T16"]` if the house style keeps curly quotes).

Add a word and both the humanizer prompt (which reads this file) and the
scanner pick it up on the next run.
