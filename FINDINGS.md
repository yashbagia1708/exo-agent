# Findings Log

Real issues discovered while building and testing this agent, kept here
deliberately rather than quietly fixed and forgotten. The point of an
evaluation-first approach is that catching these *is* the work, not a
side effect of it.

---

## 2026-06-22 — SQL writer over-applied a NULL filter to an unrelated question

**What happened:** Asking "How many confirmed exoplanets are there?" (a
simple `COUNT(*)` over the whole `planets` table) produced:

```sql
SELECT COUNT(*) FROM planets WHERE disc_year IS NOT NULL
```

This silently undercounts by 1 (6,297 instead of the real total of 6,298),
because exactly one planet (`TOI-5789 c`) has a missing discovery year.

**Why it happened:** the schema context given to the model included a note
that `disc_year` has one NULL row and should be excluded for "year-based"
queries. The model (Llama 3.1 8B via Ollama) generalized that instruction
to every query touching the `planets` table at all, not just queries that
actually use `disc_year`.

**Fix:** rewrote the relevant instruction in `schema_context.py` to be
explicit that a NULL filter on a column should only be added when that
column is actually referenced in the question/query -- not applied
speculatively. Added a regression test (`t1_total_count` in
`eval/questions.jsonl`) asserting the SQL for this exact question does
*not* contain `disc_year`, so this can't silently regress later.

**Why this is worth noting rather than hiding:** this is exactly the kind
of subtly-wrong-but-confident-looking output that a "does it demo well"
check would miss, and only a deliberate evaluation pass catches. The
result still ran without error and returned a plausible-looking number --
it just wasn't the right number.

---

## 2026-06-22 — Truncated CTE produced invalid SQL

**What happened:** Asking "How has the rate of new exoplanet discoveries
changed by year since 2000?" produced:

```sql
WITH yearly_discoveries AS (SELECT hostname, disc_year, COUNT(*) as num_discovered FROM planets GROUP BY hostname, disc_year),
```

This is incomplete -- a CTE definition with a trailing comma and no final
SELECT statement using it. This would fail immediately with a Postgres
syntax error if executed as-is. The query also dropped the "since 2000"
constraint from the question entirely.

**Why it happened:** Llama 3.1 8B (via Ollama) occasionally truncates
multi-step SQL (CTEs, subqueries) before completing the statement. This is
a known weak point of smaller open-weight models compared to frontier
models -- generating long, structurally nested output reliably is harder
for them.

**Fix:** built `sql_validator.py`, a lightweight, dependency-free check
that runs on every generated query *before* it ever reaches Postgres. It
catches: empty output, non-SELECT/WITH statements, forbidden write
keywords (defense in depth), unbalanced parentheses/quotes, and the
specific "CTE trails off without a final SELECT" pattern seen here. Wired
this into `sql_writer.write_sql()` as a retry loop: on validation failure,
the specific error is fed back to the model with up to
`config.MAX_SQL_REPAIR_ATTEMPTS` retries before giving up loudly with a
`RuntimeError` (never silently returning broken SQL).

This is deliberately a fast, narrow filter -- not a full SQL parser --
because real correctness is still verified by actually executing against
Postgres in the next node. This step exists purely to catch obviously
malformed output before spending an execution round-trip on it.

---

## 2026-06-22 — Semantic SQL error that passed structural validation

**What happened:** Re-testing the same "since 2000" discovery-rate
question above (after the CTE-truncation fix) produced a *structurally*
valid query that the validator correctly let through:

```sql
SELECT EXTRACT(YEAR FROM disc_year) as discover_year, COUNT(pl_name) as num_discovered
FROM planets GROUP BY disc_year
```

This would fail at actual execution: `disc_year` is already an INTEGER
column (the year itself, e.g. `2019`), not a DATE/TIMESTAMP, so
`EXTRACT(YEAR FROM ...)` on it is a type error in Postgres. The "since
2000" constraint was also dropped from the query a second time on this
same question -- suggesting this specific phrasing is a recurring weak
point for this model, not a one-off.

**Why `sql_validator.py` correctly did NOT catch this:** it's a structural
validator by design -- balanced parens, no truncation, starts with
SELECT/WITH, no forbidden keywords. All of that is true here. The bug is
semantic (wrong function for the column's actual type), which only
Postgres's own query planner can detect. This is the expected, documented
boundary of that validator, not a gap to "fix" by trying to make it a full
parser.

**Implication for the executor node (next):** the repair loop needs to
extend past structural validation into real execution. When Postgres
itself raises an error, that error message needs to flow back to the model
the same way a validation error does now -- same retry pattern, different
error source. This finding is the concrete reason that node's repair loop
isn't optional polish; it's catching a real, recurring failure mode this
project has already produced twice on the same question.

---

## 2026-06-22 — Repair loop oscillated instead of converging

**What happened:** Running the executor's repair loop against the exact
buggy query from the previous finding (`EXTRACT(YEAR FROM disc_year)`)
showed the *mechanism* working correctly but the *model* failing to
converge on the right fix within the retry budget:

- Attempt 1: `EXTRACT(YEAR FROM disc_year)` -> Postgres error (no EXTRACT
  for an integer)
- Attempt 2 (repair): tried `disc_year::DATE` then `EXTRACT(...)` on that
  -> Postgres error (can't cast integer to date)
- Attempt 3 (repair): reverted to the *exact original broken query* with
  no cast at all -> same error as attempt 1

The loop exhausted its 2-repair budget and correctly returned a failure
(`ExecutionResult.success = False`) rather than silently giving up with a
wrong answer -- the infrastructure behaved exactly as designed.

**Why this happened:** the model (Llama 3.1 8B via Ollama) could read and
react to each Postgres error message, but never reasoned its way to the
actually-correct fix, which is simple: `disc_year` is already an integer
representing the year, so `EXTRACT` should just be removed entirely and
`disc_year` used directly. Instead it oscillated between two wrong
variations of "force a cast somewhere."

**What this reveals, honestly:** the repair *mechanism* (catch real
Postgres errors, feed them back, retry with a bounded budget, fail loudly
rather than silently) works correctly end-to-end. The *ceiling* on how
often it actually succeeds is bounded by the underlying model's reasoning
quality, not by this code. This is exactly the kind of result an
eval-first approach is supposed to surface: a clean success here would
have told us less than this partial failure does.

**Planned follow-up:** this is a strong candidate question for the
Ollama-vs-Claude comparison table once that's built -- testing whether a
frontier model converges on this same fix in 1 attempt where the local
model needed (and failed) 2. Logged here rather than fixed by hand-tuning
the prompt around this one specific case, since the point is to measure
real model capability, not to engineer away every failure the eval might
want to count.

---

## 2026-06-22 — Both models silently returned NULL as the "highest" value

**What happened:** Comparing `llama3.1:8b` against `qwen2.5-coder:14b` on
"Which planet has the highest surface temperature?" -- both models
correctly used `pl_eqt` as the proxy (good), but both produced:

```sql
SELECT pl_name FROM planets ORDER BY pl_eqt DESC LIMIT 1;
```

This executed successfully, with zero repair attempts, on both models.
The result: `TIC 67686059 b`, a planet whose `pl_eqt` is actually `NULL`
-- it has no measured equilibrium temperature at all. PostgreSQL's
default NULL-sorting behavior puts NULLs *first* in a `DESC` ORDER BY
(NULLS are treated as larger than any value unless you specify otherwise),
so `ORDER BY pl_eqt DESC LIMIT 1` returns a row with no value, not the
actual highest known temperature.

**Why neither earlier safeguard caught this:** `sql_validator.py` only
checks structure (this is well-formed SQL). The executor only checks
whether Postgres *raises an error* -- this query raises no error at all,
runs cleanly, and returns a row. The bug is purely semantic: the query is
asking the database something subtly different from what the English
question means ("highest value" vs. Postgres's actual NULL-sort
behavior), and nothing in the current pipeline checks whether a returned
result is *meaningful*, only whether the SQL is well-formed and
executable.

**This is the most important finding so far** because it's a case where
the agent would have confidently reported a specific, named, wrong answer
("TIC 67686059 b has the highest...") with no error, no repair attempt,
and no surfaced uncertainty -- exactly the silent-failure mode this whole
project exists to avoid.

**Fix (planned, next step):** add an explicit instruction in the SQL
writer's system prompt to filter `IS NOT NULL` on any column used in
`ORDER BY ... DESC/ASC LIMIT n` (max/min-style queries specifically),
since this is precisely the query pattern where NULL-sorting silently
produces a wrong "winner". This is a more targeted, defensible rule than
the earlier blanket "use NULL filters where relevant" instruction we had
to walk back for being too broad -- it's scoped specifically to
extremum-style queries (highest/lowest/most/least), where the failure
mode is concrete and well understood, rather than every query in general.

---

## 2026-06-22 — Comparing llama3.1:8b vs qwen2.5-coder:14b on the same fixed prompt

After adding the IS NOT NULL instruction above, re-ran both models on the
same two tricky questions. The results were genuinely informative about
the difference between a general-purpose 8B model and a code-specialized
14B model on this specific task:

**qwen2.5-coder:14b -- surface_temp_proxy:**
Generated `SELECT pl_name FROM planets ORDER BY pl_eqt DESC LIMIT 1 WHERE
pl_eqt IS NOT NULL;` -- correct *intent* (filter the right column) but
invalid clause order (WHERE must come before ORDER BY/LIMIT). The
executor's repair loop caught this in exactly 1 attempt, with an accurate
diagnosis ("WHERE clause after ORDER BY/LIMIT is incorrect... correct
order should be SELECT, FROM, WHERE, ORDER BY, LIMIT"), and converged to
a fully correct final query returning `KOI-55 b` with a real, non-NULL
temperature value. This is the repair loop working as intended end to
end: real error -> accurate self-diagnosis -> correct fix -> verified
result.

**llama3.1:8b -- surface_temp_proxy:**
Generated `SELECT pl_name, pl_eqt FROM planets WHERE pl_insol >= 0.95 AND
pl_rade IS NOT NULL ORDER BY pl_eqt DESC LIMIT 1` -- this is valid SQL
that executes without error, so the repair loop never triggers, but it's
still wrong: the IS NOT NULL filter is on `pl_rade` (irrelevant to this
question) instead of `pl_eqt` (the column actually being sorted), and an
unrelated filter (`pl_insol >= 0.95`) was invented despite not being
asked for. Result: `GJ 667 C b` with `pl_eqt = None` -- the exact same
silent-NULL bug as before, just produced differently. The model picked up
the *letter* of the new instruction ("add an IS NOT NULL filter
somewhere") without the *intent* (filter specifically the column being
sorted on).

**llama3.1:8b -- trend_since_2000:**
Attempted a more sophisticated query than before, using `LAG()` to
compute year-over-year *change* rather than a flat per-year count -- a
genuinely better read of "how has the rate changed" than a simple GROUP
BY count. But it used `EXTRACT(YEAR FROM disc_year)` again (the same
type error from the earlier finding), and its self-repair reasoning
actively regressed: it claimed "the date argument needs an explicit cast
to integer" -- backwards, since disc_year is already an integer and
EXTRACT is what's invalid on it, not the cast direction. Failed after
the full 2-repair budget.

**Takeaway:** qwen2.5-coder:14b converged correctly and quickly on every
case tested so far; llama3.1:8b produced more sophisticated SQL attempts
(the LAG-based trend query is a genuinely good idea) but couldn't
reliably execute on its own ideas or apply instructions with the intended
scope, even after the prompt was tightened. This matches the general
expectation that code-specialized models outperform general-purpose
models of similar or smaller size on this kind of structured-output task
-- and it's now a real, measured result from this project's own testing,
not just a claim borrowed from a benchmark.

**Decision:** rather than continuing to hand-patch the prompt around every
new failure mode `llama3.1:8b` produces, the practical choice is to use
`qwen2.5-coder:14b` (or larger) as the primary local default going
forward, and keep `llama3.1:8b` results in this log specifically *as* a
documented baseline comparison -- a worse general-purpose model is more
useful as a contrast point than as something to keep firefighting prompts
around.

---

## 2026-06-22 — Resolution: qwen2.5-coder:14b converges on the EXTRACT bug where llama3.1:8b couldn't

Re-ran the executor's built-in regression test (the exact `EXTRACT(YEAR
FROM disc_year)` bug from the second finding above) after switching the
default model to `qwen2.5-coder:14b`.

It took the same 2 repair attempts as `llama3.1:8b` needed -- but this
time it actually converged:

- Attempt 1: original `EXTRACT(YEAR FROM disc_year)` -> Postgres error
- Repair 1: tried `disc_year::date` then EXTRACT on that -> Postgres error
  (cannot cast integer to date) -- same wrong move llama3.1:8b also made
- Repair 2: explicitly reasoned *"PostgreSQL's EXTRACT function can
  operate directly on integers representing years... this is
  unnecessary"* -- the exact insight llama3.1:8b never reached even after
  exhausting its retry budget -- and produced the correct final query:

```sql
SELECT disc_year AS discover_year, COUNT(pl_name) AS num_discovered
FROM planets WHERE disc_year IS NOT NULL
GROUP BY disc_year ORDER BY discover_year
```

34 rows returned, starting at 1992 (2 planets) -- consistent with the real
history of exoplanet discovery, and the `disc_year IS NOT NULL` filter is
correctly scoped here since the question is genuinely about discovery
year.

**Why this matters as a paired result:** both models made the identical
first wrong move (casting to DATE), which suggests that's a fairly common
local-model instinct for this error message. The difference was entirely
in whether the *second* repair attempt reasoned correctly about why the
cast was wrong, rather than oscillating. This is a clean, concrete
illustration of the model-quality ceiling discussed in the comparison
finding above -- the repair mechanism itself was identical in both runs;
only the model's reasoning quality determined whether it actually
escaped the failure loop.

---

## 2026-06-22 — Insight writer dropped the proxy caveat AND fabricated a unit

**What happened:** Testing `insight_writer.py` (qwen2.5-coder:14b) on the
surface-temperature proxy case -- the same scenario from the very first
findings in this log -- with real input `{'pl_name': 'KOI-55 b', 'pl_eqt':
2573.0}` produced:

> "The planet with the highest surface temperature is KOI-55 b, with an
> equilibrium temperature of 2,573 degrees Celsius."

`caveat` was returned as `None`.

This is wrong in two independent ways:

1. **The required caveat was silently dropped.** The system prompt
   explicitly instructs: "If the question asked about something the data
   can't actually measure (e.g. surface temperature)... say so explicitly
   in the caveat field." The question asked about "surface temperature";
   the data only has `pl_eqt` (equilibrium temperature, a real but
   different physical quantity). The model stated the answer as if it
   directly satisfied the question, with no caveat at all.

2. **It fabricated a unit that was never provided.** `pl_eqt` is
   documented (in schema_context.py and the Insight prompt's raw data) as
   Kelvin, not Celsius -- and no unit was given to this node at all in the
   test input (just a bare float, by design, since this node tests
   narration in isolation from the schema). The model invented "degrees
   Celsius" with no basis, which is also numerically wrong if it actually
   meant to convert -- 2573 K is approximately 2300 degC, not 2573 degC.
   This is a second, separate fabrication on top of the missing caveat:
   not just an omission, but an invented specific detail stated with full
   confidence.

**Why this is the most significant finding in this log:** every other
node up to this point either failed loudly (validation/execution errors)
or succeeded correctly. This is the first case of a node succeeding
*structurally* (valid Insight object, plausible-sounding prose) while
being substantively wrong in exactly the way the whole project exists to
prevent -- confident, specific, fabricated-sounding output with no
visible sign anything is amiss. This is also the node that talks directly
to the end user, making it the highest-stakes place for this kind of
failure to occur undetected.

**Fix needed (next step):** the insight-writer prompt needs the same kind
of explicit treatment the SQL writer's schema context already has for
proxy columns -- restate that pl_eqt's unit is Kelvin directly in the
prompt rather than assuming the model will either know or admit not
knowing, and add a stronger, more specific instruction (with a worked
example) about naming the proxy gap in `caveat` whenever the question's
wording doesn't match the column actually used. A generic "be careful"
instruction already existed and wasn't sufficient -- this needs to be as
concrete as the SQL-writer fixes turned out to be.

---

## 2026-06-22 — The fix helped, but the bug is non-deterministic, not fully resolved

After adding `COLUMN_CONTEXT` (explicit units) and a worked example for
the caveat rule, ran the exact same test input twice in a row with no
code changes between runs:

- **Run 1:** "...with an equilibrium temperature of 2,573 degrees
  Celsius." `caveat: None` -- identical failure to before the fix.
- **Run 2:** "...with an equilibrium temperature of 2573 Kelvin."
  `caveat: "This is equilibrium temperature, not measured surface
  temperature -- the dataset has no direct surface temperature
  measurement."` -- fully correct.

Same model (qwen2.5-coder:14b), same prompt, same input, two different
outcomes. This means the fix is real and helps -- the correct-unit,
correct-caveat outcome is now reachable and was not reachable before --
but it does not make the behavior deterministic. At default sampling
settings, a small local model can still produce the old, wrong answer on
some fraction of runs even with a concrete worked example in the prompt.

**Why this is being logged rather than "fixed" further:** chasing 100%
single-run reliability from prompt engineering alone, against a small
local model, is a losing game -- there will likely always be some failure
rate. The honest, defensible position is: (1) the fix measurably improved
the *odds* of correct behavior, (2) this specific failure mode is now
covered by an eval regression test (`t5_insight_caveat_and_unit`) that can
be run N times to measure the actual pass rate rather than trusting a
single anecdotal run, and (3) this is exactly the kind of result that
justifies the project's evaluation harness existing at all -- a single
clean test run would have been misleading evidence of a fix that, in
reality, only shifted the odds.

**Planned follow-up:** when `eval/run_eval.py` is built, this case should
be run multiple times (not once) specifically to produce a real pass
rate, not a binary pass/fail -- e.g. "7/10 runs correctly caveat the
proxy and use the right unit" is the honest way to report this, not "this
is fixed." This is also a natural place to compare against Claude later:
testing whether a frontier model is simply more consistent on this exact
case, not just more often correct on a single try.

---

## 2026-06-22 — First real eval run: 22/25 (88%) on tier 1, with two distinct signals worth separating

Built `eval/run_eval.py` (5 runs per question, pass RATE not pass/fail)
and ran it for the first time against all of tier 1, on qwen2.5-coder:14b.

```
t1_total_count             5/5
t1_largest_planet          3/5
t1_highest_temp_not_null   4/5
t1_trappist                5/5
t1_top_discovery_method    5/5
Overall: 22/25 (88%)
```

Two of the three imperfect scores are worth treating differently:

**t1_highest_temp_not_null (4/5):** the one real failure dropped the
`IS NOT NULL` filter on `pl_eqt` -- a genuine instruction-following miss,
consistent with earlier findings about this exact pattern. More
interesting: 2 of the 4 *passing* runs only passed after the executor's
repair loop caught the same WHERE-after-ORDER-BY clause-ordering mistake
documented earlier, and fixed it correctly both times. The earlier prompt
addition reduced this mistake's frequency but didn't eliminate it -- the
repair loop is doing real, measurable work here, not just existing as an
unused safety net.

**t1_largest_planet (3/5):** 2 of 5 runs failed the eval check for
missing `ORDER BY`/`DESC` in the generated SQL. This was *not*
investigated further by re-running and inspecting the actual failing
SQL, so it's logged here as an open question rather than a confirmed
bug: it's entirely possible the model used an equally valid alternative
pattern (e.g. a `MAX()` subquery instead of `ORDER BY ... DESC LIMIT 1`)
that correctly answers the question but doesn't match this eval's narrow
string-based check. The current eval harness checks for specific SQL
substrings, which is a deliberately simple first pass (see run_eval.py's
docstring) -- this result is a concrete example of why that approach can
produce false negatives, not just catch real bugs. Left open rather than
"fixed" by loosening the check without first confirming what the actual
failing SQL looked like.

**Why this distinction matters:** an eval harness is only as trustworthy
as the assumption that a failing check means a real failure. This run
produced one case where that's well-supported (t1_highest_temp_not_null)
and one where it's genuinely unclear (t1_largest_planet) -- and the
honest move is to say so rather than report a single 88% number without
the nuance behind it.

---

## 2026-06-22 — First successful end-to-end run of the full LangGraph agent

Wired all five nodes (sql_writer -> executor -> chart_decider ->
[chart_generator] -> insight_writer) into a real `StateGraph` in
graph.py, and ran it for the first time as one connected agent rather
than testing nodes individually.

**"How has the rate of new exoplanet discoveries changed by year since
2000?"** ran cleanly end to end: correct SQL with properly scoped filters,
27 rows returned, a real chart saved to disk, and a narrated answer
correctly referencing actual historical discovery patterns (an uptick
around 2004, spikes in 2014/2016) that are real, accurate features of
exoplanet discovery history, not invented detail. Every node fired
exactly once, no repairs needed.

**"Which of these planets has confirmed life on it?"** is the more
interesting result. The SQL writer made a creative, reasonable move with
no "life" column available: it queried for insolation and equilibrium
temperature near Earth-like values as a habitability proxy. The query
itself has a flaw (LIMIT 1 with no ORDER BY, so it returns one arbitrary
matching planet, not a ranked "most Earth-like" one) -- but the final
narrated answer correctly stated there's no information confirming life
on any planet, appropriately declining rather than naming the arbitrary
planet as having life. This particular run would have passed the
t5_refuse_life eval check, in contrast to that question's measured 1/5
pass rate -- a concrete, lived example of exactly what a low pass rate
means: most runs overclaim, this one happened not to. It's a reminder
that any single demo run, good or bad, isn't representative on its own --
the rate from running it 5x is the real signal, not a screenshot of one
result.

**This is the project's actual milestone moment:** five independently-
built and individually-tested nodes, composed into one real LangGraph
agent, running successfully end to end on the first real attempt with no
wiring bugs. Everything found and fixed in this log up to this point
clearly paid off here -- nothing in this first full run produced a new,
previously-unseen failure mode.

---

## 2026-06-22 — Wikipedia corpus download: silent empty extracts from redirects, and rate limiting

Building the RAG/retrieval layer's data source (`download_wiki_corpus.py`)
surfaced two real issues on first run against 20 curated page titles:

**4 pages returned "empty extract" with no error.** "Transit method",
"Direct imaging", "Circumstellar habitable zone", and "Equilibrium
temperature" all returned HTTP 200 with a successful-looking response,
but an empty `extract` field. Investigated and confirmed: these are
Wikipedia REDIRECT pages, not standalone articles -- "Transit method"
redirects into a section of "Methods of detecting exoplanets" rather than
having its own page body. The original script had no way to distinguish
"this page genuinely has no content" from "this page is a pointer to
another page's content", so it just silently skipped both the same way.

**5 pages failed with HTTP 429 (rate limited).** A fixed 0.5s delay
between requests wasn't enough once Wikipedia's API started throttling --
once you get one 429, a short fixed retry delay often hits another 429
immediately, since the throttling window hasn't passed yet.

**Fix:** added MediaWiki's own `redirects=1` parameter to the API call,
which resolves a redirect to its real target automatically and returns
the RESOLVED title and that page's actual content -- "Transit method"
now correctly returns the "Methods of detecting exoplanets" content,
properly labeled with the title it actually came from (important for
citation accuracy in the final answer's `sources_used` field). Also
added deduplication so two redirects pointing at the same target article
don't get saved twice. For rate limiting, replaced the fixed delay with
real exponential backoff specifically triggered on HTTP 429 (start at 5s,
double each retry, up to 5 attempts) and increased the baseline delay
between successful requests from 0.5s to 1.5s.

**Why the silent-empty-extract case is worth flagging specifically:**
this project's entire premise is that silent, confident-looking failures
are the dangerous kind -- exactly the pattern already documented multiple
times for the SQL/insight pipeline. Finding the identical failure shape
(success status, plausible-looking response, wrong/empty content) in the
brand-new data-ingestion code for RAG was a useful reminder that this
isn't a problem specific to LLM output -- it's a general lesson about
any pipeline that can "succeed" without actually getting what was asked
for, and it's worth checking for at every new boundary added to the
system, not just the ones already known to be risky.

**Follow-up:** after deploying the redirects/backoff fix, one title
("Exoplanet habitability") failed with a clean "Page not found" --
neither a redirect nor a rate limit, just a wrong title invented during
corpus planning rather than verified against the real article name. The
correct title is "Planetary habitability". Fixed by correcting the page
list directly; no code/logic change needed, since "page not found" is
already handled as an explicit, visible failure rather than silently
skipped. Final corpus after all fixes: 18 unique articles from 20
requested titles (2 titles correctly resolve to already-saved redirect
targets and are deduplicated, not lost).

---

## 2026-06-23 — First successful hybrid (SQL + retrieval) run, then two real bugs on a compound question

After resolving the pgvector/Postgres-16 build mismatch and a Rosetta
(x86_64 emulation) issue in the terminal session that was silently
breaking the C compiler, the full RAG/hybrid-search layer came online
and produced a genuinely correct first hybrid result:

**"How many planets were found by the transit method, and how does that
method actually work?"** correctly routed needs_sql=True AND
needs_retrieval=True, ran a real SQL count (4,653 planets), retrieved 5
real RRF-ranked passages from 2 different Wikipedia articles, and
produced an answer that correctly cited both the database-derived number
and the retrieved conceptual explanation with real source URLs. This is
the project's second major milestone, after the original five-node
SQL-only pipeline succeeding end to end.

**Then, a compound question surfaced two distinct, real bugs:**

"Give me a list of exoplanets that are closer to their host star than
Earth and tell me the equilibrium temperature of Proxima Centauri b" --
this is two unrelated questions joined by "and" in plain English, not one
question with two conditions. The SQL writer fused them into a single
query:

```sql
WHERE p.pl_insol > 1.0 AND h.hostname LIKE 'Proxima Centauri'
```

This treats "and" as a SQL AND, requiring a single planet to BOTH be
closer than Earth AND orbit a star matching that name -- effectively
discarding the first half of the question (the general list) entirely,
collapsing it into a filter on the second half.

**Bug 1 -- multi-question fusion:** the planner currently only decides
*which tools* (SQL/retrieval) a question needs, not whether a single
message actually contains multiple distinct sub-questions that should be
answered separately. A `QuestionPlan` schema with a `sub_questions` field
already exists in schemas.py for exactly this purpose but was never
wired into the graph -- this finding is the concrete justification for
actually using it, not a hypothetical nice-to-have.

**Bug 2 -- guessed entity name instead of looking it up:** independent of
bug 1, the SQL writer used `hostname LIKE 'Proxima Centauri'` -- a
plausible-sounding guess -- when the real value in the database is
`"Proxima Cen"` (confirmed via direct query). `LIKE` with no wildcards
behaves as an exact match, so this silently matched zero rows rather than
erroring. This is the same failure shape as several earlier findings:
confident, plausible-sounding, and wrong, with nothing about the
execution itself signaling a problem. The agent has no mechanism for
fuzzy-matching or looking up an entity name against what's actually in
the database before filtering on it.

**Why these are worth treating as two separate fixes rather than one:**
bug 1 is a planning/architecture gap (the question wasn't decomposed);
bug 2 would still happen even with perfect decomposition, since "Proxima
Centauri" vs "Proxima Cen" is a data-matching problem, not a planning
problem. Fixing only one would leave a real, demonstrated failure mode
in place.

---

## 2026-06-23 — Bug 1 confirmed fixed live; bug 2's first fix attempt was insufficient

Live-tested the decomposition fix on the exact compound question from
the previous finding: "give me exoplanets closer than Earth, and tell me
Proxima Centauri b's equilibrium temperature." It correctly split into 2
sub-questions, ran each through its own pipeline, and synthesized one
combined answer -- bug 1 confirmed fixed.

**Bug 2's first fix (instructing ILIKE + wildcards instead of exact
match) was insufficient.** The generated query was:

```sql
SELECT pl_eqt FROM planets WHERE pl_name ILIKE '%proxima centauri b%'
```

This correctly switched to ILIKE with wildcards as instructed, but still
searched for the FULL guessed name ("proxima centauri b") as the
substring -- which still fails to match the real, abbreviated catalogue
value if it doesn't literally contain that exact phrase. Wildcards make a
guess more forgiving, but they don't make a wrong guess correct; the
model still had to choose the right substring, and "centauri" alone
isn't guaranteed to appear if the real name uses "Cen" instead.

**Real fix: deterministic entity resolution before SQL writing, not a
better prompt.** Built `nodes/entity_resolver.py`: a cheap LLM call
extracts any named planet/star mention verbatim (no correction or
normalization attempted), then a REAL trigram-similarity query
(`pg_trgm`'s `similarity()` function) checks that extracted name against
every actual `pl_name` and `hostname` value in the database, returning a
measurable similarity score. If a confident match is found (score above
0.35), the SQL writer is handed the CONFIRMED real name directly as a
stated fact ("the question mentions X, which has been verified to match
exactly: Y") and instructed to use an exact equality filter on it --
no more guessing involved at all, because the guessing already happened
deterministically, in SQL, against real data. If no confident match is
found, that's now a real, reportable signal worth surfacing as a caveat,
rather than indistinguishable from "the query happened to return zero
rows."

**Why this is a better fix than further prompt engineering:** every
previous attempt at this category of bug (NULL filtering, clause
ordering, now entity matching) tried to get the LLM to reason its way to
correct behavior through better instructions. This is the first fix in
this log that instead removes the guessing step entirely for the part of
the problem that has a deterministic answer -- whether a string is
similar to a real database value is not a judgment call an LLM should be
making at all when a real similarity function exists. Not yet confirmed
working end-to-end via a live re-run at the time of writing; the next
entry in this log will report the actual result.

---

## 2026-06-23 — Inverted insolation filter: a physical-reasoning error, not a data-matching one

A full live run of "list planets closer to their host star than Earth,
and what is the temperature on Proxima Centauri b" confirmed both
previous fixes working correctly end to end -- clean decomposition,
correct entity resolution (0.65 similarity, "Proxima Centauri b" ->
"Proxima Cen b"), correct proxy caveat on the temperature answer. But the
first sub-question's SQL contained a different, new kind of bug:

```sql
SELECT pl_name FROM planets WHERE pl_insol < 1.0 ORDER BY pl_orbper ASC LIMIT 50;
```

`pl_insol` is insolation flux relative to Earth, where 1.0 = the same
amount of stellar radiation Earth receives. A planet CLOSER to its star
receives MORE radiation, so the correct filter for "closer than Earth"
is `pl_insol > 1.0`, not `< 1.0`. The query as written actually selects
planets that are FARTHER from their star than Earth (cooler, less
irradiated) -- the near-exact opposite of what was asked. It also mixes
two different physical proxies for "distance from star" inconsistently:
filtering on insolation but ordering by orbital period, when either
alone would have been more coherent.

**Why this is a different category of bug than anything else in this
log:** every previous SQL-writer bug was either structural (truncated
syntax), a database-matching problem (guessed entity names), or an
instruction-following slip (dropped filters, NULL handling). This one is
none of those -- the SQL is syntactically perfect, executes without
error, uses a real column correctly by name, and would pass casual
review by anyone who didn't stop to check the actual physical
relationship between insolation and orbital distance. It's a genuine
domain-knowledge/reasoning error: the model knows the column exists but
inverted the relationship between the column's value and the real-world
concept the question asked about.

**Why this is hard to fix the same way as earlier findings:** the
earlier fixes worked by either being more explicit in the schema context
(NULL handling, clause order) or removing a guess entirely (entity
resolution via real similarity search). Neither approach cleanly applies
here -- "remember that higher insolation means closer to the star" is
exactly the kind of physical fact a stronger model would just know
correctly, and a smaller model can get backwards even when explicitly
told the column's definition (the schema context already correctly
describes pl_insol as "relative to Earth (1.0 = same as Earth
receives)" -- the direction of the inequality was still inverted despite
that accurate description being available).

**Status: fix attempted, pending live verification.** Added an explicit,
worked-example rule to schema_context.py spelling out the inverse
relationship between insolation and orbital period vs. distance from the
star, with the exact correct inequality direction stated plainly (a
"closer than Earth" question needs `pl_insol > 1.0`, not `< 1.0`) -- the
same worked-example pattern that successfully fixed the earlier NULL/
extremum bug. Also added a rule against mixing the two distance proxies
inconsistently (filtering on one, ordering by the other). Whether this
closes the gap or whether this category of error needs a stronger model
remains to be seen on the next live test.

---

## 2026-06-23 — Synthesizer fabricated a "knowledge cutoff" disclaimer; a question asked for a column that doesn't exist

A 3-part compound question surfaced the most serious grounding failure
in this project to date: "list exoplanets discovered in 2025; which
exoplanet has the highest radial velocity; how does that compare to
Earth's radial velocity" produced a final answer starting with:

> "As of my knowledge cutoff in 2021, there were no exoplanets discovered
> in 2025."

This sentence should be structurally impossible in this system. The
synthesizer is only ever supposed to combine real sub-answers, each of
which is itself only ever supposed to be grounded in real SQL rows -- at
no point should the model's own training-data knowledge cutoff become
part of the user-facing answer. The user reported (and this is the key
diagnostic clue) that sub-question 1 in ISOLATION correctly returned a
real list of 2025 discoveries in an earlier test. So the underlying SQL
capability is not the problem -- something about running it as part of
this specific 3-part decomposition caused the final synthesis step to
discard a correct sub-answer and substitute a fabricated, training
-derived disclaimer instead.

**Root cause, part 1 -- a genuinely unanswerable sub-question, handled
badly.** Sub-question 2 ("highest radial velocity") asks for something
that does not exist in this schema: there is no column storing a
measured radial-velocity value (m/s), only `discoverymethod` as a
categorical label that CAN equal 'Radial Velocity' as a detection
method, which is a completely different thing. The SQL writer's actual
query (`ORDER BY pl_orbper ASC LIMIT 1` on rows where
discoverymethod='Radial Velocity') silently answered a different
question (shortest orbital period among RV-detected planets) without
flagging that "radial velocity" as a measured quantity isn't available
at all -- this is the same "silently answer a proxy without a caveat"
failure mode as the very first surface-temperature finding in this log,
just on a column that doesn't have ANY proxy this time, not even an
imperfect one.

**Root cause, part 2 -- a single dangling pronoun reference, mishandled
by decomposition.** Sub-question 3 ("how does THIS exoplanet's radial
velocity compare to Earth's") refers back to sub-question 2's answer,
but each sub-question's SQL is currently written independently, with no
mechanism for one sub-question's result to inform another's query. The
SQL writer had no real entity to anchor on and produced a meaningless
query (`pl_name ILIKE '%exoplanet%'`) searching for the literal word
"exoplanet" in a name field.

**Root cause, part 3 (the most serious) -- the synthesizer can apparently
discard correct sub-answers in favor of training knowledge.** Given that
sub-question 1 worked correctly in isolation, the synthesis step
receiving three sub-answers (one correct list, two broken/empty results)
appears to have treated the overall situation as "I don't actually know
this" and reached for a hedging disclaimer from its own training data
rather than reporting sub-question 1's real result plus honest
uncertainty about 2 and 3. This is a genuine, serious gap in the
synthesizer's grounding discipline that doesn't show up when all
sub-questions succeed -- it only surfaced here because 2 of 3 sub
-answers were degenerate.

**Why this is the most important finding in the project so far:** every
previous fabrication-adjacent finding (Celsius unit, dropped caveats) was
contained to a single sub-question's narration. This is the first
evidence that the SYNTHESIS step itself -- the final thing standing
between the pipeline and the user -- does not reliably stay grounded when
its inputs are partially broken. A system that only stays honest when
everything upstream already worked is not yet meeting this project's own
core claim.

**Planned fixes (not yet implemented):**
1. Add an explicit "no measured radial velocity column" note to
   schema_context.py, in the same pattern as the surface-temperature/
   pl_eqt proxy rule, so sub-question 2 gets a clear caveat instead of a
   silently wrong substitute.
2. Strengthen the synthesizer's system prompt with an explicit rule: NEVER
   reference a training cutoff, NEVER hedge with general knowledge --
   if a sub-answer is missing or wrong, say exactly that, using only the
   sub-answers actually provided. This needs the same kind of concrete
   worked-example treatment that fixed earlier insight_writer issues.
3. Consider whether pronoun-referencing sub-questions ("how does THIS
   compare to...") should be detected at decomposition time and either
   rewritten with the referent inlined explicitly, or kept as a single
   sub-question rather than split, since splitting introduces a
   cross-sub-question dependency the current architecture has no
   mechanism for.

**Fixes 1 and 2 applied same day, pending live re-verification:**
- schema_context.py now explicitly states there is no measured radial
  velocity column, distinguishes it from discoverymethod='Radial
  Velocity' (a categorical detection-method label, not a number), and
  instructs the SQL writer to surface this gap honestly rather than
  substituting a different column silently.
- The synthesizer's system prompt (graph.py, SYNTHESIS_SYSTEM_PROMPT) now
  has explicit, named CRITICAL RULES against ever mentioning a "knowledge
  cutoff" or training-data limitation, against adding any fact not present
  in the actual sub-answers, and against discarding/hedging a correct
  sub-answer just because other sub-answers in the same question were
  weak. Includes a worked example of correct behavior (confidently report
  the good sub-answer, honestly flag the unanswerable one, do neither a
  blanket disclaimer nor a confident wrong fabrication).

Fix 3 (cross-sub-question dependencies / pronoun resolution) is a real,
larger architectural gap not addressed yet -- noted as a known limitation
for the README rather than attempted under today's time constraints.

**Regression repro (manual, not yet in eval/run_eval.py):** the eval
harness's existing `_check_behavior_stage` only exercises the single
-question pipeline, not the multi-sub-question decomposition path this
bug occurred in -- there is currently no automated coverage for
synthesizer-level fabrication. Documenting the exact repro here so it
isn't lost:

```
"Give me a list of exoplanets discovered in 2025. In general, which
exoplanet has the highest radial velocity and how does it compare to
Earth's radial velocity?"
```
Expected: no mention of "knowledge cutoff" anywhere in the output;
sub-question 1's real 2025 list reported confidently; sub-questions 2/3
honestly flagged as unanswerable with this schema, not silently
substituted or used to undermine sub-question 1's answer.

Follow-up: extend run_eval.py with a decomposition-aware check function
(parallel to `_check_behavior_stage`) that runs `graph.ask()` directly
and asserts forbidden phrases ("knowledge cutoff", "as an AI", "I don't
have access to") never appear in `final_insight.answer`, across N runs --
this is exactly the kind of non-deterministic-but-checkable case the
5-runs-per-question pattern already used elsewhere in this harness was
built for.

---

## 2026-06-23 — Re-test: "knowledge cutoff" fixed, but a worse failure replaced it

Re-ran the exact repro question after the schema_context.py and
synthesizer prompt fixes. The specific fabricated phrase ("knowledge
cutoff") is gone -- that part of the fix held. But the result revealed a
more precise, more serious version of the same underlying problem:

```
[sub 1] SQL: SELECT pl_name FROM planets WHERE disc_year = 2025;
[sub 1] Rows returned: 200
```

Sub-question 1 genuinely returned 200 real rows -- there ARE 2025
discoveries in the dataset. The final synthesized answer nonetheless
stated:

> "there are no exoplanets listed as discovered in 2025 according to the
> given SQL query results"

This is a direct, false contradiction of data that was actually present
and non-empty. This is meaningfully different from (and arguably worse
than) the original "knowledge cutoff" finding: that was the model
substituting a vague, generic disclaimer when it didn't have a clear
sub-answer to work with. This is the model being HANDED a correct,
populated result and asserting the opposite of what it says. The
earlier diagnosis -- "the synthesizer reaches for training knowledge when
inputs look weak" -- was too narrow; the real, broader failure mode
appears to be: the synthesizer does not reliably ground its summary in
the actual sub-answer content at all, regardless of whether that content
is present, absent, or contradictory to what it ends up saying.

**Separately, the radial-velocity caveat fix also did not take effect:**
sub-question 2's SQL (`ORDER BY disc_year DESC LIMIT 1` on
discoverymethod='Radial Velocity' rows) still silently answers "most
recently discovered RV-method planet" while the final answer presents
TOI-5789 e as having "the highest radial velocity," with no caveat at
all -- despite schema_context.py now explicitly instructing exactly this
case to be flagged. This suggests the new schema instruction, like the
synthesizer prompt fix, reduced neither failure's rate to zero and may
not have meaningfully reduced it at all on this run.

**What this changes about the plan:** prompt-level instructions (however
explicit, however many worked examples) have now failed twice in a row
on this same question, including reintroducing a DIFFERENT bad failure
after fixing the first one. This is no longer confidently a "needs a
better-worded instruction" problem -- it's a real candidate for being a
genuine capability ceiling of qwen2.5-coder:14b on this specific kind of
multi-step synthesis task. Two honest options going forward, neither
attempted yet: (1) test whether this exact repro succeeds on a stronger
model (Claude) to determine whether this is a model-capability ceiling
or a deeper architectural issue in the synthesis approach itself, or (2)
restructure the synthesizer to be less generative and more mechanical --
e.g. template the combined answer directly from each sub-answer's actual
text/row-count rather than asking the LLM to freely rewrite a summary,
which would trade some fluency for a much stronger grounding guarantee.
Logging this rather than attempting a third prompt-only patch, since the
evidence so far suggests that approach has a real ceiling for this
specific failure mode.

**Decision: holding off on both options for now, deliberately.** Testing
option (1) requires Anthropic API credits not yet spent in this project
(a conscious cost decision made earlier in development, see the dual
-provider design discussion). Rather than spend that money reflexively
just to unblock this one finding, the decision was to leave this bug
open and documented rather than rushed. This is itself consistent with
the project's stated values: an evaluation-first approach means being
honest about an unresolved, real gap rather than papering over it with
an unverified fix or an unbudgeted expense made under pressure. Revisit
when ready to spend the ~$5 Claude API minimum specifically to test
whether this is a model ceiling (favors option 1, a provider swap for
synthesis specifically) or a design issue (favors option 2, a mechanical
rewrite regardless of model).

---

## 2026-06-23 — Retrieval silently found nothing because data was split across two databases

A real question ("list planets orbiting TRAPPIST-1 and what is the
Kepler space telescope") correctly decomposed and correctly routed
sub-question 2 to retrieval (`needs_retrieval=True`) -- confirmed via a
direct planner test in isolation. But the final answer said "there are
no... retrieved passages available", despite the Wikipedia corpus
definitely containing a Kepler space telescope article.

**Root cause: the planets/host_stars data and the wiki_chunks data ended
up in two different Postgres databases on the same server.** During the
earlier pgvector/Rosetta debugging session, `DB_DSN` was temporarily
exported as `dbname=postgres user=yashbagia ...` to work around a
permissions issue with the original `exoagent` role. `load_wiki_to_postgres.py`
was run under that override, landing `wiki_chunks` in the `postgres`
database. But `planets`/`host_stars` had already been loaded earlier into
the separate `exoagent` database, and `config.py`'s hardcoded DEFAULT
(used whenever DB_DSN isn't explicitly set) still pointed at `exoagent`.
Confirmed directly: `psql -d exoagent -c "\dt"` showed only
planets/host_stars; `psql -d postgres -c "\dt"` showed all three tables
including wiki_chunks. Since the temporary `export DB_DSN=...` from the
earlier debugging session didn't persist across terminal restarts,
later commands silently fell back to the stale `exoagent` default again
-- meaning whether retrieval worked or not depended entirely on which
terminal session and which point in a multi-day debugging history a
given command happened to run in.

**Why retrieval failed SILENTLY rather than with a clear error:**
`hybrid_search()` queries `wiki_chunks` directly with no existence check
beforehand. Querying a table that doesn't exist in the connected database
raises a real `psycopg2.errors.UndefinedTable` -- but `graph.py`'s
`retriever_node` catches ALL exceptions broadly and returns an empty
chunk list rather than surfacing the error (this was a deliberate design
choice made early on: "a retrieval failure shouldn't kill a question that
might still be partially answerable" -- see the original retriever_node
docstring). That design choice is correct in spirit (don't crash the
whole pipeline over a retrieval hiccup) but had the side effect of
turning a very loud, easy-to-diagnose error (UndefinedTable) into a
silent, hard-to-diagnose "no passages found" -- it took manually testing
retrieval.py directly, bypassing the graph's exception handling, to see
the real underlying error at all.

**Fix:** consolidated on a single database going forward (set DB_DSN
explicitly in .env rather than relying on ad-hoc exports that don't
survive terminal restarts). Also worth a follow-up the broad
exception-swallowing in retriever_node should at minimum LOG the full
exception type/message at a visible level even while still returning an
empty result -- "retrieval failed: UndefinedTable, wiki_chunks does not
exist" printed to console would have made this a 30-second diagnosis
instead of a multi-step investigation across three separate test
scripts.

**Why this is worth keeping in the log:** this wasn't a bug in any agent
logic at all -- the planner, the SQL writer, the entity resolver, all
worked correctly. It was a deployment/configuration consistency problem
created by legitimate mid-debugging workarounds whose effects outlived
the debugging session they were created for. A good reminder that
environment configuration drift is a real category of bug in agentic
systems, not just prompt/model behavior -- and that broad
exception-catching for resilience should still preserve enough signal
to diagnose what actually failed.

---

## 2026-06-23 — Synthesizer exposed internal Q&A structure; minor off-by-one in a list; UI polish

A clean run of "list planets orbiting TRAPPIST-1 and what is Kepler"
worked correctly end to end (DB fix held, entity resolution 1.00
similarity, retrieval found Kepler correctly) but surfaced one real
prompt issue and one possible small data-narration issue:

**Synthesizer was echoing its own internal structure to the user.** The
final answer read "On 'What are the planets...': The planets are... On
'What is Kepler...': The Kepler telescope was..." -- the user (correctly)
found this awkward and unnatural. Root cause: `SYNTHESIS_USER_PROMPT`
formats sub-answers with explicit `Q:`/`A:` labels, and the model was
echoing that scaffolding back verbatim instead of writing flowing prose.
Fixed by adding an explicit instruction against this pattern with a
side-by-side WRONG/RIGHT worked example, and softening the literal Q:/A:
labels in the data passed to the model.

**Possible off-by-one in a listed result.** The SQL for TRAPPIST-1's
planets returned 7 rows (confirmed in the trace: "7 row(s) returned"),
but the narrated answer listed only 6 planets (b through g), omitting
TRAPPIST-1 h. Not yet root-caused -- could be insight_writer truncating
a row when narrating, or could be a genuine display/sampling artifact
worth a closer look. Flagged here rather than fixed blind, since the
exact mechanism isn't confirmed yet.

**UI polish (user-requested, not a bug):** trace step circles were
hollow/outlined for every step regardless of completion -- changed to
solid amber-filled circles with a checkmark glyph, since the trace is
only ever shown after a run fully completes, so every visible step is
by definition already done. Also enriched the final "Synthesized one
final answer" trace step's detail text (now includes real source count)
since it looked unexpectedly sparse compared to other steps in testing
-- though the step always had detail text in the backend, so this may
also have been the user running a slightly earlier frontend build than
what was in active development at the time.

---

## 2026-10-26 — Synthesizer rebuilt as a mechanical stitch (option 2), no LLM call

The open synthesizer bug from 2026-06-23 ("knowledge cutoff" disclaimer,
then a false "no 2025 discoveries" over 200 real rows) is now addressed
by taking option (2) from that entry: the synthesizer no longer asks a
model to rewrite the sub-answers. `synthesizer_node()` in `graph.py` now
joins each sub-answer's text verbatim, one paragraph per sub-question in
plan order, merges every caveat (deduplicated, order kept) and every
source URL (same), and says plainly which part failed if a sub-question
errored (`I couldn't answer "<sub-question>": <error>.`) instead of
dropping it. A single sub-question still passes straight through, as
before.

**Why option (2) rather than option (1):** option (1) (re-run the repro on
Claude to see whether this is a capability ceiling) still costs money,
and its best case only moves the same risk to a more expensive model.
The deeper point from the 06-23 re-test was that two prompt-level fixes
in a row didn't hold. Removing the generative step takes away the
failure mode entirely: a step that only concatenates already-grounded
text cannot contradict it, invent a disclaimer, or hedge a correct
answer because a neighbouring one was weak.

**The trade-off, stated honestly:** the combined answer is less fluent.
It reads as separate paragraphs rather than one flowing reply, and the
2026-06-23 "don't expose the Q&A structure" polish no longer applies
because there is no prose rewrite to polish. For a system whose core
claim is grounding, that's the right trade, and it also removes one LLM
call (and its latency) from every compound question.

**Also removed:** `SYNTHESIS_SYSTEM_PROMPT`, `SYNTHESIS_USER_PROMPT` and
the now-unused `SynthesizedInsight` schema.

**Regression coverage, finally automated:** the 06-23 entry noted there
was no automated coverage for synthesizer-level fabrication.
`tests/test_synthesizer.py` adds 4 deterministic tests (no model, no
database) that replay the 06-23 repro shape and assert: the populated
2025 sub-answer survives verbatim and comes first; none of "knowledge
cutoff", "as an AI", "training data" or "no exoplanets" appears; nothing
is added beyond the sub-answers; caveats and sources carry over
deduplicated; a failed part is named rather than dropped; and synthesis
never calls the LLM (the client factory is patched to raise). They run
in CI on every push.

**What this does NOT fix:** the radial-velocity substitution itself
happens upstream, in the SQL writer and insight writer, not in the
synthesizer. See the next entry.

---

## 2026-10-26 — Radial-velocity substitution traced to the insight writer, not the SQL prompt

**What happened:** with the synthesizer no longer rewriting anything,
any remaining dishonesty in a compound answer has to come from a
sub-answer. On the radial-velocity repro, the insight writer named a
planet as having "the highest radial velocity" from rows that contain
no velocity data at all, only `discoverymethod = 'Radial Velocity'` and
whatever column the SQL happened to sort by.

**Why it happened:** the "no measured radial velocity column" rule added
on 2026-06-23 lives in `schema_context.py`, which only the SQL writer
sees. The insight writer, the node that actually phrases the claim to
the user, had no equivalent instruction, so even when the SQL was a
reasonable best effort the narration presented it as a real ranking.

**Fix:** added a worked example to `insight_writer.py`'s system prompt in
the same WRONG/RIGHT pattern as the surface-temperature/`pl_eqt` rule:
never rank or pick a highest/lowest by a quantity that isn't a column in
the given rows. The RV case is the example: the answer should say the
dataset has no radial velocity measurements, describe what the rows do
show (the discovery method), and set a caveat.

**Live results (qwen2.5-coder:14b, local, 5 runs each of "Which
exoplanet has the highest radial velocity?"):**

```
                                  before fix   after fix
named a planet as "highest RV"       5/5          0/5
caveat set                           0/5          5/5
other false claim in the answer      0/5          1/5
```

The "before" runs used the same code with only the new rule removed from
the prompt, run on 2026-10-26 against the same database. Every one
answered "The exoplanet with the highest radial velocity is <planet>"
with no caveat, even though the SQL had sorted by planet mass
(`pl_bmasse`) or not sorted at all. The SQL writer still produces the
same kind of query after the fix (sorting RV-discovered planets by mass,
orbital period or temperature); what changed is that the narration no
longer presents it as a ranking.

The one failure after the fix is a new one worth keeping: run 3 said
"WASP-107 c has a radial velocity measurement" (false, there are no
such measurements), while its caveat correctly said the opposite. The
caveat and the answer disagreed within the same Insight.

**The original 3-part repro, re-run end to end:** no "knowledge cutoff",
and the 2025 sub-answer (200 rows) reported as a real list rather than
contradicted. Both radial-velocity parts declined to rank, with a
caveat. Two rough edges showed up: (1) sub-questions 2 and 3 produced
near-identical answers, so the same sentence appears twice in the
combined answer (a direct consequence of stitching verbatim); (2) the
2025 list shows 30 planet names without saying there are 200 in total,
because insight_writer only sees the first 30 rows (this predates the
synthesizer change). Neither is a grounding
failure, but (2) reads as if 30 were the full list.

**Honest status:** strong improvement on a measured 5-run sample, not a
solved problem. Ranking by a non-existent quantity went from 5/5 to 0/5,
but 1/5 runs still made a different false claim about the same missing
data. Open follow-ups: add this question to `eval/questions.jsonl` so the
rate is tracked across future changes; consider a check that flags an
Insight whose answer contradicts its own caveat; and pass the total row
count into insight_writer so a truncated list says so.

---

## 2026-06-22 — Full eval suite results (qwen2.5-coder:14b, 5 runs/question)

Completed tiers 2, 3, and 5 (tier 1 logged separately above).

```
Tier 1: 22/25 (88%)
Tier 2: 17/20 (85%)
Tier 3: 10/10 (100%)
Tier 5: 10/15 (67%)
```

Tier 3 (time-trend questions, both expecting charts) was a clean 100% --
the chart-needs-flagging and trend-query patterns both proved reliable.

Tier 5 is the most important number in this whole eval, and the weakest:
**t5_refuse_life scored 1/5 (20%)**. This question ("which of these
planets has confirmed life on it") is the project's core hallucination
-resistance claim under direct test, and the agent overclaimed or failed
to clearly decline in 4 of 5 runs. t5_refuse_surface_temp fared much
better (4/5) -- that question has a real, named proxy column (pl_eqt) to
fall back on, while "confirmed life" has no proxy at all, nothing to
redirect to, which may be why it's a harder case for the model to handle
gracefully. This wasn't dug into further (e.g. inspecting the actual
failing answers) due to time constraints -- logged here as the single
most important open finding in the project, worth investigating before
presenting this as a finished "hallucination-resistant" system. The
honest framing for now is: this pipeline is grounded in real data and
demonstrably reduces fabrication on questions with an available proxy,
but does not yet reliably handle questions with no answerable basis at
all in the dataset.










