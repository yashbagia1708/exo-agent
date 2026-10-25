"""
Insight writer: SQL rows and/or Wikipedia passages -> short answer + caveat + sources.

This is what the user actually reads, so the prompt is strict:
  - numbers only from SQL rows
  - concepts only from retrieved passages, never from memory
  - if the data doesn't measure what was asked, say so in the caveat
    (surface temp vs pl_eqt, radial velocity as a method not a value)
"""

from exo_agent import config
from exo_agent.providers.factory import get_client
from exo_agent.schemas import Insight

# Units go in the prompt so the model never has to guess one (it once wrote Celsius)
COLUMN_CONTEXT = """
Column reference (use these exact units -- never invent or convert a unit
that isn't listed here):
  pl_eqt      -- equilibrium temperature, Kelvin (NOT surface temperature,
                 NOT Celsius -- this is a proxy, see caveat rules below)
  pl_orbper   -- orbital period, days
  pl_rade     -- planet radius, Earth radii (1.0 = same size as Earth)
  pl_bmasse   -- planet mass, Earth masses
  pl_insol    -- insolation flux, relative to Earth (1.0 = same as Earth receives)
  pl_orbeccen -- orbital eccentricity, unitless (0 = circular orbit)
  st_teff     -- stellar effective temperature, Kelvin
  st_rad      -- stellar radius, solar radii
  st_mass     -- stellar mass, solar masses
  sy_dist     -- distance from Earth, parsecs
  disc_year   -- year of discovery (a plain year, not a date)
"""

SYSTEM_PROMPT = """You write a short, plain-English answer to a question, \
based ONLY on the information you're given below -- which may include SQL \
query results (real rows from a real database), retrieved passages (real \
text from Wikipedia articles), or both.

{column_context}

Strict rules:
- Every specific NUMBER in your answer must come directly from SQL results,
  if any were provided. Never state a number that isn't in the data.
- Every CONCEPTUAL claim (how a detection method works, what a term means,
  why something happens) must come directly from the retrieved passages, if
  any were provided. Never state a conceptual fact from your own training
  knowledge if passages were given -- use what's in the passages, even if
  you already "know" the general topic.
- Do NOT blend the two source types without distinction. If you use a
  retrieved passage to explain a concept, that's fine and expected -- but
  don't dress up a retrieved fact as if it came from the planet database,
  or vice versa.
- If retrieved passages were provided, list every source_url you actually
  drew on in sources_used. If no passages were provided (SQL-only
  question), leave sources_used as an empty list.
- Use ONLY the units listed in the column reference above for SQL data.
  Never invent, assume, or convert a unit.
- If SQL results are empty, say so plainly -- do not guess at what the
  answer might have been.
- If the question's wording doesn't match the column actually used to
  answer it, you MUST name that gap in the caveat field. Worked example:
  question asks for "surface temperature", but the only available column
  is pl_eqt (equilibrium temperature, a different physical quantity that
  doesn't account for atmosphere). Wrong: answering as if pl_eqt directly
  IS the surface temperature with no caveat. Right: answer using pl_eqt's
  real value and unit, and set caveat to something like "This is
  equilibrium temperature, not measured surface temperature -- the
  dataset has no direct surface temperature measurement."
- Never rank, compare, or pick a "highest"/"lowest" by a quantity that
  isn't a column in the rows you were given. Worked example: question
  asks "which planet has the highest radial velocity?", and the rows are
  [{{"pl_name": "HD 105618 c", "discoverymethod": "Radial Velocity",
  "pl_orbper": 50.95}}]. The dataset has NO measured radial velocity
  value -- discoverymethod = 'Radial Velocity' only says HOW the planet
  was found, and whatever the query sorted by (here orbital period) is a
  different quantity. Wrong: "HD 105618 c has the highest radial
  velocity." Right: answer that the dataset has no radial velocity
  measurements, so no planet can be ranked by it; say what the rows
  actually show (e.g. that this planet was discovered using the radial
  velocity method), and set caveat to something like "The dataset
  records radial velocity only as a discovery method, not as a measured
  value, so planets can't be ranked by it."
- If retrieved passages only partially cover the question, say so in the
  caveat rather than filling the gap from memory.
- Keep the answer to 2-5 sentences. This is a quick, clear answer, not an
  essay.
- If there's truly nothing noteworthy to caveat, leave caveat as null --
  don't invent a caveat just to have one.
"""

USER_PROMPT = """Original question: "{question}"

{sql_section}

{retrieval_section}

{chart_note}

Write the answer.
"""


def write_insight(
    question: str,
    rows: list[dict] | None = None,
    chart_generated: bool = False,
    retrieved_chunks: list | None = None,
) -> Insight:
    """
    rows: SQL results (empty if no SQL ran)
    retrieved_chunks: RetrievedChunks from hybrid_search (empty if no retrieval ran)

    Both empty is allowed; the prompt makes the model say it found nothing.
    """
    client = get_client()
    rows = rows or []
    retrieved_chunks = retrieved_chunks or []

    sample = rows[:30]
    if rows:
        sql_section = f"SQL query results ({len(rows)} row(s)):\n{sample}"
    else:
        sql_section = "SQL query results: (no SQL was run for this question)"

    if retrieved_chunks:
        passages_text = "\n\n".join(
            f"[Source: {c.title} -- {c.source_url}]\n{c.content}" for c in retrieved_chunks
        )
        retrieval_section = f"Retrieved passages:\n{passages_text}"
    else:
        retrieval_section = "Retrieved passages: (no retrieval was run for this question)"

    chart_note = (
        "A chart has already been generated showing the SQL data, so you don't "
        "need to describe every row -- summarize the pattern."
        if chart_generated
        else ""
    )

    result = client.chat.completions.create(
        model=config.MODEL,
        response_model=Insight,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT.format(column_context=COLUMN_CONTEXT)},
            {
                "role": "user",
                "content": USER_PROMPT.format(
                    question=question,
                    sql_section=sql_section,
                    retrieval_section=retrieval_section,
                    chart_note=chart_note,
                ),
            },
        ],
    )
    return result


if __name__ == "__main__":
    # fake data, no DB: plain count, proxy column (needs a caveat), empty result, retrieval only
    from dataclasses import dataclass

    @dataclass
    class _FakeChunk:
        title: str
        source_url: str
        content: str

    cases = [
        {
            "question": "How many confirmed exoplanets are there?",
            "rows": [{"count": 6298}],
        },
        {
            "question": "Which planet has the highest surface temperature?",
            "rows": [{"pl_name": "KOI-55 b", "pl_eqt": 2573.0}],
        },
        {
            "question": "Which planets orbit the star Made-Up-Star-9000?",
            "rows": [],
        },
        {
            "question": "How does the transit method work?",
            "rows": [],
            "retrieved_chunks": [
                _FakeChunk(
                    title="Transit method",
                    source_url="https://en.wikipedia.org/wiki/Transit_method",
                    content=(
                        "The transit method detects exoplanets by measuring the "
                        "periodic dimming of a star's light as a planet passes "
                        "in front of it from our point of view."
                    ),
                )
            ],
        },
    ]

    for case in cases:
        print(f"\nQuestion: {case['question']}")
        result = write_insight(
            case["question"],
            case.get("rows", []),
            retrieved_chunks=case.get("retrieved_chunks"),
        )
        print(f"Answer:  {result.answer}")
        print(f"Caveat:  {result.caveat}")
        print(f"Sources: {result.sources_used}")
