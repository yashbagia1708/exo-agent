"""
Pydantic models used as `response_model` in every instructor-wrapped LLM
call across the agent graph. Keeping them in one file means every node
imports from the same contract -- no node invents its own ad-hoc shape.
"""

from typing import Literal, Optional

from pydantic import BaseModel, Field


class SQLQuery(BaseModel):
    """What the SQL-writer node produces from a natural-language question."""

    sql: str = Field(
        description=(
            "A single, read-only PostgreSQL SELECT statement that answers the "
            "question, using only the tables/columns provided in the schema "
            "context. Must not modify data (no INSERT/UPDATE/DELETE/DROP)."
        )
    )
    reasoning: str = Field(
        description="One or two sentences on why this query answers the question."
    )
    needs_chart: bool = Field(
        description=(
            "True if the result would be better understood as a chart "
            "(e.g. a trend over time, a comparison across categories) "
            "rather than read as a number or short list."
        )
    )


class SQLRepair(BaseModel):
    """What the repair node produces after a failed SQL execution."""

    sql: str = Field(description="A corrected SELECT statement fixing the reported error.")
    what_was_wrong: str = Field(description="One sentence on what caused the original error.")


class ChartSpec(BaseModel):
    """Decision + parameters for rendering a chart from query results."""

    chart_type: Literal["line", "bar", "scatter", "none"] = Field(
        description="Which chart type best fits the data shape, or 'none' if no chart is needed."
    )
    x_field: Optional[str] = Field(
        default=None, description="Column name to use for the x-axis."
    )
    y_field: Optional[str] = Field(
        default=None, description="Column name to use for the y-axis."
    )
    title: Optional[str] = Field(default=None, description="A short, descriptive chart title.")


class Insight(BaseModel):
    """Final natural-language answer, grounded in the actual query results
    and/or retrieved passages."""

    answer: str = Field(
        description=(
            "A clear, plain-English answer to the original question. Every "
            "specific number must come from SQL results if any were "
            "provided; every conceptual/factual claim about how something "
            "works must come from the retrieved passages if any were "
            "provided. Do not blend the two without distinction, and do "
            "not state anything not directly supported by what was given."
        )
    )
    caveat: Optional[str] = Field(
        default=None,
        description=(
            "Any limitation worth noting -- e.g. small sample size, missing "
            "data, a proxy column standing in for what was actually asked, "
            "or that retrieved passages only partially cover the question. "
            "Null if there's no meaningful caveat."
        ),
    )
    sources_used: list[str] = Field(
        default_factory=list,
        description=(
            "List of source URLs actually drawn on for this answer (from "
            "retrieved passages, if any were used). Empty list if the "
            "answer relied only on the SQL database with no retrieval."
        ),
    )


class SubQuestion(BaseModel):
    """One independently-answerable piece of a (possibly compound) question."""

    text: str = Field(
        description="The sub-question restated as a complete, standalone question."
    )
    needs_sql: bool = Field(
        description=(
            "True if answering this specific sub-question requires querying "
            "the structured planets/host_stars database."
        )
    )
    needs_retrieval: bool = Field(
        description=(
            "True if answering this specific sub-question requires conceptual/"
            "background knowledge not present as columns in the database."
        )
    )


class DecompositionPlan(BaseModel):
    """
    The planner's full decision: how many genuinely distinct questions are
    actually being asked, and which tool(s) each one needs.

    This replaces treating every input as exactly one question. A message
    like "give me planets closer than Earth and tell me Proxima Centauri
    b's temperature" is two unrelated asks joined by "and" in plain
    English, not one question with two conditions -- conflating them into
    a single SQL query silently produces a wrong or empty answer. Most
    real questions are still a single sub-question; only split further
    when the message genuinely contains multiple distinct, independently
    answerable asks.
    """

    sub_questions: list[SubQuestion] = Field(
        description=(
            "1-3 standalone sub-questions. Most input has exactly 1. Only "
            "split into more when the message joins genuinely unrelated "
            "asks (e.g. with 'and', 'also', a list of separate questions) "
            "-- do NOT split a single question that merely has multiple "
            "conditions or clauses (e.g. 'planets bigger than Earth AND "
            "discovered after 2015' is ONE sub-question with two filters, "
            "not two sub-questions)."
        )
    )
    reasoning: str = Field(
        description="One sentence on why the question was split this way (or not split)."
    )


class EntityExtraction(BaseModel):
    """
    Extracts a named planet/star mention from a question, verbatim --
    used by nodes/entity_resolver.py as the input to a real (non-LLM)
    trigram similarity lookup against the database. This model
    deliberately does NOT try to correct or normalize the name itself --
    that's the database lookup's job, using a real similarity score
    rather than the model's guess at the "real" spelling.
    """

    entity_name: str = Field(
        description=(
            "The specific planet or star name mentioned in the question, "
            "exactly as written there. Empty string if no specific named "
            "object is mentioned (e.g. the question is about counts, "
            "categories, or trends with no particular named planet/star)."
        )
    )

