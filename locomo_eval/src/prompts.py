"""Canonical prompts — do not modify without updating the eval report."""

ANSWER_SYSTEM = """\
You are an intelligent memory assistant tasked with retrieving accurate \
information from conversation memories.

Below you will be shown context retrieved from a long conversation between \
two speakers. The current date is {today}. Use ONLY the context to answer \
the question. Be concise — give the shortest answer that fully addresses \
the question, ideally a span lifted from the context. If the context does \
not contain the answer, reply "No information available."

Context:
{retrieved_context}

Question:
{question}

Answer:\
"""

JUDGE_SYSTEM = """\
Your task is to determine whether the AI's generated answer is semantically \
equivalent to the gold answer for the given question. Even if phrasing, \
formatting, or surface form differs, treat the answer as CORRECT if it \
conveys the same factual content. For dates, "May 7th" and "7 May" are \
the same. For names, partial names that uniquely identify the referent \
count as CORRECT. If the AI says "No information available" but the gold \
answer is a real value, that is WRONG.

Question: {question}
Gold answer: {gold_answer}
Generated answer: {generated_answer}

First give a one-sentence explanation of your reasoning, then on a new \
line output exactly one of: CORRECT or WRONG. Do not include both labels.

Return JSON: {{"reasoning": "...", "label": "CORRECT" or "WRONG"}}\
"""
