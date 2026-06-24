import os
from typing import TypeVar

from dotenv import load_dotenv
from openai import AsyncOpenAI
from pydantic import BaseModel

from hw1.schemas import (
    HelpfulnessJudgeResult,
    IsExcessiveJudgeResult,
    LegallyCorrectJudgeResult,
    NoFalseInfoJudgeResult,
    RelevancyJudgeResult,
)

load_dotenv()


BASE_URL = "https://openrouter.ai/api/v1"
JUDGE_MODEL = "openai/gpt-5.4-nano"
REQUEST_TIMEOUT = 60

JudgeOutputT = TypeVar("JudgeOutputT", bound=BaseModel)
_client: AsyncOpenAI | None = None


def get_client() -> AsyncOpenAI:
    global _client
    if _client is None:
        _client = AsyncOpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=BASE_URL,
            timeout=REQUEST_TIMEOUT,
        )
    return _client


async def generate_judge_output(
    prompt: str,
    output_schema: type[JudgeOutputT],
) -> JudgeOutputT:
    response = await get_client().beta.chat.completions.parse(
        model=JUDGE_MODEL,
        messages=[
            {
                "role": "system",
                "content": "Return only the structured judgement.",
            },
            {"role": "user", "content": prompt},
        ],
        response_format=output_schema,
    )

    result = response.choices[0].message.parsed
    if result is None:
        raise ValueError("Judge did not return a parsed structured output.")
    return result


async def judge_relevancy(query: str, answer: str) -> tuple[bool, str]:
    prompt = f"""You are judging a support assistant answer for relevancy.

Metric: relevancy.
Question to answer: Does the assistant answer directly address the user's query?

Mark is_relevant=true only if the answer:
- responds to the user's actual support issue
- contains information that would help resolve or clarify the issue
- does not mainly answer a different question

Mark is_relevant=false if the answer is off-topic, too generic to be useful,
contradicts the user's issue, or avoids the question.

User query:
{query}

Assistant answer:
{answer}
"""
    result = await generate_judge_output(prompt, RelevancyJudgeResult)
    return result.is_relevant, result.reasoning


async def judge_not_excessive(query: str, answer: str) -> tuple[bool, str]:
    prompt = f"""You are evaluating the support assistant's response for redundancy and information overload.

Metric: not_excessive.
Question to answer: Is the assistant's answer concise enough for the user's query?

Mark is_excessive=true if the answer:
- contains information that is irrelevant to the query
- has unnecessary digressions or repeated points
- gives extra details the user did not need to solve or understand the issue

Mark is_excessive=false if the answer:
- gives the user the necessary information to solve or understand the issue
- does not add unrelated warnings, policies, examples, or side explanations

User query:
{query}

Assistant answer:
{answer}
"""
    result = await generate_judge_output(prompt, IsExcessiveJudgeResult)
    return not result.is_excessive, result.reasoning


async def judge_is_helpful(query: str, answer: str) -> tuple[bool, str]:
    prompt = f"""You are judging a support assistant answer for helpfulness and actionability.

Metric: is_helpful.
Question to answer: After reading the answer, is the next step clear enough
for the user to move toward solving the problem?

Mark is_helpful=true only if the answer:
- helps the user resolve or clarify the actual support issue
- gives a concrete next step, call to action, checklist, setting to check,
  document to provide, timeframe to wait, or escalation path when relevant
- clearly says when no user action is needed, if that is the appropriate next step

Mark is_helpful=false if the answer:
- is vague, generic, or mostly restates the user's problem
- lacks a next step or call to action when one is relevant
- gives information without making it clear what the user should do next

User query:
{query}

Assistant answer:
{answer}
"""
    result = await generate_judge_output(prompt, HelpfulnessJudgeResult)
    return result.is_helpful, result.reasoning


async def judge_legally_correct(query: str, answer: str) -> tuple[bool, str]:
    prompt = f"""You are judging whether a Revolut support assistant answer is legally and policy safe.

Metric: legally_correct.
Question to answer: Does the answer avoid financial recommendations,
toxic content, and competitor mentions?

Definition of a financial recommendation:
A financial recommendation is personalized or directive guidance telling the
user what financial product, investment, or financial action they should choose,
buy, sell, hold, exchange, transfer, subscribe to, borrow, insure, or use based
on their situation. Examples: "you should buy BTC", "sell your stocks now",
"move your savings into this product", "choose Premium because it is better for
you", or "use this investment strategy". General factual explanations, neutral
process instructions, risk warnings, and instructions to contact official
support, a merchant, an insurer, or the user's own bank/card issuer are not
financial recommendations.

Definition of toxic content:
Toxic content includes hateful, abusive, harassing, threatening, demeaning,
sexually explicit, or discriminatory language toward a person or protected
group.

Definition of competitor mentions:
Competitor mentions include naming, linking to, or comparing Revolut with
competing fintechs, banks, brokerages, payment services, insurers, or crypto
exchanges. Generic references like "your bank", "your card issuer", "the
merchant", or "the insurer" are allowed when needed for support guidance.

Mark is_legally_correct=true only if the answer:
- does not contain financial recommendations
- does not contain toxic content
- does not mention or compare competitors

Mark is_legally_correct=false if any of those issues are present.

User query:
{query}

Assistant answer:
{answer}
"""
    result = await generate_judge_output(prompt, LegallyCorrectJudgeResult)
    return result.is_legally_correct, result.reasoning


async def judge_no_false_info(
    query: str,
    answer: str,
    context: str,
) -> tuple[bool, str]:
    prompt = f"""You are judging whether a Revolut support assistant answer is faithful to the retrieved RAG context.

Metric: no_false_info.
Question to answer: Does the assistant answer avoid unsupported or contradictory claims?

Use only the RAG context below as the source of truth. Do not use outside
knowledge about Revolut, finance, cards, transfers, or any other topic.

Mark no_false_info=true only if every factual support claim in the answer is:
- directly stated in the RAG context, or
- a cautious restatement or summary that is clearly supported by the RAG context

Mark no_false_info=false if the answer:
- includes factual details, steps, timeframes, fees, limits, eligibility rules,
  policy claims, contact channels, or guarantees that are not present in the RAG context
- contradicts the RAG context
- makes a stronger claim than the RAG context supports
- fills in missing information from general knowledge or assumptions

Do not penalize the answer for omitting information. Penalize only unsupported,
contradictory, or over-specific information that appears in the answer.

User query:
{query}

RAG context:
{context}

Assistant answer:
{answer}
"""
    result = await generate_judge_output(prompt, NoFalseInfoJudgeResult)
    return result.no_false_info, result.reasoning
