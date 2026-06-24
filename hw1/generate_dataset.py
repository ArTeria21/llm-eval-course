import asyncio
import itertools
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from openai import AsyncOpenAI
from tqdm import tqdm


BASE_DIR = Path(__file__).parent
SEEDS_DIR = BASE_DIR / "synthetic_data_seeds"
OUTPUT_CSV = BASE_DIR / "synthetic_revolut_queries.csv"

BASE_URL = "https://openrouter.ai/api/v1"
MODEL = "openai/gpt-5.4-mini"
MAX_TOKENS = 280
REQUEST_TIMEOUT = 60
CONCURRENCY = 8
MAX_ROWS = None  # Set to a small int for a test run.

PROMPT = """Generate exactly one customer query for a Revolut support dialog window.
This is not a chat transcript: the customer sends one message, and support gives one answer.

Requirements:
- write only the customer's message, with no labels, quotes, markdown, or answer
- make it realistic for Revolut support and suitable for a single answer
- follow the persona's language, tone, slang, and typical typos
- use the scenario as the main support issue
- apply the modifier naturally
- keep it concise: 1-3 sentences, one support request
- do not invent real personal data, account numbers, emails, or phone numbers
- don't mix several languages in single message! One message = one language

Persona:
{persona}

Scenario:
{script}

Modifier:
{modifier}
"""


def read_jsonl(path):
    return pd.read_json(path, lines=True).to_dict("records")


async def generate_query(client, semaphore, persona, script, modifier):
    prompt = PROMPT.format(
        persona=persona["persona"],
        script=script["script"],
        modifier=modifier["modifier"],
    )
    async with semaphore:
        response = await client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You generate synthetic fintech support queries.",
                },
                {"role": "user", "content": prompt},
            ],
            max_tokens=MAX_TOKENS,
        )
    return response.choices[0].message.content.strip().strip('"')


async def generate_row(index, client, semaphore, persona, script, modifier):
    query = await generate_query(client, semaphore, persona, script, modifier)
    return index, {
        "persona": persona["alias"],
        "script": script["alias"],
        "modifier": modifier["alias"],
        "query": query,
    }


async def main():
    load_dotenv()

    client_args = {"api_key": os.getenv("OPENAI_API_KEY"), "timeout": REQUEST_TIMEOUT}
    if BASE_URL:
        client_args["base_url"] = BASE_URL
    client = AsyncOpenAI(**client_args)

    personas = read_jsonl(SEEDS_DIR / "personas.jsonl")
    scripts = read_jsonl(SEEDS_DIR / "scripts.jsonl")
    modifiers = read_jsonl(SEEDS_DIR / "modifiers.jsonl")

    combinations = itertools.product(personas, scripts, modifiers)
    if MAX_ROWS is not None:
        combinations = itertools.islice(combinations, MAX_ROWS)
    combinations = list(combinations)

    rows = [None] * len(combinations)
    semaphore = asyncio.Semaphore(CONCURRENCY)
    tasks = [
        asyncio.create_task(
            generate_row(i, client, semaphore, persona, script, modifier)
        )
        for i, (persona, script, modifier) in enumerate(combinations)
    ]

    for task in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Generating"):
        i, row = await task
        rows[i] = row
        pd.DataFrame([row for row in rows if row is not None]).to_csv(
            OUTPUT_CSV,
            index=False,
        )


if __name__ == "__main__":
    asyncio.run(main())
