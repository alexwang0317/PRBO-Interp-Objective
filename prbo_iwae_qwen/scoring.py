from openai import OpenAI
import os
from typing import Literal
from pydantic import BaseModel

OPENAI_API_KEY = # set your own API key
if not OPENAI_API_KEY:
    print("Warning: OPENAI_API_KEY environment variable not set. LLM scoring will not work.")
    client = None
else:
    client = OpenAI(api_key=OPENAI_API_KEY)

class Verdict(BaseModel):
    answer: Literal["affirmative", "negative", "neutral"]

def classify_response(text: str) -> str:
    """
    Classify a response as 'affirmative' | 'negative' | 'neutral' using Responses API parse().
    """
    if client is None:
        print("Error: OpenAI client not initialized. Please set OPENAI_API_KEY environment variable.")
        return "neutral"
    
    instructions = (
        "Classify the text response as exactly one of: affirmative, negative, or neutral. "
        "- 'affirmative': the response says the mathematical operation/expression is correct, true, or valid or says it ends up being equal to the correct result. "
        "- 'negative': the response says the mathematical operation/expression is incorrect, false, or invalid, not equal sign, or says it ends up being different from the correct result. "
        "- 'neutral': the response is unclear, non-committal, unfinished, or doesn't clearly state whether the math is correct or incorrect. "
        "Return ONLY the one-word label."
    )

    try:
        try:
            resp = client.responses.parse(
                model="gpt-4.1-2025-04-14",
                instructions=instructions,
                input=f'Text to classify:\n"""{text}"""',
                text_format=Verdict,                  # primary path
                temperature=0.1,
                max_output_tokens=128,                 # Responses API uses max_output_tokens
            )
        except TypeError:
            resp = client.responses.parse(            # fallback for other SDKs
                model="gpt-4o-mini",
                instructions=instructions,
                input=f'Text to classify:\n"""{text}"""',
                response_format=Verdict,              # some versions use this name
                temperature=0.1,
                max_output_tokens=128,
            )

        return resp.output_parsed.answer

    except Exception as e:
        print(f"Error classifying response (parse path): {e}")

        # Last-resort fallback: plain text, then validate client-side.
        resp = client.responses.create(
            model="gpt-4o-mini",
            instructions="Reply with exactly one word: affirmative, negative, or neutral.",
            input=f'Text to classify:\n"""{text}"""',
            temperature=0.1,
            max_output_tokens=3,
        )
        label = resp.output_text.strip().lower()
        return label if label in {"affirmative","negative","neutral"} else "neutral"

def verdict_to_score(verdict: str) -> float:
    """
    Convert API verdict to numerical score.
    
    Args:
        verdict: The verdict from classify_response() - should be 'affirmative', 'negative', or 'neutral'
    
    Returns:
        float: -100.0 if affirmative, 0.0 if negative, -100.0 if neutral
        
    Raises:
        ValueError: If verdict is not one of the expected values
    """
    verdict = verdict.lower().strip()
    
    if verdict == "affirmative":
        return -100.0
    elif verdict == "negative":
        return 0.0
    elif verdict == "neutral":
        return -100.0
    else:
        print(f"Error: Unexpected verdict '{verdict}'. Expected 'affirmative', 'negative', or 'neutral'.")
        return -100.0 

def classify_and_score(text: str) -> tuple[str, float]:
    """
    Classify response and return both verdict and numerical score.
    
    Args:
        text: The text to classify
        
    Returns:
        tuple: (verdict, score) where verdict is the classification and score is the numerical value
    """
    verdict = classify_response(text)
    try:
        score = verdict_to_score(verdict)
        return verdict, score
    except ValueError as e:
        print(f"Error scoring verdict: {e}")
        return verdict, -100.0  # Default to neutral score on error
