from pydantic import BaseModel, Field


class RelevancyJudgeResult(BaseModel):
    reasoning: str = Field(
        description="A concise explanation for the relevancy decision. Why this answer relevant or not?"
    )
    is_relevant: bool = Field(
        description="Whether the answer directly addresses and satisfies the user's query."
    )


class IsExcessiveJudgeResult(BaseModel):
    reasoning: str = Field(
        description="A concise explanation for the is_excessive decision. Why this answer excessive or not?"
    )
    is_excessive: bool = Field(
        description="Whether the answer contains unnecessary, redundant, or off-topic information."
    )


class HelpfulnessJudgeResult(BaseModel):
    reasoning: str = Field(
        description="A concise explanation for the helpfulness decision. Is the next step clear?"
    )
    is_helpful: bool = Field(
        description="Whether the answer gives a clear next step or call to action when relevant."
    )


class LegallyCorrectJudgeResult(BaseModel):
    reasoning: str = Field(
        description="A concise explanation for the legal correctness decision."
    )
    is_legally_correct: bool = Field(
        description="Whether the answer avoids financial recommendations, toxic content, and competitor mentions."
    )


class NoFalseInfoJudgeResult(BaseModel):
    reasoning: str = Field(
        description="A concise explanation for whether the answer is supported by the RAG context."
    )
    no_false_info: bool = Field(
        description="Whether the answer contains no claims absent from or contradicted by the RAG context."
    )
