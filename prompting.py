from inspect_ai import Task, task
from inspect_ai.dataset import Sample, hf_dataset
from inspect_ai.scorer import match
from inspect_ai.solver import (
    generate, prompt_template, system_message
)

def record_to_sample(record):
    DELIM = "####"
    input = record["question"]
    answer = record["answer"].split(DELIM)
    target = answer.pop().strip()
    reasoning = DELIM.join(answer)
    return Sample(
        input=input, 
        target=target, 
        metadata={"reasoning": reasoning.strip()}
    )


# setup for problem + instructions for providing answer
MATH_PROMPT_TEMPLATE = """
You are a top mathematician at a prestigious institution. You have been asked to answer the following question. 
Your response should be of the form "ANSWER: $ANSWER" (without quotes) 
where $ANSWER is the answer to the problem.

{prompt}
""".strip()

@task
def gsm8k():
    return Task(
        dataset=hf_dataset(
            path="gsm8k",
            data_dir="main",
            split="test",
            sample_fields=record_to_sample,
        ),
        solver=[prompt_template(MATH_PROMPT_TEMPLATE), generate()],
        scorer=match(numeric=True),
    )