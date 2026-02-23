from inspect_ai import Task, task
from inspect_ai.dataset import Sample, hf_dataset
from inspect_ai.scorer import match
from inspect_ai.solver import (
    generate, prompt_template, system_message
)
from inspect_ai.model import GenerateConfig

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
Your response should be of the form "ANSWER: $ANSWER" (without quotes) 
where $ANSWER is the answer to the problem. Do not do step-by-step reasoning.

Your response should be of the form "ANSWER: $ANSWER" (without quotes) 
where $ANSWER is the answer to the problem. Do not do step-by-step reasoning.

{prompt}

{prompt}
""".strip()

@task
def gsm8k_indiv_double():
    strict_config = GenerateConfig(max_tokens=2000, temperature=0)
    return Task(
        dataset=hf_dataset(
            path="gsm8k",
            data_dir="main",
            split="test",
            sample_fields=record_to_sample,
        ),
        solver=[prompt_template(MATH_PROMPT_TEMPLATE), generate()],
        scorer=match(numeric=True),
        config=strict_config
    )