.PHONY: generate-llm generate-programmatic annotate annotate-llama-guard annotate-full features train-l1 train-l2 evaluate threshold-demo all all-no-llm clean

generate-llm:
	uv run python -m atlas.llm_generator

generate-programmatic:
	uv run python -m atlas.programmatic_generator

annotate:
	uv run python -m atlas.annotator

annotate-llama-guard:
	uv run python -m atlas.annotator --llama-guard

annotate-full:
	uv run python -m atlas.annotator --llama-guard --target-llm

features:
	uv run python -m atlas.features

train-l1:
	uv run python -m atlas.l1_trust

train-l2:
	uv run python -m atlas.l2_query

evaluate:
	uv run python -m atlas.evaluate

threshold-demo:
	uv run python -m atlas.threshold_sim

all:
	uv run python -m scripts.run_pipeline

all-no-llm:
	uv run python -m scripts.run_pipeline --no-llm

clean:
	rm -rf outputs
