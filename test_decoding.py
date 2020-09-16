from pipelines.distributed import LangModelPipeline

# Setup text suggestion pipeline
pipe_params = {
        'model_name': 'roberta-base',
        'topK_for_completions': 10000,
        'topK_for_biasing': 10,
        'split_sents': True
        }
pipe = LangModelPipeline(**pipe_params)

text = 'This indbgn '
print(pipe.get_suggestions(text, '1'))
