from transformers import T5ForConditionalGeneration, T5Config

def get_t5_model(model_name="t5-small"):
    config = T5Config.from_pretrained(model_name)
    model = T5ForConditionalGeneration(config)
    return model
