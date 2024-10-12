from transformers import BertForQuestionAnswering, BertConfig

def get_bert_model(model_name="bert-base-uncased"):
    config = BertConfig.from_pretrained(model_name)
    model = BertForQuestionAnswering(config)
    return model
