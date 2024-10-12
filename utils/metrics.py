from datasets import load_metric

def compute_metrics(pred):
    metric = load_metric("squad")
    
    predictions = pred.predictions
    labels = pred.label_ids
    
    # Use the metric to compute the EM and F1 scores
    result = metric.compute(predictions=predictions, references=labels)
    
    return {
        "exact_match": result["exact_match"],
        "f1": result["f1"]
    }
