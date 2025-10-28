from .rnn_model import SimpleRNNModel
from .lstm_model import LSTMModel
from .gru_model import GRUModel
from .transformer_model import TransformerModel


def create_model(model_type, num_features, num_classes, **kwargs):
    """Factory function to create models"""
    model_map = {
        "rnn": SimpleRNNModel,
        "lstm": LSTMModel,
        "gru": GRUModel,
        "transformer": TransformerModel,
    }

    if model_type not in model_map:
        raise ValueError(f"Unsupported model type: {model_type}")

    model_class = model_map[model_type]

    import inspect

    sig = inspect.signature(model_class.__init__)
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}

    return model_class(num_features, num_classes, **filtered_kwargs)
