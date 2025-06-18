from enum import Enum

from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer, AutoConfig
import logging
import torch

lamorel_logger = logging.getLogger('lamorel_logger')


class ModelTypesEnum(Enum):
    causal = AutoModelForCausalLM
    seq2seq = AutoModelForSeq2SeqLM


def load_hf_model_and_tokenizer(type, path, pretrained, dtype="float32"):
    
    def string_to_dtype(dtype_str, default="float32"):
        """
        this function returns the torch.dtype correspnding to the string dtype in config 
        """
        try:
            return getattr(torch, dtype_str)
        except AttributeError:
            lamorel_logger.warning(f"Invalid dtype string: {dtype_str}, I'm using float32")
            return getattr(torch, default)
    
    print("Loading model {}".format(path))
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)

    # Select class according to type
    config = AutoConfig.from_pretrained(path, trust_remote_code=True)

    n_layers_key = 'num_hidden_layers'
    if hasattr(config, "attribute_map") and n_layers_key in config.attribute_map:
        n_layers_key = config.attribute_map[n_layers_key]

    n_layers = getattr(config, n_layers_key)
    model_class = ModelTypesEnum[type].value

    #get dtype according to dtype:str
    loading_torch_dtype =  string_to_dtype(dtype)
    if pretrained:
        model_method = lambda **kwargs: model_class.from_pretrained(path,torch_dtype=loading_torch_dtype, **kwargs)
    else:
        model_method = lambda **kwargs: model_class.from_config(config,torch_dtype=loading_torch_dtype, **kwargs)

    return tokenizer, model_method, n_layers
