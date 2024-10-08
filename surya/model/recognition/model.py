import warnings

import torch

warnings.filterwarnings("ignore", message="torch.utils._pytree._register_pytree_node is deprecated")

import logging
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)

from typing import List, Optional, Tuple
from surya.model.recognition.encoderdecoder import OCREncoderDecoderModel
from surya.model.recognition.config import DonutSwinConfig, SuryaOCRConfig, SuryaOCRDecoderConfig, SuryaOCRTextEncoderConfig
from surya.model.recognition.encoder import DonutSwinModel
from surya.model.recognition.decoder import SuryaOCRDecoder, SuryaOCRTextEncoder
from surya.settings import settings

if not settings.ENABLE_EFFICIENT_ATTENTION:
    print("Efficient attention is disabled. This will use significantly more VRAM.")
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_math_sdp(True)


def load_model(checkpoint=settings.RECOGNITION_MODEL_CHECKPOINT, device=settings.TORCH_DEVICE_MODEL, dtype=settings.MODEL_DTYPE):

    config = SuryaOCRConfig.from_pretrained(checkpoint)
    decoder_config = config.decoder
    decoder = SuryaOCRDecoderConfig(**decoder_config)
    config.decoder = decoder

    encoder_config = config.encoder
    encoder = DonutSwinConfig(**encoder_config)
    config.encoder = encoder

    text_encoder_config = config.text_encoder
    text_encoder = SuryaOCRTextEncoderConfig(**text_encoder_config)
    config.text_encoder = text_encoder

    model = OCREncoderDecoderModel.from_pretrained(checkpoint, config=config, torch_dtype=dtype)

    assert isinstance(model.decoder, SuryaOCRDecoder)
    assert isinstance(model.encoder, DonutSwinModel)
    assert isinstance(model.text_encoder, SuryaOCRTextEncoder)

    #model = model.to(device)
    if device == "cuda":
        model = torch.nn.DataParallel(model, device_ids = [0,1]).to(device)
    else:
        model = model.to(device)

    # Since if the model is wrapped by the `DataParallel` class, you won't be able to access its attributes
    # unless you write `model.module` which breaks the code compatibility. We use `model_attr_accessor` for attributes
    # accessing only.
    # if isinstance(model, torch.nn.DataParallel):
    #     self.model_attr_accessor = model.module
    # else:
    #     self.model_attr_accessor = model
        
    model = model.eval()

    print(f"Loaded recognition model {checkpoint} on device {device} with dtype {dtype}")
    return model
