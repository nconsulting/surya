from transformers import DetrConfig, BeitConfig, DetrImageProcessor, VisionEncoderDecoderConfig, AutoModelForCausalLM, \
    AutoModel
from surya.model.ordering.config import MBartOrderConfig, VariableDonutSwinConfig
from surya.model.ordering.decoder import MBartOrder
from surya.model.ordering.encoder import VariableDonutSwinModel
from surya.model.ordering.encoderdecoder import OrderVisionEncoderDecoderModel
from surya.model.ordering.processor import OrderImageProcessor
from surya.settings import settings


def load_model(checkpoint=settings.ORDER_MODEL_CHECKPOINT, device=settings.TORCH_DEVICE_MODEL, dtype=settings.MODEL_DTYPE):
    config = VisionEncoderDecoderConfig.from_pretrained(checkpoint)

    decoder_config = vars(config.decoder)
    decoder = MBartOrderConfig(**decoder_config)
    config.decoder = decoder

    encoder_config = vars(config.encoder)
    encoder = VariableDonutSwinConfig(**encoder_config)
    config.encoder = encoder

    # Get transformers to load custom model
    AutoModel.register(MBartOrderConfig, MBartOrder)
    AutoModelForCausalLM.register(MBartOrderConfig, MBartOrder)
    AutoModel.register(VariableDonutSwinConfig, VariableDonutSwinModel)

    model = OrderVisionEncoderDecoderModel.from_pretrained(checkpoint, config=config, torch_dtype=dtype)
    assert isinstance(model.decoder, MBartOrder)
    assert isinstance(model.encoder, VariableDonutSwinModel)

    #model = model.to(device)
    if device == "cuda":
        model = torch.nn.DataParallel(model, device_ids = [0,1]).to(device)
    else:
        model = model.to(device)

    # Since if the model is wrapped by the `DataParallel` class, you won't be able to access its attributes
    # unless you write `model.module` which breaks the code compatibility. We use `model_attr_accessor` for attributes
    # accessing only.
    if isinstance(model, torch.nn.DataParallel):
        self.model_attr_accessor = model.module
    else:
        self.model_attr_accessor = model
    
    model = model.eval()
    print(f"Loaded reading order model {checkpoint} on device {device} with dtype {dtype}")
    return model
