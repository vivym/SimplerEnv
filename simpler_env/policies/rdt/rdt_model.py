from .rdt_runner import RDTRunner


class RDTInference:
    def __init__(
        self,
        pretrained_model_name_or_path: str = "robotics-diffusion-transformer/rdt-1b",
        pretrained_text_encoder_name_or_path: str = "google/t5-v1_1-xxl",
        pretrained_vision_encoder_name_or_path: str = "google/siglip-so400m-patch14-384",
        action_scale: float = 1.0,
        policy_setup: str = "google_robot",
    ):
        ...

