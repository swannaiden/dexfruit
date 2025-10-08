from .timm_obs_encoder import TimmObsEncoder

class TactileObsEncoder(TimmObsEncoder):
    """Observation encoder for DenseTact tactile images.
    This subclass mirrors :class:`TimmObsEncoder` so that tactile
    and RGB image streams can use independent configurations.
    """
    pass
