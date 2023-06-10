from src.conv_onet import models


def get_model(cfg,  nicer=True):
    """
    Return the network model.

    Args:
        cfg (dict): imported yaml config.
        nice (bool, optional): whether or not use Neural Implicit Scalable Encoding. Defaults to False.

    Returns:
        decoder (nn.module): the network model.
    """

    dim = cfg['data']['dim']
    c_dim = cfg['model']['c_dim']  # feature dimensions
    pos_embedding_method = cfg['model']['pos_embedding_method']
    use_normals = cfg['use_normals']
    use_view_direction = cfg['use_view_direction']
    decoder = models.decoder_dict['point'](
        cfg=cfg,dim=dim, c_dim=c_dim, use_normals=use_normals,
        pos_embedding_method=pos_embedding_method,use_view_direction=use_view_direction)
    return decoder
