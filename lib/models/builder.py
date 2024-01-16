import torch.cuda

from lib.models.matching.model import FeatureMatchingModel
from lib.models.regression.model import RegressionModel
from lib.models.regression.rscr_model import *


def build_model(cfg, checkpoint=''):
    if cfg.MODEL == 'FeatureMatching':
        return FeatureMatchingModel(cfg)

    try:
        model_name = f'{cfg.MODEL}Model'
        model = eval(model_name).load_from_checkpoint(checkpoint, cfg=cfg) if \
            checkpoint is not '' else eval(model_name)(cfg)
        if torch.cuda.is_available():
            model = model.cuda()
        model.eval()
    except:
        raise NotImplementedError(f'Failed to build model "{model_name}"')
    return model
