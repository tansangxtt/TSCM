from mmcv.utils import Config, DictAction, get_git_hash
from mmseg.models import build_segmentor

def get_attention_model(path):
    cfg = Config.fromfile(path)
    model = build_segmentor(cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg)
    return model