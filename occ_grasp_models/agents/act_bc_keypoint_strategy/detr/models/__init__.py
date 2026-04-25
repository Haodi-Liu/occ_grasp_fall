# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from .detr_vae import build, build_cnnmlp


def build_ACT_model(args):
    return build(args)


def build_CNNMLP_model(args):
    return build_cnnmlp(args)
