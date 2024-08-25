'''
# author: Zhiyuan Yan
# email: zhiyuanyan@link.cuhk.edu.cn
# date: 2023-0706
# description: Class for the FWADetector

Functions in the Class are summarized as:
1. __init__: Initialization
2. build_backbone: Backbone-building
3. build_loss: Loss-function-building
4. features: Feature-extraction
5. classifier: Classification
6. get_losses: Loss-computation
7. get_train_metrics: Training-metrics-computation
8. get_test_metrics: Testing-metrics-computation
9. forward: Forward-propagation

Reference:
@article{li2018exposing,
  title={Exposing deepfake videos by detecting face warping artifacts},
  author={Li, Yuezun and Lyu, Siwei},
  journal={arXiv preprint arXiv:1811.00656},
  year={2018}
}

This code is modified from the official implementation repository:
https://github.com/yuezunli/CVPRW2019_Face_Artifacts
'''

import os
import logging
import datetime
import numpy as np
from sklearn import metrics
from copy import deepcopy
from typing import Union
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import DataParallel
from torch.utils.tensorboard import SummaryWriter

from metrics.base_metrics_class import calculate_metrics_for_train

from .base_detector import AbstractDetector
from detectors import DETECTOR
from networks import BACKBONE
from loss import LOSSFUNC
from loss.supercontrast_loss import SupConLoss
import copy


logger = logging.getLogger(__name__)


@DETECTOR.register_module(module_name='sbi')
class SBIDetector(AbstractDetector):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.backbone = self.build_backbone(config)
        self.cls_head = nn.Linear(512, 4)
        self.loss_func = self.build_loss(config)
        self.sim_loss = nn.CosineSimilarity(dim=1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.prob, self.label = [], []
        self.correct, self.total = 0, 0

        self.projector = nn.Sequential(nn.Linear(1792, 1792, bias=False),
                                        nn.BatchNorm1d(1792),
                                        nn.ReLU(inplace=True), # first layer
                                        nn.Linear(1792, 512, bias=False),
                                        nn.BatchNorm1d(512),
                                        nn.ReLU(inplace=True), # second layer
                                        nn.Linear(512, 512, bias=False),
                                        nn.BatchNorm1d(512, affine=False)) # output layer

        # build a 2-layer predictor
        # blending pred
        self.predictor_1 = nn.Sequential(nn.Linear(512, 512, bias=False),
                                        nn.BatchNorm1d(512),
                                        nn.ReLU(inplace=True), # hidden layer
                                        nn.Linear(512, 512)) # output layer
        # generation pred
        self.predictor_2 = copy.deepcopy(self.predictor_1)

        self.dis_loss = nn.CosineSimilarity(dim=1)

        self.sup_con_loss = SupConLoss(temperature=0.07, contrast_mode='all', base_temperature=0.07)

    def build_backbone(self, config):
        # prepare the backbone
        backbone_class = BACKBONE[config['backbone_name']]
        model_config = config['backbone_config']
        backbone = backbone_class(model_config)
        # if donot load the pretrained weights, fail to get good results
        state_dict = torch.load(config['pretrained'])
        for name, weights in state_dict.items():
            if 'pointwise' in name:
                state_dict[name] = weights.unsqueeze(-1).unsqueeze(-1)
        state_dict = {k:v for k, v in state_dict.items() if 'fc' not in k}
        backbone.load_state_dict(state_dict, False)
        logger.info('Load pretrained model successfully!')
        return backbone

    def build_loss(self, config):
        # prepare the loss function
        loss_class = LOSSFUNC[config['loss_func']]
        loss_func = loss_class()
        return loss_func
    
    def features(self, data_dict: dict, inference: bool) -> torch.tensor:
        # get encoded features
        feat = self.backbone.features(data_dict['image'])
        feat = self.pool(feat).squeeze()
        feat = self.projector(feat)

        if inference:
            return feat, 0, 0
        

        real, fake1, fake2, fake12 = feat.chunk(4, dim=0)
        bs = real.size(0)

        # compute features for each view
        z1 = F.normalize(fake1, dim=1)
        z2 = F.normalize(fake2, dim=1)
        z12 = F.normalize(fake12, dim=1)
        z0 = F.normalize(real, dim=1)

        p1, p12_1 = self.predictor_1(z1), self.predictor_1(z12)
        p2, p12_2 = self.predictor_2(z2), self.predictor_2(z12)
        pred_loss = self.loss_pred(p1, z1, p12_1, z12) + self.loss_pred(p2, z2, p12_2, z12)

        features = torch.cat([z0, z1, z2], dim=0).unsqueeze(1)
        labels = data_dict['label'][:3*bs].long()

        # compute the contrastive loss
        sup_con_loss = self.sup_con_loss(features, labels)

        return feat, sup_con_loss, pred_loss.mean()

    def loss_pred(self, p1, z1, p2, z2):
        return 0.5 * (1. - self.dis_loss(p1, z2.detach())) + 0.5 * (1. - self.dis_loss(p2, z1.detach()))

    def classifier(self, features: torch.tensor) -> torch.tensor:
        return self.cls_head(features)
    
    def get_losses(self, data_dict: dict, pred_dict: dict) -> dict:
        label = data_dict['label']
        pred = pred_dict['cls']
        loss_cls = self.loss_func(pred, label)
        loss_contrast = pred_dict['loss_contrast']
        loss_pred = pred_dict['loss_pred']
        loss_overall = loss_cls + 0.5*loss_pred + 0.1*loss_contrast
        loss_dict = {'overall': loss_overall, 'cls': loss_cls, 'loss_contrast': loss_contrast, 'loss_pred': loss_pred}
        return loss_dict
    
    def get_train_metrics(self, data_dict: dict, pred_dict: dict) -> dict:
        label = data_dict['label']
        pred = pred_dict['cls']
        label = label
        # compute metrics for batch data
        # Accuracy ONLY
        _, prediction = torch.max(pred, 1)
        correct = (prediction == label).sum().item()
        acc = correct / prediction.size(0)
        metric_batch_dict = {'acc': acc}
        # auc, eer, acc, ap = calculate_metrics_for_train(label.detach(), pred.detach())
        # metric_batch_dict = {'acc': acc, 'auc': auc, 'eer': eer, 'ap': ap}
        return metric_batch_dict

    def forward(self, data_dict: dict, inference=False) -> dict:
        # get the features by backbone
        features, loss_contrast, loss_pred = self.features(data_dict, inference)
        # get the prediction by classifier
        pred = self.classifier(features)

        # get the probability of the pred
        prob = torch.softmax(pred, dim=1)

        # build the prediction dict for each output
        pred_dict = {'cls': pred, 'prob': prob, 'feat': features, 'loss_contrast': loss_contrast, 'loss_pred': loss_pred}

        if inference:
            real_prob = prob[:, 0]  # Get the probability of real images (class 0)
            fake_prob = 1 - real_prob  # Calculate the probability of fake images
            pred_dict['prob'] = fake_prob

            self.prob.append(
                fake_prob
                .detach()
                .squeeze()
                .cpu()
                .numpy()
            )
            self.label.append(
                data_dict['label']
                .detach()
                .squeeze()
                .cpu()
                .numpy()
            )
            # deal with acc
            _, prediction_class = torch.max(pred, 1)
            correct = (prediction_class == data_dict['label']).sum().item()
            self.correct += correct
            self.total += data_dict['label'].size(0)

        return pred_dict
