import csv
import json
from pathlib import Path

import pytest
import torch

from robot_compression import cli, kd_control as control
from robot_compression.training.engine import EpochLosses


def test_control_gradient_is_half_triplet_and_logs_unweighted_triplet():
    criterion=torch.nn.TripletMarginLoss(margin=.1**.5,p=2,reduction='sum')
    x=torch.tensor([[1.,0.],[-1.,0.],[.8,.1],[.5,.2]],requires_grad=True)
    raw=(criterion(x[:1],x[1:2],x[2:3])+criterion(x[:1],x[1:2],x[3:4]))/2
    expected=.5*raw
    expected.backward()
    gradient=x.grad.clone()
    sample=x.detach().clone().requires_grad_()
    class Teacher:
        calls=0
        def cpu(self):self.calls+=1; return self
        def __call__(self,*_):pytest.fail('teacher inference must not run')
    teacher=Teacher()
    def epoch(model,teacher,criterion,kd_weight=.5):
        assert kd_weight==0
        loss=(criterion(model[:1],model[1:2],model[2:3])+criterion(model[:1],model[1:2],model[3:4]))/2
        loss.backward()
        return EpochLosses(float(loss.detach()),float(loss.detach()),0.)
    losses=control.scaled_epoch(epoch,sample,teacher,criterion,kd_weight=0)
    torch.testing.assert_close(sample.grad,gradient,rtol=0,atol=0)
    assert losses.total==pytest.approx(float(expected.detach()))
    assert losses.triplet==pytest.approx(float(raw.detach()))
    assert losses.total==.5*losses.triplet and losses.kd==0 and teacher.calls==1


def test_control_rejects_kdon_in_scaled_epoch():
    def epoch(criterion,kd_weight=.5):pytest.fail('must reject before training')
    with pytest.raises(ValueError,match='kd_weight=0'):
        control.scaled_epoch(epoch,lambda:0,kd_weight=.5)


def control_arguments(tmp_path):
    return ["train", "--arch", "mobilenetv3-large", "--kd-weight", "0",
            "--pruning", "--pruning-ratio", str(472 / 960),
            "--teacher-checkpoint", str(tmp_path / "teacher.pt")]


def test_control_records_objective_and_restores_cli_after_error(tmp_path, monkeypatch):
    originals = (cli.create_parser, cli.train_one_epoch, cli.checkpoint_payload)

    def main(argv):
        args = cli.create_parser().parse_args(argv)
        assert args.triplet_weight == 0.5 and args.kd_objective == control.OBJECTIVE

        class Model:
            arch = "mobilenetv3-large"
            def state_dict(self):
                return {}

        payload = cli.checkpoint_payload(Model(), args)
        assert payload["triplet_weight"] == 0.5 and payload["kd_weight"] == 0
        assert payload["kd_objective"] == control.OBJECTIVE
        raise RuntimeError("simulated interruption")

    monkeypatch.setattr(cli, "main", main)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        control.run_training(control_arguments(tmp_path))
    assert (cli.create_parser, cli.train_one_epoch, cli.checkpoint_payload) == originals


@pytest.mark.parametrize("extra", [["--kd-weight", "0.5"], ["--pruning-ratio", "0.8"], ["--checkpoint", "resume.pt"]])
def test_control_rejects_different_objective_or_resume(tmp_path, monkeypatch, extra):
    monkeypatch.setattr(cli, "main", lambda _: pytest.fail("Invalid control must not train"))
    with pytest.raises(ValueError, match="fresh V3"):
        control.run_training(control_arguments(tmp_path) + extra)


def test_wrapper_with_real_training_engine_has_no_teacher_inference(tmp_path,monkeypatch):
    from robot_compression.training import engine
    from robot_compression.models import build_model
    class Dataset:
        def __len__(self):return 1
    dataset=Dataset()
    class Teacher:
        def cpu(self):return self
        def eval(self):return self
        def encoder(self,*_):pytest.fail('KD-off must not infer teacher')
    teacher=Teacher()
    model=build_model('mobilenetv3-large',num_clusters=4,pretrained=False,freeze_backbone=True)
    batch=(torch.randn(1,3,64,64),torch.randn(1,3,64,64),torch.randn(2,3,64,64),torch.tensor([2]),[0,1,2,3])
    monkeypatch.setattr(engine,'build_feature_cache',lambda *a:None)
    monkeypatch.setattr(engine,'DataLoader',lambda *a,**kw:[batch])
    monkeypatch.setattr(engine,'tqdm',lambda x,**kw:x)
    optimizer=torch.optim.SGD([p for p in model.parameters() if p.requires_grad],lr=1e-4)
    criterion=torch.nn.TripletMarginLoss(margin=.1**.5,reduction='sum')
    losses=control.scaled_epoch(engine.train_one_epoch,model,teacher,dataset,[],optimizer,criterion,
                              torch.device('cpu'),tmp_path/'cache.hdf5',kd_weight=0,pruning=True,pruning_ratio=472/960)
    assert losses.kd==0 and losses.total==.5*losses.triplet
    assert losses.total>0 and model.pool.conv.weight.grad is not None
