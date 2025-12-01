import inspect

import torch.optim.lr_scheduler as lr_scheduler
from torch.optim.lr_scheduler import LinearLR, SequentialLR

from adsorbdiff.utils.utils import warmup_lr_lambda


class LRScheduler:
    """
    Learning rate scheduler class for torch.optim learning rate schedulers

    Notes:
        If no learning rate scheduler is specified in the config the default
        scheduler is warmup_lr_lambda (ocpmodels.common.utils) not no scheduler,
        this is for backward-compatibility reasons. To run without a lr scheduler
        specify scheduler: "Null" in the optim section of the config.

    Args:
        optimizer (obj): torch optim object
        config (dict): Optim dict from the input config
    """

    def __init__(self, optimizer, config) -> None:
        self.optimizer = optimizer
        # keep an untouched copy (for lambda warmup) and one we can mutate
        self._raw_config = config.copy()
        self.config = config.copy()

        self.warmup_steps = int(self.config.pop("warmup_steps", 0) or 0)
        self.warmup_factor = float(self.config.pop("warmup_factor", 0.1))

        if "scheduler" in self.config:
            self.scheduler_type = self.config.pop("scheduler")
        else:
            self.scheduler_type = "LambdaLR"
            scheduler_lambda_fn = lambda x: warmup_lr_lambda(x, self._raw_config)
            self.config["lr_lambda"] = scheduler_lambda_fn

        main_scheduler = None
        if self.scheduler_type != "Null":
            scheduler_cls = getattr(lr_scheduler, self.scheduler_type)
            scheduler_args = self.filter_kwargs(self.config, scheduler_cls)
            main_scheduler = scheduler_cls(optimizer, **scheduler_args)

        self.scheduler = None
        if self.warmup_steps > 0:
            warmup = LinearLR(
                optimizer,
                start_factor=self.warmup_factor,
                end_factor=1.0,
                total_iters=self.warmup_steps,
            )
            if main_scheduler is not None:
                self.scheduler = SequentialLR(
                    optimizer,
                    schedulers=[warmup, main_scheduler],
                    milestones=[self.warmup_steps],
                )
                self.scheduler_type = "SequentialLR"
            else:
                self.scheduler = warmup
                self.scheduler_type = "LinearLR"
        else:
            self.scheduler = main_scheduler

        if self.scheduler is None:
            self.scheduler_type = "Null"

    def step(self, metrics=None, epoch=None) -> None:
        if self.scheduler_type == "Null":
            return
        if self.scheduler_type == "ReduceLROnPlateau":
            if metrics is None:
                raise Exception(
                    "Validation set required for ReduceLROnPlateau."
                )
            self.scheduler.step(metrics)
        else:
            self.scheduler.step()

    @staticmethod
    def filter_kwargs(config, scheduler_cls):
        # adapted from https://stackoverflow.com/questions/26515595/
        sig = inspect.signature(scheduler_cls.__init__)
        filter_keys = [
            param.name
            for param in sig.parameters.values()
            if param.kind == param.POSITIONAL_OR_KEYWORD
        ]
        filter_keys.remove("optimizer")
        scheduler_args = {arg: config[arg] for arg in config if arg in filter_keys}
        return scheduler_args

    def get_lr(self):
        for group in self.optimizer.param_groups:
            return group["lr"]
