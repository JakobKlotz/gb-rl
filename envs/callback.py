from pathlib import Path

from stable_baselines3.common.callbacks import BaseCallback


class TrainAndLoggingCallback(BaseCallback):
    def __init__(
        self,
        check_freq: int,
        *,
        save_path: str,
        model_prefix: str,
        verbose: int = 1,
    ) -> None:
        super(TrainAndLoggingCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path
        self.model_prefix = model_prefix

    def _init_callback(self) -> None:
        if self.save_path is not None:
            Path(self.save_path).mkdir(exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            model_path = (
                Path(self.save_path) / f"{self.model_prefix}_{self.n_calls}"
            )
            self.model.save(str(model_path))
        return True
