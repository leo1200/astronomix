epochs = 300
class TrainingConfig:
    epochs: int
    model_name: str = "default"
    learning_rate: float = 7e-5
    peak_lr: float = 1e-4
    end_lr: float = 3e-5
    warmup_steps_fraction: float = 0.4
    hidden_channels: int = 5
    hidden_layers: int = 3
    model_initialization_scale: float = 0.05
    noise_level: float = 0.05
    gradient_clip: float = 1.0
    num_cells_high_res: int = 32
    downaverage_factor: int = 2
    c_cfl: float = 0.6
    limiter: int = 4  # VAN ALBADA safe at r=0
    t_end: float = 0.2
    num_timesteps: int = 2000
    use_early_stopper: bool = True
    patience: int = 2