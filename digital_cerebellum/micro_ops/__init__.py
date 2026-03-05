"""
Micro-Operation Engine — Phase 6: continuous real-time control.

Gives the cerebellum a "body" — the ability to execute continuous
actions in an environment at 60Hz+, learning from prediction errors.
"""

from digital_cerebellum.micro_ops.engine import (
    MicroOpEngine,
    MicroOpConfig,
    StepResult,
    Environment,
)
from digital_cerebellum.micro_ops.screen_state_encoder import (
    ScreenStateEncoder,
    ScreenStateConfig,
    ROISpec,
)
from digital_cerebellum.micro_ops.gui_action_space import (
    GUIActionSpace,
    GUIActionSpaceConfig,
    GUIAction,
    GUIActionType,
)
from digital_cerebellum.micro_ops.aim_trainer import (
    AimTrainerEnv,
    AimTrainerConfig,
)
from digital_cerebellum.micro_ops.gui_controller import (
    GUIController,
    GUIControlConfig,
    CorrectionMicrozone,
    TrialResult,
)
from digital_cerebellum.micro_ops.openclaw_env import (
    OpenClawEnvironment,
    OpenClawEnvConfig,
)
from digital_cerebellum.micro_ops.tank_env import (
    TankBattleEnv,
    TankConfig,
    TankController,
)

__all__ = [
    "MicroOpEngine",
    "MicroOpConfig",
    "StepResult",
    "Environment",
    "ScreenStateEncoder",
    "ScreenStateConfig",
    "ROISpec",
    "GUIActionSpace",
    "GUIActionSpaceConfig",
    "GUIAction",
    "GUIActionType",
    "AimTrainerEnv",
    "AimTrainerConfig",
    "GUIController",
    "GUIControlConfig",
    "CorrectionMicrozone",
    "TrialResult",
    "OpenClawEnvironment",
    "OpenClawEnvConfig",
    "TankBattleEnv",
    "TankConfig",
    "TankController",
]
