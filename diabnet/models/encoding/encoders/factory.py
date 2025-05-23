TRAJECTORY_ENCODER_REGISTRY = {}
NO_TRAJECTORY_ENCODER_ERR = (
    "Trajectory encoder {} not in TRAJECTORY_ENCODER_REGISTRY! Available encoders are {} "
)


def RegisterTrajectoryEncoder(trajectory_encoder_name):
    """Registers a configuration."""

    def decorator(f):
        TRAJECTORY_ENCODER_REGISTRY[trajectory_encoder_name] = f
        return f

    return decorator


def get_trajectory_encoder(trajectory_encoder_name, args):
    """
    Get model from MODEL_REGISTRY based on args.encoder
    Args:
        name: Name of model, must exit in registry
        args: Arguments to pass to model

    Returns:
        model
    """
    if trajectory_encoder_name not in TRAJECTORY_ENCODER_REGISTRY:
        raise Exception(
            NO_TRAJECTORY_ENCODER_ERR.format(
                trajectory_encoder_name, TRAJECTORY_ENCODER_REGISTRY.keys()
            )
        )
    trajectory_encoder = TRAJECTORY_ENCODER_REGISTRY[trajectory_encoder_name]
    return trajectory_encoder(args)
