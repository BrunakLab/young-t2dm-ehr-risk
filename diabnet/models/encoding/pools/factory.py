POOL_REGISTRY = {}
NO_POOL_ERR = "Pool {} not in POOL_REGISTRY! Available pools are {}"


def RegisterTrajectoryPooler(pool_name):
    """Registers a pool."""

    def decorator(f):
        POOL_REGISTRY[pool_name] = f
        return f

    return decorator


def get_trajectory_pooler(pool_name, **kwargs):
    """Get pool from POOL_REGISTRY based on pool_name."""

    if pool_name not in POOL_REGISTRY:
        raise Exception(NO_POOL_ERR.format(pool_name, POOL_REGISTRY.keys()))

    pool = POOL_REGISTRY[pool_name]

    return pool(**kwargs)
