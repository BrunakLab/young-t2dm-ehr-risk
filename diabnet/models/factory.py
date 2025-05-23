RISK_MODEL_REGISTRY = {}
NO_RISK_MODEL_ERR = "Risk model {} not in RISK_MODEL_REGISTRY! Available risk models are {}"


class ModelName:
    BagOfWords = "BagOfWords"
    MultiRegistryTrajectory = "MultiTrajectory"
    SingleRegistryTrajectory = "SingleTrajectory"


def RegisterRiskModel(model_name):
    """Registers a risk_model."""

    def decorator(f):
        RISK_MODEL_REGISTRY[model_name] = f
        return f

    return decorator


def get_risk_model(risk_model_name, **kwargs):
    """Get risk_model from RISK_MODEL_REGISTRY based on risk_model_name.

    Inputs:
        risk_model_name: String matching an implemented Trajectory Risk Model
        kwargs: Arguments to pass to the Trajectory Risk Model
    Outputs:
        risk_model: TrajectoryRiskModel
    """

    if risk_model_name not in RISK_MODEL_REGISTRY:
        raise Exception(NO_RISK_MODEL_ERR.format(risk_model_name, RISK_MODEL_REGISTRY.keys()))

    risk_model = RISK_MODEL_REGISTRY[risk_model_name]

    return risk_model(**kwargs)
