RISK_CLASSIFIER_REGISTRY = {}
NO_RISK_CLASIFIER_ERR = (
    "RiskClassifier {} not in RISK_CLASSIFIER_REGISTRY! Available models are {} "
)


def RegisterClassifier(classifier_name):
    """Registers a configuration."""

    def decorator(f):
        RISK_CLASSIFIER_REGISTRY[classifier_name] = f
        return f

    return decorator


def get_classifier(risk_classifier_name, args, input_dim):
    """
    Get classifier from RISK_CLASSIFIER_REGISTRY based on risk_classifier_name
    Args:
        risk_classifier_name: Name of model, must exist in registry
        args: Arguments to pass to model

    Returns:
        risk_classifier: Instance of a Classifier
    """
    if risk_classifier_name not in RISK_CLASSIFIER_REGISTRY:
        raise Exception(
            NO_RISK_CLASIFIER_ERR.format(risk_classifier_name, RISK_CLASSIFIER_REGISTRY.keys())
        )
    risk_classifier = RISK_CLASSIFIER_REGISTRY[risk_classifier_name]
    return risk_classifier(args=args, input_dim=input_dim)
