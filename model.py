import segmentation_models_pytorch as smp

def get_model(model_name="Unet", encoder_name="resnet34", encoder_weights="imagenet", in_channels=3, classes=1):

    if model_name == 'Unet':
        model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            classes=classes,
            activation=None,
        )
    elif model_name == 'FPN':
        model = smp.FPN(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            classes=classes,
            activation=None,
        )
    elif model_name == 'Linknet':
        model = smp.Linknet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            classes=classes,
            activation=None,
        )
    elif model_name == 'PSPNet':
        model = smp.PSPNet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            classes=classes,
            activation=None,
        )
    else:
        raise ValueError(f"Unsupported model name: {model_name}")
    
    return model