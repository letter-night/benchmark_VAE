import argparse
import logging
import os

import numpy as np

from pythae.pipelines import TrainingPipeline
from pythae.trainers import BaseTrainerConfig

logger = logging.getLogger(__name__)
console = logging.StreamHandler()
logger.addHandler(console)
logger.setLevel(logging.INFO)

PATH = os.path.dirname(os.path.abspath(__file__))

ap = argparse.ArgumentParser()

# Training setting
ap.add_argument(
    "--dataset",
    type=str,
    default="mnist",
    choices=["mnist", "cifar10", "celeba"],
    help="The data set to use to perform training. It must be located in the folder 'data' at the "
    "path 'data/datset_name/' and contain a 'train_data.npz' and a 'eval_data.npz' file with the "
    "data being under the key 'data'. The data must be in the range [0-255] and shaped with the "
    "channel in first position (im_channel x height x width).",
    required=True,
)
ap.add_argument(
    "--model_name",
    help="The name of the model to train",
    choices=[
        "ae",
        "vae",
        "beta_vae",
        "wae",
        "vqvae",
        "vae_nf",
        "vae_iaf",
        "vae_lin_nf",
    ],
    required=True,
)
ap.add_argument(
    "--model_config",
    help="path to model config file (expected json file)",
    default=None,
)
ap.add_argument(
    "--nn",
    help="neural nets to use",
    default="convnet",
    choices=["default", "convnet", "resnet"],
)
ap.add_argument(
    "--training_config",
    help="path to training config_file (expected json file)",
    default=os.path.join(PATH, "configs/base_training_config.json"),
)
ap.add_argument(
    "--use_wandb",
    help="whether to log the metrics in wandb",
    action="store_true",
)
ap.add_argument(
    "--wandb_project",
    help="wandb project name",
    default="test-project",
)
ap.add_argument(
    "--wandb_entity",
    help="wandb entity name",
    default="benchmark_team",
)

args = ap.parse_args()


def main(args):

    if args.dataset == "mnist":

        if args.nn == "convnet":

            from pythae.models.nn.benchmarks.mnist import (
                Decoder_Conv_AE_MNIST as Decoder_AE,
            )
            from pythae.models.nn.benchmarks.mnist import (
                Decoder_Conv_AE_MNIST as Decoder_VQVAE,
            )
            from pythae.models.nn.benchmarks.mnist import (
                Encoder_Conv_AE_MNIST as Encoder_AE,
            )
            from pythae.models.nn.benchmarks.mnist import (
                Encoder_Conv_AE_MNIST as Encoder_VQVAE,
            )
            from pythae.models.nn.benchmarks.mnist import (
                Encoder_Conv_VAE_MNIST as Encoder_VAE,
            )

        elif args.nn == "resnet":
            from pythae.models.nn.benchmarks.mnist import (
                Encoder_ResNet_AE_MNIST as Encoder_AE,
            )
            from pythae.models.nn.benchmarks.mnist import (
                Encoder_ResNet_VAE_MNIST as Encoder_VAE,
            )
            from pythae.models.nn.benchmarks.mnist import (
                Encoder_ResNet_VQVAE_MNIST as Encoder_VQVAE,
            )
            from pythae.models.nn.benchmarks.mnist import (
                Decoder_ResNet_AE_MNIST as Decoder_AE,
            )
            from pythae.models.nn.benchmarks.mnist import (
                Decoder_ResNet_VQVAE_MNIST as Decoder_VQVAE,
            )

    elif args.dataset == "cifar10":

        if args.nn == "convnet":

            from pythae.models.nn.benchmarks.cifar import (
                Decoder_Conv_AE_CIFAR as Decoder_AE,
            )
            from pythae.models.nn.benchmarks.cifar import (
                Decoder_Conv_AE_CIFAR as Decoder_VQVAE,
            )
            from pythae.models.nn.benchmarks.cifar import (
                Encoder_Conv_AE_CIFAR as Encoder_AE,
            )
            from pythae.models.nn.benchmarks.cifar import (
                Encoder_Conv_AE_CIFAR as Encoder_VQVAE,
            )
            from pythae.models.nn.benchmarks.cifar import (
                Encoder_Conv_VAE_CIFAR as Encoder_VAE,
            )

        elif args.nn == "resnet":
            from pythae.models.nn.benchmarks.cifar import (
                Decoder_ResNet_AE_CIFAR as Decoder_AE,
            )
            from pythae.models.nn.benchmarks.cifar import (
                Decoder_ResNet_VQVAE_CIFAR as Decoder_VQVAE,
            )
            from pythae.models.nn.benchmarks.cifar import (
                Encoder_ResNet_AE_CIFAR as Encoder_AE,
            )
            from pythae.models.nn.benchmarks.cifar import (
                Encoder_ResNet_VAE_CIFAR as Encoder_VAE,
            )
            from pythae.models.nn.benchmarks.cifar import (
                Encoder_ResNet_VQVAE_CIFAR as Encoder_VQVAE,
            )

    elif args.dataset == "celeba":

        if args.nn == "convnet":

            from pythae.models.nn.benchmarks.celeba import (
                Decoder_Conv_AE_CELEBA as Decoder_AE,
            )
            from pythae.models.nn.benchmarks.celeba import (
                Decoder_Conv_AE_CELEBA as Decoder_VQVAE,
            )
            from pythae.models.nn.benchmarks.celeba import (
                Encoder_Conv_AE_CELEBA as Encoder_AE,
            )
            from pythae.models.nn.benchmarks.celeba import (
                Encoder_Conv_AE_CELEBA as Encoder_VQVAE,
            )
            from pythae.models.nn.benchmarks.celeba import (
                Encoder_Conv_VAE_CELEBA as Encoder_VAE,
            )

        elif args.nn == "resnet":
            from pythae.models.nn.benchmarks.celeba import (
                Decoder_ResNet_AE_CELEBA as Decoder_AE,
            )
            from pythae.models.nn.benchmarks.celeba import (
                Decoder_ResNet_VQVAE_CELEBA as Decoder_VQVAE,
            )
            from pythae.models.nn.benchmarks.celeba import (
                Encoder_ResNet_AE_CELEBA as Encoder_AE,
            )
            from pythae.models.nn.benchmarks.celeba import (
                Encoder_ResNet_VAE_CELEBA as Encoder_VAE,
            )
            from pythae.models.nn.benchmarks.celeba import (
                Encoder_ResNet_VQVAE_CELEBA as Encoder_VQVAE,
            )

    try:
        logger.info(f"\nLoading {args.dataset} data...\n")
        train_data = (
            np.load(os.path.join(PATH, f"data/{args.dataset}", "train_data.npz"))[
                "data"
            ]
            / 255.0
        )
        eval_data = (
            np.load(os.path.join(PATH, f"data/{args.dataset}", "eval_data.npz"))["data"]
            / 255.0
        )
    except Exception as e:
        raise FileNotFoundError(
            f"Unable to load the data from 'data/{args.dataset}' folder. Please check that both a "
            "'train_data.npz' and 'eval_data.npz' are present in the folder.\n Data must be "
            " under the key 'data', in the range [0-255] and shaped with channel in first "
            "position\n"
            f"Exception raised: {type(e)} with message: " + str(e)
        ) from e

    logger.info("Successfully loaded data !\n")
    logger.info("------------------------------------------------------------")
    logger.info("Dataset \t \t Shape \t \t \t Range")
    logger.info(
        f"{args.dataset.upper()} train data: \t {train_data.shape} \t [{train_data.min()}-{train_data.max()}] "
    )
    logger.info(
        f"{args.dataset.upper()} eval data: \t {eval_data.shape} \t [{eval_data.min()}-{eval_data.max()}]"
    )
    logger.info("------------------------------------------------------------\n")

    data_input_dim = tuple(train_data.shape[1:])

    if args.model_name == "ae":
        from pythae.models import AE, AEConfig

        if args.model_config is not None:
            model_config = AEConfig.from_json_file(args.model_config)

        else:
            model_config = AEConfig()

        model_config.input_dim = data_input_dim

        model = AE(
            model_config=model_config,
            encoder=Encoder_AE(model_config),
            decoder=Decoder_AE(model_config),
        )

    elif args.model_name == "vae":
        from pythae.models import VAE, VAEConfig

        if args.model_config is not None:
            model_config = VAEConfig.from_json_file(args.model_config)

        else:
            model_config = VAEConfig()

        model_config.input_dim = data_input_dim

        model = VAE(
            model_config=model_config,
            encoder=Encoder_VAE(model_config),
            decoder=Decoder_AE(model_config),
        )

    elif args.model_name == "wae":
        from pythae.models import WAE_MMD, WAE_MMD_Config

        if args.model_config is not None:
            model_config = WAE_MMD_Config.from_json_file(args.model_config)

        else:
            model_config = WAE_MMD_Config()

        model_config.input_dim = data_input_dim

        model = WAE_MMD(
            model_config=model_config,
            encoder=Encoder_AE(model_config),
            decoder=Decoder_AE(model_config),
        )

    elif args.model_name == "beta_vae":
        from pythae.models import BetaVAE, BetaVAEConfig

        if args.model_config is not None:
            model_config = BetaVAEConfig.from_json_file(args.model_config)

        else:
            model_config = BetaVAEConfig()

        model_config.input_dim = data_input_dim

        model = BetaVAE(
            model_config=model_config,
            encoder=Encoder_VAE(model_config),
            decoder=Decoder_AE(model_config),
        )

    elif args.model_name == "vqvae":
        from pythae.models import VQVAE, VQVAEConfig

        if args.model_config is not None:
            model_config = VQVAEConfig.from_json_file(args.model_config)

        else:
            model_config = VQVAEConfig()

        model_config.input_dim = data_input_dim

        model = VQVAE(
            model_config=model_config,
            encoder=Encoder_VQVAE(model_config),
            decoder=Decoder_VQVAE(model_config),
        )

    elif args.model_name == "vae_iaf":
        from pythae.models import VAE_IAF, VAE_IAF_Config

        if args.model_config is not None:
            model_config = VAE_IAF_Config.from_json_file(args.model_config)

        else:
            model_config = VAE_IAF_Config()

        model_config.input_dim = data_input_dim

        model = VAE_IAF(
            model_config=model_config,
            encoder=Encoder_VAE(model_config),
            decoder=Decoder_AE(model_config),
        )

    elif args.model_name == "vae_lin_nf":
        from pythae.models import VAE_LinNF, VAE_LinNF_Config

        if args.model_config is not None:
            model_config = VAE_LinNF_Config.from_json_file(args.model_config)

        else:
            model_config = VAE_LinNF_Config()

        model_config.input_dim = data_input_dim

        model = VAE_LinNF(
            model_config=model_config,
            encoder=Encoder_VAE(model_config),
            decoder=Decoder_AE(model_config),
        )

        print(model)

    logger.info(f"Successfully build {args.model_name.upper()} model !\n")

    encoder_num_param = sum(
        p.numel() for p in model.encoder.parameters() if p.requires_grad
    )
    decoder_num_param = sum(
        p.numel() for p in model.decoder.parameters() if p.requires_grad
    )
    total_num_param = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        "----------------------------------------------------------------------"
    )
    logger.info("Model \t Encoder params \t Decoder params \t Total params")
    logger.info(
        f"{args.model_name.upper()} \t {encoder_num_param} \t \t {decoder_num_param}"
        f" \t \t {total_num_param}"
    )
    logger.info(
        "----------------------------------------------------------------------\n"
    )

    logger.info(f"Model config of {args.model_name.upper()}: {model_config}\n")

    training_config = BaseTrainerConfig.from_json_file(args.training_config)

    logger.info(f"Training config: {training_config}\n")

    callbacks = []

    if args.use_wandb:
        from pythae.trainers.training_callbacks import WandbCallback

        wandb_cb = WandbCallback()
        wandb_cb.setup(
            training_config,
            model_config=model_config,
            project_name=args.wandb_project,
            entity_name=args.wandb_entity,
        )

        callbacks.append(wandb_cb)

    pipeline = TrainingPipeline(training_config=training_config, model=model)

    pipeline(train_data=train_data, eval_data=eval_data, callbacks=callbacks)


if __name__ == "__main__":

    main(args)