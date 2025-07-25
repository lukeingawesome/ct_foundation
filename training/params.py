import argparse
import os

def get_default_params(model_name):
    # Params from paper (https://arxiv.org/pdf/2103.00020.pdf)
    model_name = model_name.lower()
    if "vit" in model_name:
        return {"lr": 5.0e-4, "beta1": 0.9, "beta2": 0.98, "eps": 1.0e-6}
    else:
        return {"lr": 5.0e-4, "beta1": 0.9, "beta2": 0.999, "eps": 1.0e-8}


def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train-data",
        type=str,
        default=None,
        help="Path to csv file with training data",
    )
    parser.add_argument(
        "--train-data-list",
        type=str,
        default=None,
        help="Path to list of csv files with training data",
    )
    parser.add_argument(
        "--train-data-file",
        type=str,
        default=None,
        help="Path to list of json files with training data",
    )
    parser.add_argument(
        "--eval-data-file",
        type=str,
        default=None,
        help="Path to list of json files with eval data",
    )
    parser.add_argument(
        "--val-data",
        type=str,
        default=None,
        help="Path to csv file with validation data",
    )
    parser.add_argument(
        "--train-num-samples",
        type=int,
        default=None,
        help="Number of samples in dataset. Required for webdataset if not available in info file",
    )
    parser.add_argument(
        "--zeroshot-data",
        type=str,
        default=None,
        help="Path to csv file with zeroshot data",
    )
    parser.add_argument(
        '--train-num-samples-list', type=int, nargs='+', default=[],
        help='Number of samples in list of datasets. Required for webdataset while using multiple datasets')
    parser.add_argument(
        "--val-num-samples",
        type=int,
        default=None,
        help="Number of samples in dataset. Useful for webdataset if not available in info file.",
    )
    parser.add_argument(
        "--keep-image-id",
        type=bool,
        default=False,
        help="Keep image id in output data."
    )
    parser.add_argument(
        "--dataset-type",
        choices=["cxr", "ct"],
        default="ct",
        help="Which type of dataset to process."
    )
    parser.add_argument(
        "--dataset-type-list",
        type=str,
        default=None,
        help="Which types of list of datasets to process."
    )
    parser.add_argument(
        "--img-root",
        type=str,
        default="filepath",
        help="For json-like datasets, the name of the image paths."
    )
    parser.add_argument(
        "--flickr-img-root",
        type=str,
        default="",
        help="For retrieval dataset of flickr, the name of the image paths."
    )
    parser.add_argument(
        "--flickr-test-file",
        type=str,
        default="",
        help="For retrieval dataset of flickr, the name of the json paths."
    )
    parser.add_argument(
        "--flickr-text-features",
        type=str,
        default="",
        help="For retrieval dataset of flickr, the name of the features paths."
    )
    parser.add_argument(
        "--coco-img-root",
        type=str,
        default="",
        help="For retrieval dataset of coco, the name of the image paths."
    )
    parser.add_argument(
        "--coco-test-file",
        type=str,
        default="",
        help="For retrieval dataset of coco, the name of the json paths."
    )
    parser.add_argument(
        "--coco-text-features",
        type=str,
        default="",
        help="For retrieval dataset of coco, the name of the features paths."
    )
    parser.add_argument(
        "--dataset-resampled",
        default=False,
        action="store_true",
        help="Whether to use sampling with replacement for webdataset shard selection."
    )
    parser.add_argument(
        "--csv-separator",
        type=str,
        default="\t",
        help="For csv-like datasets, which separator to use."
    )
    parser.add_argument(
        "--csv-img-key",
        type=str,
        default="filepath",
        help="For csv-like datasets, the name of the key for the image paths."
    )
    parser.add_argument(
        "--csv-caption-key",
        type=str,
        default="title",
        help="For csv-like datasets, the name of the key for the captions."
    )
    parser.add_argument(
        "--text-separator",
        type=str,
        default=" [SEP] ",
        help="For text processing, the separator used to split text for embed_mask creation."
    )
    parser.add_argument(
        "--split-column",
        type=str,
        default="split",
        help="For csv-like datasets, the name of the column for data splits."
    )
    parser.add_argument(
        "--train-split",
        type=str,
        default="train",
        help="For csv-like datasets, the value in split column for training data."
    )
    parser.add_argument(
        "--val-split",
        type=str,
        default="val",
        help="For csv-like datasets, the value in split column for validation data."
    )
    parser.add_argument(
        "--imagenet-val",
        type=str,
        default=None,
        help="Path to imagenet val set for conducting zero shot evaluation.",
    )
    parser.add_argument(
        "--imagenet-classname-feautres",
        type=str,
        default=None,
        help="Path to imagenet classname features for conducting zero shot evaluation.",
    )
    parser.add_argument(
        "--imagenet-val-text",
        type=str,
        default=None,
        help="Path to imagenet val set for conducting zero shot evaluation.",
    )
    parser.add_argument(
        "--imagenet-v2",
        type=str,
        default=None,
        help="Path to imagenet v2 for conducting zero shot evaluation.",
    )
    parser.add_argument(
        "--logs",
        type=str,
        default="/opt/project/logs",
        help="Where to store tensorboard logs. Use None to avoid storing logs.",
    )
    parser.add_argument(
        "--log-local",
        action="store_true",
        default=False,
        help="log files on local master, otherwise global master only.",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Optional identifier for the experiment when storing logs. Otherwise use current time.",
    )
    parser.add_argument(
        "--workers", type=int, default=4, help="Number of dataloader workers per GPU."
    )
    parser.add_argument(
        "--batch-size", type=int, default=64, help="Batch size per GPU."
    )
    parser.add_argument(
        "--eval-batch-size", type=int, default=64, help="Batch size per GPU for eval."
    )
    parser.add_argument(
        "--epochs", type=int, default=32, help="Number of epochs to train for."
    )
    parser.add_argument("--lr", type=float, default=None, help="Learning rate.")
    parser.add_argument("--text-lr", type=float, default=None, help="Learning rate of text encoder.")
    parser.add_argument("--visual-lr", type=float, default=None, help="Learning rate of visual encoder.")

    parser.add_argument("--beta1", type=float, default=None, help="Adam beta 1.")
    parser.add_argument("--beta2", type=float, default=None, help="Adam beta 2.")
    parser.add_argument("--eps", type=float, default=None, help="Adam epsilon.")

    parser.add_argument("--wd", type=float, default=0.2, help="Weight decay.")
    parser.add_argument("--text-wd", type=float, default=None, help="Weight decay of text encoder.")
    parser.add_argument("--visual-wd", type=float, default=None, help="Weight decay of visual encoder.")

    parser.add_argument("--ld", type=float, default=1.0, help="Learning rate Layer decay.")
    parser.add_argument("--text-ld", type=float, default=None, help="Learning rate Layer decay of text encoder.")
    parser.add_argument("--visual-ld", type=float, default=None, help="Learning rate Layer decay of visual encoder.")
    
    parser.add_argument(
        "--warmup", type=int, default=10000, help="Number of steps to warmup for."
    )
    parser.add_argument(
        "--use-bn-sync",
        default=False,
        action="store_true",
        help="Whether to use batch norm sync.")
    parser.add_argument(
        "--skip-scheduler",
        action="store_true",
        default=False,
        help="Use this flag to skip the learning rate decay.",
    )
    parser.add_argument(
        "--save-frequency", type=int, default=5, help="How often to save checkpoints."
    )
    parser.add_argument(
        "--save-most-recent",
        action="store_true",
        default=False,
        help="Always save the most recent model trained to epoch_latest.pt.",
    )
    parser.add_argument(
        "--zeroshot-frequency", type=int, default=2, help="How often to run zero shot."
    )
    parser.add_argument(
        "--val-frequency", type=int, default=1, help="How often to run evaluation with val data."
    )
    parser.add_argument(
        "--resume",
        default=None,
        type=str,
        help="path to latest checkpoint (default: none)",
    )
    parser.add_argument(
        "--precision",
        choices=["amp", "amp_bf16", "amp_bfloat16", "bf16", "fp16", "fp32"],
        default="amp",
        help="Floating point precision."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="EVA02-CLIP-B-16",
        help="Name of the vision backbone to use.",
    )
    parser.add_argument(
        "--pretrained",
        default='',
        type=str,
        help="Use a pretrained CLIP model weights with the specified tag or file path.",
    )
    parser.add_argument(
        "--pretrained-image",
        default='',
        type=str,
        help="Load pretrained image model weights for image tower backbone if available.",
    )
    parser.add_argument(
        "--pretrained-visual-model",
        default=None,
        help="pretrained model.visual."
    )
    parser.add_argument(
        "--pretrained-text",
        default='',
        type=str,
        help="Load pretrained text model weights for text tower backbone if available.",
    )
    parser.add_argument(
        "--pretrained-text-model",
        default=None,
        help="pretrained model.text."
    )
    parser.add_argument(
        "--text-base",
        type=str,
        default="microsoft/DialoGPT-medium",
        help="Base model name for text encoder."
    )
    parser.add_argument(
        "--model-pth",
        type=str,
        default=None,
        help="Path to model checkpoint file."
    )
    parser.add_argument(
        "--llm2vec-path",
        type=str,
        default="/data/research/tmp/checkpoint-12600/",
        help="Path to the separate text encoder model."
    )
    parser.add_argument(
        '--skip-list', type=str, nargs='+', default=[],
        help='skip list of weights to load')

    parser.add_argument('--enable-kd', action='store_true', default=False)
    parser.add_argument(
        "--kd-model",
        type=str,
        default="ViT-L-14",
        help="Name of the vision backbone in teacher model to use.",
    )
    parser.add_argument(
        "--kd-pretrained",
        default='',
        type=str,
        help="Use a pretrained model weights with the specified tag or file path (teacher model).",
    )
    parser.add_argument(
        "--kd-tower",
        choices=["text", "image", "both"],
        default="image",
        help="Type of tower to distill."
    )

    parser.add_argument(
        "--kd-loss",
        choices=["mse", "clip"],
        default="mse",
        help="Type of loss in distillation."
    )

    parser.add_argument(
        "--lock-image",
        default=False,
        action='store_true',
        help="Lock full image tower by disabling gradients.",
    )
    parser.add_argument(
        "--lora",
        default=False,
        action='store_true',
        help="Use lora to image tower",
    )
    parser.add_argument(
        "--lock-image-unlocked-groups",
        type=int,
        default=0,
        help="Leave last n image tower layer groups unlocked.",
    )
    parser.add_argument(
        "--lock-image-freeze-bn-stats",
        default=False,
        action='store_true',
        help="Freeze BatchNorm running stats in image tower for any locked layers.",
    )
    parser.add_argument(
        '--image-mean', type=float, nargs='+', default=None, metavar='MEAN',
        help='Override default image mean value of dataset')
    parser.add_argument(
        '--image-std', type=float, nargs='+', default=None, metavar='STD',
        help='Override default image std deviation of of dataset')
    parser.add_argument(
        "--grad-checkpointing",
        default=False,
        action='store_true',
        help="Enable gradient checkpointing.",
    )
    parser.add_argument(
        "--local-loss",
        default=False,
        action="store_true",
        help="calculate loss w/ local features @ global (instead of realizing full global @ global matrix)"
    )
    parser.add_argument(
        "--gather-with-grad",
        default=False,
        action="store_true",
        help="enable full distributed gradient for feature gather"
    )
    parser.add_argument(
        "--force-quick-gelu",
        default=False,
        action='store_true',
        help="Force use of QuickGELU activation for non-OpenAI transformer models.",
    )
    parser.add_argument(
        "--force-patch-dropout",
        default=None,
        type=float,
        help="Override the patch dropout during training, for fine tuning with no dropout near the end as in the paper",
    )
    parser.add_argument(
        "--force-custom-clip",
        default=False,
        action='store_true',
        help="Force use of CustomCLIP model (separate text-tower).",
    )
    parser.add_argument(
        "--torchscript",
        default=False,
        action='store_true',
        help="torch.jit.script the model, also uses jit version of OpenAI models if pretrained=='openai'",
    )
    parser.add_argument(
        "--trace",
        default=False,
        action='store_true',
        help="torch.jit.trace the model for inference / eval only",
    )
    # arguments for distributed training
    parser.add_argument(
        "--dist-url",
        default="env://",
        type=str,
        help="url used to set up distributed training",
    )
    parser.add_argument(
        "--dist-backend", default="nccl", type=str, help="distributed backend"
    )
    parser.add_argument(
        "--report-to",
        default='',
        type=str,
        help="Options are ['wandb', 'tensorboard', 'wandb,tensorboard']"
    )
    parser.add_argument(
        "--wandb-notes",
        default='',
        type=str,
        help="Notes if logging with wandb"
    )
    parser.add_argument(
        "--debug",
        default=False,
        action="store_true",
        help="If true, more information is logged."
    )
    parser.add_argument(
        "--copy-codebase",
        default=False,
        action="store_true",
        help="If true, we copy the entire base on the log diretory, and execute from there."
    )

    parser.add_argument(
        "--ddp-static-graph",
        default=False,
        action='store_true',
        help="Enable static graph optimization for DDP in PyTorch >= 1.11.",
    )
    parser.add_argument(
        "--no-set-device-rank",
        default=False,
        action="store_true",
        help="Don't set device index from local rank (when CUDA_VISIBLE_DEVICES restricted to one per proc)."
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Default random seed."
    )
    parser.add_argument(
        "--grad-clip-norm", type=float, default=None, help="Gradient clip."
    )
    parser.add_argument(
        "--grad-accumulation-steps", type=int, default=16, help="Gradient accumulation steps for both regular training and DeepSpeed."
    )
    parser.add_argument(
        "--lock-text",
        default=False,
        action='store_true',
        help="Lock full text tower by disabling gradients.",
    )
    parser.add_argument(
        "--lock-text-unlocked-layers",
        type=int,
        default=0,
        help="Leave last n image tower layer groups unlocked.",
    )
    parser.add_argument(
        "--lock-text-freeze-layer-norm",
        default=False,
        action='store_true',
        help="Freeze BatchNorm running stats in image tower for any locked layers.",
    )

    parser.add_argument(
        "--log-every-n-steps",
        type=int,
        default=100,
        help="Log every n steps to tensorboard/console/wandb.",
    )

    parser.add_argument(
        "--wandb-project-name",
        type=str,
        default='open-clip',
        help="Name of the project if logging with wandb.",
    )

    parser.add_argument('--smoothing', type=float, default=0, help='Label smoothing (default: 0.)')
    parser.add_argument('--cache-dir', type=str, default=None, help='Default cache dir to cache model checkpoint.')
    parser.add_argument('--optimizer', type=str, default='adamw', help='Default optimizer.')

    parser.add_argument('--enable-deepspeed', action='store_true', default=False)
    parser.add_argument('--zero-stage', type=int, default=1, help='stage of ZERO')

    parser.add_argument('--extract-features', action='store_true', default=False, help='Extract features of text/image with val data.')
    parser.add_argument(
        "--img-emb-path",
        type=str,
        default=None,
        help="Path to store image embeddings",
    )

    parser.add_argument(
        "--text-emb-path",
        type=str,
        default=None,
        help="Path to store text embeddings",
    )
    parser.add_argument(
        "--img-feature-path",
        type=str,
        default=None,
        help="Path to load image embeddings",
    )

    parser.add_argument(
        "--text-feature-path",
        type=str,
        default=None,
        help="Path to load text embeddings",
    )

    parser.add_argument(
        "--text-id-path",
        type=str,
        default=None,
        help="Path to store text input ids",
    )

    parser.add_argument(
        "--text-path",
        type=str,
        default=None,
        help="Path to store raw text",
    )

    parser.add_argument(
        "--save-interval",
        type=int,
        default=100,
        help="Step interval to store embeddings",
    )

    parser.add_argument('--projection-lr', type=float, default=None,
                       help='Learning rate for projection layers (overrides --lr)')

    args = parser.parse_args(args)

    if args.cache_dir is not None:
        os.environ['TRANSFORMERS_CACHE'] = args.cache_dir  # huggingface model dir

    # If some params are not passed, we use the default values based on model name.
    default_params = get_default_params(args.model)
    for name, val in default_params.items():
        if getattr(args, name) is None:
            setattr(args, name, val)

    if args.enable_deepspeed:
        try:
            import deepspeed
            from deepspeed import DeepSpeedConfig
            os.environ['ENV_TYPE'] = "deepspeed"
            parser = deepspeed.add_config_arguments(parser)
            ds_init = deepspeed.initialize
        except:
            print("Please 'pip install deepspeed==0.8.1'")
            exit(0)
    else:
        os.environ['ENV_TYPE'] = "pytorch"
        ds_init = None

    return args, ds_init
