import os
import argparse
from datetime import datetime
import logging
from termcolor import colored
import pprint


supported_models = [
            './modelzoo/llama-2/llama-2-7b',
            #'./modelzoo/llama-2/llama-2-13b',
            './modelzoo/llama-2/llama-2-70b',
            './modelzoo/llama-2-hf/llama-2-7b-hf',
            #'./modelzoo/llama-2-hf/llama-2-13b-hf',
            './modelzoo/llama-2-hf/llama-2-70b-hf',
            './modelzoo/llama-3/llama-3-8b',
            './modelzoo/llama-3/llama-3-70b',
            './modelzoo/llama-3.1/llama-3.1-8b',
            './modelzoo/llama-3.1/llama-3.1-70b',
            './modelzoo/llama-3.1-instruct/llama-3.1-8b-instruct',
            './modelzoo/llama-3.3-instruct/llama-3.3-70b-instruct',
            './modelzoo/llama-3-instruct/llama-3-8b-instruct',
            './modelzoo/llama-3-instruct/llama-3-8b-instruct',
            './modelzoo/qwen-2.5-instruct/qwen-2.5-7b-instruct',
            './modelzoo/qwen-2.5-instruct/qwen-2.5-32b-instruct',
            ]
# supported_models = [
#             'meta-llama/Llama-2-7b-hf',
#             'meta-llama/Llama-2-13b-hf',
#             'meta-llama/Llama-2-70b-hf',
#             'meta-llama/Meta-Llama-3-8B',
#             'meta-llama/Meta-Llama-3-70B',
#             'meta-llama/Llama-3.1-8B', 
#             'meta-llama/Llama-3.1-70B', 
#             'meta-llama/Llama-3.1-8B-Instruct', 
#             'Qwen/Qwen2.5-32B-Instruct', 
#             'Qwen/Qwen2.5-7B-Instruct', 
#             ]
supported_datasets = ['wikitext2', 'c4', 'pile']


def parser_gen():
    parser = argparse.ArgumentParser()

    # General Arguments
    parser.add_argument('--model', type=str, default='meta-llama/Llama-2-7b-hf',
                        help='Model to load.', choices=supported_models)
    parser.add_argument('--seed', type=int, default=0, help='Random seed for HuggingFace and PyTorch.')
    parser.add_argument('--hf_token', type=str, default=None, help='HuggingFace token for model access.')

    # Activation Quantization Arguments
    parser.add_argument('--a_bits', type=int, default=16,
                        help='''Number of bits for inputs of the linear layers.
                                This applies to all linear layers in the model, including down-projection and out-projection.''')
    parser.add_argument('--a_groupsize', type=int, default=-1, 
                        help='Groupsize for activation quantization. Note that this should be the same as w_groupsize.')
    parser.add_argument('--a_asym', action="store_true", default=False,
                        help='Use asymmetric activation quantization.')

    # Weight Quantization Arguments
    parser.add_argument('--w_bits', type=int, default=16, 
                        help='Number of bits for weights of the linear layers.')
    parser.add_argument('--w_groupsize', type=int, default=-1, 
                        help='Groupsize for weight quantization. Note that this should be the same as a_groupsize.')
    parser.add_argument('--w_asym', action="store_true", default=False,
                        help='Use asymmetric weight quantization.')
    parser.add_argument('--gptq', action="store_true", default=False,
                        help='Quantize the weights using GPTQ. If w_bits < 16 and this flag is not set, use RtN.')
    parser.add_argument('--gptq_mse', action="store_true", default=False,
                        help='''Use MSE search to find the optimal clipping threshold for weight quantization. 
                                NOTE: Do not activate while using LWC.''')
    parser.add_argument('--percdamp', type=float, default=.01,
                        help='Percent of the average Hessian diagonal to use for dampening.')
    parser.add_argument('--act_order', action="store_true", default=False,
                        help='Use act-order in GPTQ.')

    # FlatQuant calibration Arguments
    parser.add_argument('--epochs', type=int, default=15, help='Number of training epochs.')
    parser.add_argument('--cali_dataset', type=str, default='wikitext2',
                        help='Calibration dataset for FlatQuant and GPTQ.', choices=supported_datasets)
    parser.add_argument('--nsamples', type=int, default=128,
                        help='Number of calibration data samples for FlatQuant and GPTQ.')
    parser.add_argument('--cali_bsz', type=int, default=4,
                        help='Batch size for FlatQuant. Default is 4.')
    parser.add_argument("--flat_lr", type=float, default=1e-5, 
                        help='Learning rate for learnable transformation.')
    parser.add_argument("--cali_trans", default=False, action="store_true", 
                        help="Enable calibration of transformations.")
    parser.add_argument("--add_diag", default=False, action="store_true", 
                        help="Add per-channel scaling.")
    parser.add_argument("--lwc", default=False, action="store_true", 
                        help="Use learnable weight clipping.")
    parser.add_argument("--lac", default=False, action="store_true", 
                        help="Use learnable activation clipping.")
    parser.add_argument('--resume', action="store_true", default=False, 
                        help='Resume from a previous checkpoint for evaluation.')
    parser.add_argument('--save_matrix', action="store_true", default=False, 
                        help='Save the matrix-style parameters of FlatQuant.')
    parser.add_argument('--reload_matrix', action="store_true", default=False, 
                        help='Reload matrices and the inverse matrices for evaluation.')
    parser.add_argument('--matrix_path', type=str, default=None,
                        help='Path to the pre-trained matrix-style parameters of FlatQuant.')
    parser.add_argument("--diag_init", type=str, default="sq_style", choices=["sq_style", "one_style"], 
                        help='The way to initialize per-channel scaling. Default is SmoothQuant style.')
    parser.add_argument("--diag_alpha", type=float, default=0.3, 
                        help='Hyperparameter for the SmoothQuant style initialization of per-channel scaling.')
    parser.add_argument("--warmup", default=False, action="store_true", help="Warm up the learning rate during training.")
    parser.add_argument("--deactive_amp", default=False, action="store_true", help="Disable AMP training.")
    parser.add_argument("--direct_inv", default=False, action="store_true", 
                        help="Use the inverse method in PyTorch to directly get the inverse matrix rather than SVD.")
    parser.add_argument("--separate_vtrans", default=False, action="store_true", 
                        help="Disable the integration of the vtrans transformation.")
    
    # KV-Cache Quantization Arguments
    parser.add_argument('--q_bits', type=int, default=16,
                        help='''Number of bits for queries quantization. 
                        Note that quantizing the queries needs another rotation for the keys/queries.''')
    parser.add_argument('--q_asym', action="store_true", default=False, 
                        help='Use asymmetric quantization for queries.')
    parser.add_argument('--q_groupsize', type=int, default=-1)

    parser.add_argument('--k_bits', type=int, default=16,
                        help='''Number of bits for K-cache quantization.
                        Note that quantizing the K-cache needs another rotation for the keys/queries.''')
    parser.add_argument('--k_asym', action="store_true", default=False, 
                        help='Use asymmetric quantization for K-cache.')
    parser.add_argument('--k_groupsize', type=int, default=-1, 
                    help='Groupsize for K-cache quantization.')

    parser.add_argument('--v_bits', type=int, default=16,
                        help='Number of bits for V-cache quantization.')
    parser.add_argument('--v_asym', action="store_true", default=False,
                        help='Use asymmetric quantization for V-cache.')
    parser.add_argument('--v_groupsize', type=int, default=-1)
    
    # Experiments Arguments
    parser.add_argument("--output_dir", type=str, default="./outputs", help="Output directory path.")
    parser.add_argument("--exp_name", type=str, default="exp", help="Experiment name.")

    # LM Eval Arguments
    parser.add_argument("--lm_eval", action="store_true", help="Evaluate the model on LM Eval tasks.")
    parser.add_argument(
        '--tasks',
        nargs='+',
        default=["piqa", "hellaswag", "arc_easy", "arc_challenge", "winogrande", "lambada_openai"],
        help='Tasks to evaluate on LM Eval.')
    parser.add_argument('--lm_eval_batch_size', type=int, default=128, help='Batch size for evaluation with lm eval harness.')
    parser.add_argument(
        "--distribute_model",
        action="store_true",
        help="Distribute the model across multiple GPUs for evaluation.")

    # Add quantized_save flag
    parser.add_argument('--quantized_save', action = "store_true", default = False,
                        help = 'Save the quantized model checkpoint.')

    args = parser.parse_args()
    if args.a_groupsize > -1:
        raise NotImplementedError
    
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    args.quantize = (args.w_bits < 16) or (args.a_bits < 16) or (args.q_bits < 16) or (args.k_bits < 16) or (args.v_bits < 16)
    # cache path
    args.cache_dir = os.path.join(args.output_dir, ".cache")
    os.makedirs(args.cache_dir, exist_ok=True)
    # output path
    args.model_name = args.model.split("/")[-1]
    args.exp_dir = os.path.join(args.output_dir, args.model_name, f"w{args.w_bits}a{args.a_bits}", args.exp_name)
    os.makedirs(args.exp_dir, exist_ok=True)
    
    logger = create_logger(args.exp_dir)
    logger.info('Arguments: ')
    logger.info(pprint.pformat(vars(args)))
    logger.info('--' * 30)
    return args, logger


def create_logger(exp_dir, dist_rank=0, name=''):
    # create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    # create formatter
    fmt = '[%(asctime)s %(name)s] (%(filename)s %(lineno)d): %(levelname)s %(message)s'
    color_fmt = colored('[%(asctime)s %(name)s]', 'green') + \
                colored('(%(filename)s %(lineno)d)', 'yellow') + ': %(levelname)s %(message)s'

    # create console handlers for master process
    if dist_rank == 0:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(
            logging.Formatter(fmt=color_fmt, datefmt='%Y-%m-%d %H:%M:%S'))
        logger.addHandler(console_handler)

    # create file handlers
    log_file = os.path.join(exp_dir, f'log_rank{dist_rank}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt')
    file_handler = logging.FileHandler(log_file, mode='a')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(fmt=fmt, datefmt='%Y-%m-%d %H:%M:%S'))
    logger.addHandler(file_handler)

    return logger