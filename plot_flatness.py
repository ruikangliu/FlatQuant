import os

import flatquant.utils as utils
import flatquant.args_utils as args_utils
import flatquant.flatness as flatness


def main():
    args, logger = args_utils.parser_gen()
    utils.seed_everything(seed=args.seed)
    args.vis_dir = os.path.join(args.output_dir, "flatness", args.model_name)
    os.makedirs(args.vis_dir, exist_ok=True)

    # get activation scales for smoothquant
    if not os.path.exists(f"./act_scales/{args.model_name}.pt"):
        flatness.get_act_scales(args, logger)

    # get pre-quantization transformations
    flatness_flatquant = flatness.get_flatness(args, logger, transform_type="flatquant")
    flatness_hadamard = flatness.get_flatness(args, logger, transform_type="hadamard")
    flatness_smoothquant = flatness.get_flatness(args, logger, transform_type="smoothquant")
    flatness_vanilla = flatness.get_flatness(args, logger, transform_type=None)
    
    # plot flatness
    flatness.plot_flatness_all_layers(args, flatness_flatquant, flatness_hadamard, flatness_smoothquant, flatness_vanilla)


if __name__ == '__main__':
    main()
