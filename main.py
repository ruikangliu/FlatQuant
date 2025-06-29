import transformers

import flatquant.utils as utils
import flatquant.args_utils as args_utils
import flatquant.model_utils as model_utils
import flatquant.data_utils as data_utils
import flatquant.eval_utils as eval_utils
import flatquant.train_utils as train_utils
import flatquant.flat_utils as flat_utils
import gptq_utils

def main():
    args, logger = args_utils.parser_gen()
    utils.seed_everything(seed=args.seed)

    model, apply_flatquant_to_model = model_utils.get_model(args.model, args.hf_token)
    model.eval()
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model, use_fast=False, use_auth_token=args.hf_token)

    # get calibration data
    trainloader = data_utils.get_loaders(
        args, args.cali_dataset, nsamples=args.nsamples,
        seed=args.seed, model=args.model,
        seqlen=model.seqlen, eval_mode=False
    )
    logger.info("Finished loading training data.")

    if args.quantize:
        model = apply_flatquant_to_model(args, model)
        logger.info("Finished applying FlatQuant to model.")
        if args.resume:
            flat_utils.load_flat_parameters(args, model)
        elif args.reload_matrix:
            flat_utils.load_flat_matrices(args, model, path=args.matrix_path)
        elif (args.cali_trans or args.add_diag or args.lwc or args.lac):
            train_utils.cali_flat_quant(args, model, trainloader, utils.DEV, logger=logger)
        if args.save_matrix and not args.reload_matrix:
            flat_utils.save_flat_matrices(args, model)
        flat_utils.reparameterize_model(model)
        logger.info("Finished reparameterize model.")

    if args.w_bits < 16:
        save_dict = {}
        if args.gptq: # GPTQ Weight Quantization
            quantizers = gptq_utils.gptq_fwrd(model, trainloader, utils.DEV, args)
        else: # RTN Weight Quantization
            quantizers = gptq_utils.rtn_fwrd(model, utils.DEV, args)
        save_dict["w_quantizers"] = quantizers

    if args.distribute_model:
        utils.distribute_model(model)
    else:
        model.to(utils.DEV)
    
    # Evaluating PPL
    for eval_dataset in ["wikitext2", "c4"]:
        logger.info(eval_dataset)
        testloader = data_utils.get_loaders(
                args,
                eval_dataset,
                seed=args.seed,
                model=args.model,
                seqlen=model.seqlen,
                hf_token=args.hf_token,
                eval_mode=True
            )
        dataset_ppl = eval_utils.ppl_eval(model, testloader)
        logger.info(dataset_ppl)


    if args.lm_eval:
        import lm_eval
        from lm_eval import utils as lm_eval_utils
        from lm_eval.models.huggingface import HFLM

        hflm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=args.lm_eval_batch_size)

        # task_manager = lm_eval.tasks.TaskManager(include_path="./datasets/lm_eval_configs/tasks", include_defaults=False)
        # task_names = lm_eval_utils.pattern_match(args.tasks, task_manager.all_tasks)
        task_names = args.tasks

        results = {}
        for task_name in task_names:
            logger.info(f"Evaluating {task_name}...")
            result = lm_eval.simple_evaluate(hflm, tasks=[task_name], batch_size=args.lm_eval_batch_size)['results']
            result = result[task_name]
            acc = round(result.get('acc_norm,none', result['acc,none']) * 100, 2)
            results[task_name] = acc
            logger.info(f"acc: {acc}%")
        metric_vals = {task: result for task, result in results.items()}
        metric_vals['acc_avg'] = round(sum(metric_vals.values()) / len(metric_vals.values()), 2)
        logger.info(metric_vals)


if __name__ == '__main__':
    main()