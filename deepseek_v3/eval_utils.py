import torch

def get_wikitext2_test(tokenizer):
    data_list = []
    with open("/home/ma-user/work/sunyuxuan/deepseek/datasets/wikitext/wikitext-2-raw/wiki.test.raw", encoding="utf-8") as f:
        for idx, row in enumerate(f):
            if row.strip():
                data_list.append(row.strip())
            else:
                data_list.append("")
    testenc = tokenizer("\n\n".join(data_list), return_tensors='pt')
    return testenc

@torch.no_grad()
def ppl_eval(model, testenc=None):
    if testenc is None:
        testenc = get_wikitext2_test(tokenizer)
    print('Evaluating ppl...')
    model.eval()
    max_length = 2048   # fix model max length

    testenc = testenc.input_ids
    nsamples = testenc.numel() // max_length

    dev = next(model.parameters()).device

    testenc = testenc.to(dev)
    nlls = []
    for i in tqdm(range(nsamples)):
        batch = testenc[:, (i * max_length): ((i + 1) * max_length)]
        lm_logits = model.forward(testenc[:, (i * max_length): ((i + 1) * max_length)], 0, cal_ppl=True)

        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = testenc[
            :, (i * max_length): ((i + 1) * max_length)
        ][:, 1:].to(shift_logits.device)

        loss_fct = torch.nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        neg_log_likelihood = loss.float() * max_length
        nlls.append(neg_log_likelihood)
    ppl = torch.exp(torch.stack(nlls).sum() / (len(nlls) * max_length))
    return ppl.item()


