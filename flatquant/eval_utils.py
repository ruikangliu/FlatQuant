import torch
from tqdm import tqdm

@torch.no_grad()
def ppl_eval(model, testenc):
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
        lm_logits = model(batch).logits
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

