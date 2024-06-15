from import_reqs import *
from validate import validate

def fine_tune(datasets):
    # print if cuda is available
    print("is CUDA available?? " + str(torch.cuda.is_available()))

    moondream = AutoModelForCausalLM.from_pretrained(
        "vikhyatk/moondream2", revision=MD_REVISION, trust_remote_code=True,
        torch_dtype=DTYPE, device_map={"": DEVICE}
    )

    sample = datasets['train'][0]
    # save the image to a file
    sample['image'].save('captcha.png')


    for qa in sample['qa']:
        print('Question:', qa['question'])
        print('Ground Truth:', qa['answer'])
        print('Moondream:', moondream.answer_question(
            moondream.encode_image(sample['image']),
            qa['question'],
            tokenizer=tokenizer,
        ))

    # Number of times to repeat the training dataset. Increasing this may cause the model to overfit or
    # lose generalization due to catastrophic forgetting. Decreasing it may cause the model to underfit.
    EPOCHS = 2

    # Number of samples to process in each batch. Set this to the highest value that doesn't cause an
    # out-of-memory error. Decrease it if you're running out of memory. Batch size 8 currently uses around
    # 15 GB of GPU memory during fine-tuning.
    BATCH_SIZE = 4

    # Number of batches to process before updating the model. You can use this to simulate a higher batch
    # size than your GPU can handle. Set this to 1 to disable gradient accumulation.
    GRAD_ACCUM_STEPS = 2

    # Learning rate for the Adam optimizer. Needs to be tuned on a case-by-case basis. As a general rule
    # of thumb, increase it by 1.4 times each time you double the effective batch size.
    #
    # Source: https://www.cs.princeton.edu/~smalladi/blog/2024/01/22/SDEs-ScalingRules/
    #
    # Note that we linearly warm the learning rate up from 0.1 * LR to LR over the first 10% of the
    # training run, and then decay it back to 0.1 * LR over the last 90% of the training run using a
    # cosine schedule.
    LR = 3e-5

    # Whether to use Weights and Biases for logging training metrics.
    USE_WANDB = False


    # The current version of moondream uses "<END>" to denote the end of a response. In the future this
    # will be replaced with a special token.
    ANSWER_EOS = "<END>"

    # Number of tokens used to represent each image.
    IMG_TOKENS = 729

    def collate_fn(batch):
        images = [sample['image'] for sample in batch]
        images = torch.stack(moondream.vision_encoder.preprocess(images))
        images = rearrange(images,
                        "b c (h p1) (w p2) -> b (h w) (c p1 p2)",
                        p1=14, p2=14)

        labels_acc = []
        tokens_acc = []

        for sample in batch:
            toks = [tokenizer.bos_token_id]
            labs = [-100] * (IMG_TOKENS + 1)

            for qa in sample['qa']:
                q_t = tokenizer(
                    f"\n\nQuestion: {qa['question']}\n\nAnswer:",
                    add_special_tokens=False
                ).input_ids
                toks.extend(q_t)
                labs.extend([-100] * len(q_t))

                a_t = tokenizer(
                    f" {qa['answer']}{ANSWER_EOS}",
                    add_special_tokens=False
                ).input_ids
                toks.extend(a_t)
                labs.extend(a_t)

            tokens_acc.append(toks)
            labels_acc.append(labs)

        max_len = -1
        for labels in labels_acc:
            max_len = max(max_len, len(labels))

        attn_mask_acc = []

        for i in range(len(batch)):
            len_i = len(labels_acc[i])
            pad_i = max_len - len_i

            labels_acc[i].extend([-100] * pad_i)
            tokens_acc[i].extend([tokenizer.eos_token_id] * pad_i)
            attn_mask_acc.append([1] * len_i + [0] * pad_i)

        return (
            images.to(dtype=DTYPE),
            torch.stack([torch.tensor(t, dtype=torch.long) for t in tokens_acc]),
            torch.stack([torch.tensor(l, dtype=torch.long) for l in labels_acc]),
            torch.stack([torch.tensor(a, dtype=torch.bool) for a in attn_mask_acc]),
        )

    def compute_loss(batch):
        images, tokens, labels, attn_mask = batch

        images = images.to(DEVICE)
        tokens = tokens.to(DEVICE)
        labels = labels.to(DEVICE)
        attn_mask = attn_mask.to(DEVICE)

        with torch.no_grad():
            img_embs = moondream.vision_encoder.encoder(images)
            img_embs = moondream.vision_encoder.projection(img_embs)

        tok_embs = moondream.text_model.get_input_embeddings()(tokens)
        inputs_embeds = torch.cat((tok_embs[:, 0:1, :], img_embs, tok_embs[:, 1:, :]), dim=1)

        outputs = moondream.text_model(
            inputs_embeds=inputs_embeds,
            labels=labels,
            attention_mask=attn_mask,
        )

        return outputs.loss

    def lr_schedule(step, max_steps):
        x = step / max_steps
        if x < 0.1:
            return 0.1 * LR + 0.9 * LR * x / 0.1
        else:
            return 0.1 * LR + 0.9 * LR * (1 + math.cos(math.pi * (x - 0.1))) / 2

    dataloaders = {
        "train": DataLoader(
            datasets["train"],
            batch_size=BATCH_SIZE,
            shuffle=True,
            collate_fn=collate_fn,
        ),
        "val": DataLoader(
            datasets["val"],
            batch_size=BATCH_SIZE,
            collate_fn=collate_fn,
        ),
    }

    moondream.text_model.train()
    moondream.text_model.transformer.gradient_checkpointing_enable()

    total_steps = EPOCHS * len(dataloaders["train"]) // GRAD_ACCUM_STEPS
    optimizer = Adam8bit(
        [
            {"params": moondream.text_model.parameters()},
        ],
        lr=LR * 0.1,
        betas=(0.9, 0.95),
        eps=1e-6
    )

    i = 0
    for epoch in range(EPOCHS):
        for batch in tqdm(dataloaders["train"], desc=f"Epoch {epoch + 1}/{EPOCHS}"):
            i += 1

            loss = compute_loss(batch)
            loss.backward()

            if i % GRAD_ACCUM_STEPS == 0:
                optimizer.step()
                optimizer.zero_grad()

                lr = lr_schedule(i / GRAD_ACCUM_STEPS, total_steps)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr

            if i % 100 == 0 and USE_WANDB:
                # Calculate validation loss
                val_loss = 0
                for val_batch in tqdm(dataloaders["val"], desc="Validation"):
                    with torch.no_grad():
                        val_loss += compute_loss(val_batch).item()
                val_loss /= len(dataloaders["val"])


    moondream.save_pretrained("checkpoints/moondream-ft")

    validate(datasets, moondream)