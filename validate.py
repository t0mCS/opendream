from import_reqs import *


def validate(datasets):
    # Load the fine-tuned Moondream checkpoint
    moondream_ft = AutoModelForCausalLM.from_pretrained(
        "checkpoints/moondream-ft",
        revision=MD_REVISION, trust_remote_code=True,
        torch_dtype=DTYPE, device_map={"": DEVICE}
    )


    moondream_ft.eval()

    correct = 0

    for i, sample in enumerate(datasets['test']):
        md_answer = moondream_ft.answer_question(
            moondream_ft.encode_image(sample['image']),
            sample['qa'][0]['question'],
            tokenizer=tokenizer,
        )

        if md_answer == sample['qa'][0]['answer']:
            correct += 1

        if i < 3:
            print('Question:', sample['qa'][0]['question'])
            print('Ground Truth:', sample['qa'][0]['answer'])
            print('Moondream:', md_answer)

    print(f"\n\nAccuracy: {correct / len(datasets['test']) * 100:.2f}%")


def inference(image):
    # Load the fine-tuned Moondream checkpoint
    moondream_ft = AutoModelForCausalLM.from_pretrained(
        "checkpoints/moondream-ft",
        trust_remote_code=True,
        torch_dtype=DTYPE,
        device_map={"": DEVICE}
    )

    moondream_ft.eval()

    md_answer = moondream_ft.answer_question(
        moondream_ft.encode_image(image),
        "What does the text say?",
        tokenizer=tokenizer,
    )

    return md_answer