from import_reqs import *


def validate(datasets, moondream_ft):
    # Load the fine-tuned Moondream checkpoint


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

