from import_reqs import *

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


inference(Image.open("downloaded_images/image_60.jpg"))