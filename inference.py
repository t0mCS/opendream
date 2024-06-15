from import_reqs import *

def inference(image):
    # Load the fine-tuned Moondream checkpoint
    moondream = AutoModelForCausalLM.from_pretrained(
        "vikhyatk/moondream2", revision=MD_REVISION, trust_remote_code=True,
        torch_dtype=DTYPE, device_map={"": DEVICE}
    )

    moondream.from_pretrained("checkpoints/moondream-ft")

    moondream.eval()

    md_answer = moondream.answer_question(
        moondream.encode_image(image),
        "What does the text say?",
        tokenizer=tokenizer,
    )

    return md_answer


inference(Image.open("downloaded_images/image_60.jpg"))