from import_reqs import *

def inference(image):
    # Load the fine-tuned Moondream checkpoint
    moondream = Moondream.from_pretrained("checkpoints/moondream-ft")

    moondream.eval()

    md_answer = moondream.answer_question(
        moondream.encode_image(image),
        "What does the text say?",
        tokenizer=tokenizer,
    )

    return md_answer


inference(Image.open("downloaded_images/image_60.jpg"))