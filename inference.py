from import_reqs import *

def inference(image):
    # Load the fine-tuned Moondream checkpoint
    moondream = AutoModelForCausalLM.from_pretrained(
        "checkpoints/moondream-ft", revision=MD_REVISION, trust_remote_code=True,
        torch_dtype=DTYPE, device_map={"": DEVICE}
    )

    # moondream.from_pretrained("checkpoints/moondream-ft")

    moondream.eval()

    nu_tokenizer = AutoTokenizer.from_pretrained("checkpoints/moondream-ft", revision=MD_REVISION)


    md_answer = moondream.answer_question(
        moondream.encode_image(image),
        "Describe the image",
        tokenizer=nu_tokenizer,
    )

    print('Moondream:', md_answer)


inference(Image.open("downloaded_images/image_60.jpg"))