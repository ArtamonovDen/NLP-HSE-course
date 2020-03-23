import argparse
import files2rouge
from pathlib import Path
import os

import torch
from tqdm import tqdm

from transformers import BartForConditionalGeneration, BartTokenizer

DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]

def generate_summaries(lns, out_file, batch_size=4, device=DEFAULT_DEVICE):
    fout = Path(out_file).open("w")
    model = BartForConditionalGeneration.from_pretrained("bart-large-cnn", output_past=True,).to(device)
    tokenizer = BartTokenizer.from_pretrained("bart-large")

    max_length = 140
    min_length = 55

    for batch in tqdm(list(chunks(lns, batch_size))):
        dct = tokenizer.batch_encode_plus(batch, max_length=512, return_tensors="pt", pad_to_max_length=True)
        summaries = model.generate(
            input_ids=dct["input_ids"].to(device),1024
            attention_mask=dct["attention_mask"].to(device),
            num_beams=4,
            length_penalty=2.0,
            max_length=max_length + 2,  # +2 from original because we start at step=1 and stop before max_length
            min_length=min_length + 1,  # +1 from original because we start at step=1
            no_repeat_ngram_size=3,
            early_stopping=True,
            decoder_start_token_id=model.config.eos_token_ids[0],
        )
        dec = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summaries]
        hypothesis = dec[0]
        fout.write(hypothesis)
        fout.flush()


def run_generate():
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--device", type=str, required=False, default=DEFAULT_DEVICE, help="cuda, cuda:1, cpu etc.",
    )
    parser.add_argument(
        "--bs", type=int, default=6, required=False, help="batch size: how many to summarize at a time",
    )
    args = parser.parse_args()

    for i in range(1, 11):
        source_path = f"cnn/selected_stories/{i}.story"
        target_path = f"cnn/targets/{i}.target"
        output_path = f"cnn/summaries/{i}.output"

        lns = [x.rstrip() for x in open(source_path).readlines()]
        remove_list = ["", "@highlight"]
        lns = list(filter(lambda l: l not in remove_list, lns))
        source = lns[:-4]
        target = lns[-4:]
        
        ftarget = Path(target_path).open("w")
        ftarget.write(' '.join(target) + "\n")
        ftarget.flush()
        generate_summaries(source, output_path, batch_size=args.bs, device=args.device)

def evaluate_rouge():
    for i in range(1, 11):
        target_path = f"cnn/targets/{i}.target"
        output_path = f"cnn/summaries/{i}.output"
        rouge_path = f"cnn/rouge/{i}.txt"

        files2rouge.run(output_path, target_path, saveto=rouge_path)


if __name__ == "__main__":
    run_generate()
    evaluate_rouge()