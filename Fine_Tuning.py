from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
from datasets import Dataset
import numpy as np
import xml.etree.ElementTree as ET
import os
from PIL import Image


MODEL_PATH = "./trocr-base-printed/"
DATASET_PATH = "./archive/"
OUTPUT_PATH = "./FineTunedTrOCR_StreetView"
IMG_DIR = os.path.join(DATASET_PATH, "/img/")
#print(IMG_DIR)
processor = TrOCRProcessor.from_pretrained(MODEL_PATH)
model = VisionEncoderDecoderModel.from_pretrained(MODEL_PATH)


model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
model.config.pad_token_id = processor.tokenizer.pad_token_id
model.config.vocab_size = model.config.decoder.vocab_size


def parse_xml(xml_file, img_dir):
    tree = ET.parse(os.path.join(DATASET_PATH,xml_file))
    root = tree.getroot()
    samples = []
    for image in root.findall("image"):
        image_root = image.find("imageName").text
        #print(image_root)
        img_path = os.path.join(img_dir, image_root)
        img = Image.open(img_path).convert("RGB")
        for rect in image.find("taggedRectangles").findall("taggedRectangle"):
            x = int(rect.get("x"))
            y = int(rect.get("y"))
            w = int(rect.get("width"))
            h = int(rect.get("height"))
            word = rect.find("tag").text
            crop_img = img.crop((x,y,x+w,y+h))
            samples.append({"image":crop_img, "text": word})


    return samples

train_Dataset = Dataset.from_list(parse_xml("train.xml", DATASET_PATH))

test_Dataset = Dataset.from_list(parse_xml("test.xml", DATASET_PATH))


def preprocessing_batch(batch):
    images = [img for img in batch["image"]]
    pixel_values_ts = processor(images = images, return_tensors = "pt").pixel_values
    pixel_values = [pixel_values_ts[i] for i in range(len(images))]
    labels = processor.tokenizer(batch["text"],padding = "max_length", truncation = True).input_ids
    batch["pixel_values"] = pixel_values
    batch["labels"] = labels
    return batch

train_Dataset = train_Dataset.map(preprocessing_batch, batched = True, remove_columns = ["image", "text"])
test_Dataset = test_Dataset.map(preprocessing_batch, batched = True, remove_columns = ["image", "text"])

training_args = Seq2SeqTrainingArguments(
    output_dir = OUTPUT_PATH,
    eval_strategy= "epoch",
    per_device_train_batch_size = 6,
    per_device_eval_batch_size = 6,
    predict_with_generate = True,
    num_train_epochs = 8,
    save_strategy = "epoch",
    fp16 = True
)
trainer = Seq2SeqTrainer(model = model, args = training_args,train_dataset = train_Dataset, eval_dataset = test_Dataset, tokenizer = processor.feature_extractor)

trainer.train()

#trainer.save_model(OUTPUT_PATH)
model.save_pretrained(OUTPUT_PATH)
processor.save_pretrained(OUTPUT_PATH)
