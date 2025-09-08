# TrOCR_FineTuning_Street_View

[TrOCR model](https://huggingface.co/microsoft/trocr-base-printed), fine-tuned on [Street view](http://www.iapr-tc11.org/mediawiki/index.php?title=The_Street_View_Text_Dataset) dataset.

## Installation and usage
1) Install requirments:
```bash
pip install transformers torch datasets pandas numpy
```
2) Clone [trocr-base-printed](https://huggingface.co/microsoft/trocr-base-printed) to root folder of repository
3) Download [Street view](http://www.iapr-tc11.org/mediawiki/index.php?title=The_Street_View_Text_Dataset) and unpack it to root folder
4) Run Script
```bash
python Fine_Tuning.py
```