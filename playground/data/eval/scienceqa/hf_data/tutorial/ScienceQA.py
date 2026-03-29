import json
from pathlib import Path

import datasets

_DESCRIPTION = """Science Question Answering (ScienceQA), a new benchmark that consists of 21,208 multimodal 
multiple choice questions with a diverse set of science topics and annotations of their answers 
with corresponding lectures and explanations. 
The lecture and explanation provide general external knowledge and specific reasons,
respectively, for arriving at the correct answer."""

# Lets use the project page instead of the github repo
_HOMEPAGE = "https://scienceqa.github.io"

_CITATION = """\
@inproceedings{lu2022learn,
    title={Learn to Explain: Multimodal Reasoning via Thought Chains for Science Question Answering},
    author={Lu, Pan and Mishra, Swaroop and Xia, Tony and Qiu, Liang and Chang, Kai-Wei and Zhu, Song-Chun and Tafjord, Oyvind and Clark, Peter and Ashwin Kalyan},
    booktitle={The 36th Conference on Neural Information Processing Systems (NeurIPS)},
    year={2022}
}
"""

_LICENSE = "Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)"


class ScienceQA(datasets.GeneratorBasedBuilder):
    """Science Question Answering (ScienceQA), a new benchmark that consists of 21,208 multimodal
multiple choice questions with a diverse set of science topics and annotations of their answers
with corresponding lectures and explanations.
The lecture and explanation provide general external knowledge and specific reasons,
respectively, for arriving at the correct answer."""

    VERSION = datasets.Version("1.0.0")

    def _info(self):
        return datasets.DatasetInfo(
                description=_DESCRIPTION,
                features=datasets.Features(
                        {
                            "image": datasets.Image(),
                            "question": datasets.Value("string"),
                            "choices": datasets.features.Sequence(datasets.Value("string")),
                            "answer": datasets.Value("int8"),
                            "hint": datasets.Value("string"),
                            "task": datasets.Value("string"),
                            "grade": datasets.Value("string"),
                            "subject": datasets.Value("string"),
                            "topic": datasets.Value("string"),
                            "category": datasets.Value("string"),
                            "skill": datasets.Value("string"),
                            "lecture": datasets.Value("string"),
                            "solution": datasets.Value("string")
                            }
                        ),
                homepage=_HOMEPAGE,
                citation=_CITATION,
                license=_LICENSE,
                )

    def _split_generators(self, dl_manager):
        text_path = Path.cwd() / 'text' / 'problems.json'
        image_dir = Path.cwd() / 'images'
        return [
            datasets.SplitGenerator(
                    name=datasets.Split.TRAIN,
                    # These kwargs will be passed to _generate_examples
                    gen_kwargs={
                        "text_path": text_path,
                        "image_dir": image_dir,
                        "split": "train",
                        },
                    ),
            datasets.SplitGenerator(
                    name=datasets.Split.VALIDATION,
                    # These kwargs will be passed to _generate_examples
                    gen_kwargs={
                        "text_path": text_path,
                        "image_dir": image_dir,
                        "split": "val",
                        },
                    ),
            datasets.SplitGenerator(
                    name=datasets.Split.TEST,
                    # These kwargs will be passed to _generate_examples
                    gen_kwargs={
                        "text_path": text_path,
                        "image_dir": image_dir,
                        "split": "test"
                        },
                    ),
            ]

    # method parameters are unpacked from `gen_kwargs` as given in `_split_generators`
    def _generate_examples(self, text_path, image_dir, split):
        with open(text_path, encoding="utf-8") as f:
            # Load all the text. Note that if this was HUGE, we would need to find a better way to load the json
            data = json.load(f)
            ignore_keys = ['image', 'split']

            # Get image_id from its annoying location
            for image_id, row in data.items():
                # Only look for the rows in our split
                if row['split'] == split:

                    # Note, not all rows have images.
                    # Get all the image data we need
                    if row['image']:
                        image_path = image_dir / split / image_id / 'image.png'
                        image_bytes = image_path.read_bytes()
                        image_dict = {'path': str(image_path), 'bytes': image_bytes}
                    else:
                        image_dict = None

                    # Keep only the keys we need
                    relevant_row = {k: v for k, v in row.items() if k not in ignore_keys}

                    return_dict = {
                        'image': image_dict,
                        **relevant_row
                        }
                    yield image_id, return_dict
