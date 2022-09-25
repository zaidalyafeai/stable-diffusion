# coding=utf-8
# Copyright 2022 The HuggingFace Team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import textwrap

import datasets

from PIL import Image
import glob 
import re 


_CITATION = """\\n"""

_DESCRIPTION = """\\n"""


class CatsImageConfig(datasets.BuilderConfig):
    """BuilderConfig for COCO cats image."""

    def __init__(
        self,
        data_url,
        url,
        task_templates=None,
        **kwargs,
    ):
        super(CatsImageConfig, self).__init__(
            version=datasets.Version("1.9.0", ""), **kwargs
        )
        self.data_url = data_url
        self.url = url
        self.task_templates = task_templates


class CatsImage(datasets.GeneratorBasedBuilder):
    """Cats image. You know, THE cats image from the COCO dataset."""

    BUILDER_CONFIGS = [
        CatsImageConfig(
            name="image",
            description=textwrap.dedent(""),
            url="",
            data_url="",
        )
    ]

    DEFAULT_CONFIG_NAME = "image"

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "image": datasets.Image(),
                    "text": datasets.Value("string"),
                }
            ),
            supervised_keys=("image",),
            homepage=self.config.url,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"path": "/content/stable-diffusion/server/static/larger_images"},
            ),
        ]
    def _generate_examples(self, path):
        """Generate examples."""
        image_paths = glob.glob(f"{path}/**.jpg")
        print(path)
        _id = 0 

        for path in image_paths:
          text = path.split("/")[-1].split('.')[0]
          text = re.sub(r'[0-9]', '', text).strip()
          if len(text) > 0:
            image = Image.open(path)
            yield _id, {'text': text, 'image':image}
            _id += 1
