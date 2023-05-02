# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from concurrent.futures import ThreadPoolExecutor
from functools import partial
import io
import json
import os

from typing import Callable, Tuple, Union

import PIL.Image
from torch.utils.data import Dataset
from datasets import load_dataset

from datasets.utils.file_utils import get_datasets_user_agent
import urllib
from torchvision.datasets.folder import find_classes, make_dataset


IMG_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".ppm",
    ".bmp",
    ".pgm",
    ".tif",
    ".tiff",
    ".webp",
)


class ImageNet1K(Dataset):
    r"""The ImageNet-1K dataset.

    The ImageNet-1K dataset spans 1000 object classes and contains
    1,281,671 training images, 50,000 validation images and 100,000
    test images.

    Webpage: https://www.image-net.org/
    Reference: https://ieeexplore.ieee.org/abstract/document/5206848
    """

    def __init__(
        self,
        root: str,
        split: str,
        transform: Callable = None,
    ) -> None:
        r"""Constructor for ImageNet1K

        Parameters:
        -----------
        root: str
            Path where train and val split are saved.
        split: str
            The split (train or val).
        transform: Callable
            An image transformation.

        Returns:
        --------
        None
        """
        super().__init__()
        self._root = root
        self._split = split
        self._transform = transform
        self._data = self._load_data()

    def _load_data(self) -> None:
        directory = os.path.join(
            self._root,
            self._split,
        )
        _, synset_to_idx = find_classes(directory)
        data = make_dataset(
            directory=directory,
            class_to_idx=synset_to_idx,
            extensions=IMG_EXTENSIONS,
        )
        return data

    def __len__(self) -> int:
        r"""Return number of samples in the dataset.
        Parameters
        ----------
        None
        Returns
        -------
        len: int
            Number of samples in the dataset.
        """
        return len(self._data)

    def __getitem__(self, index: int) -> Union[Tuple, None]:
        r"""Return sample as tuple (image, caption).
        Parameters:
        -----------
        index: int
            The index of the sample.
        Returns:
        --------
        image: array_like
            The image for the `index` sample.
        label: int
            The class label for the `index` sample.
        """
        path, label = self._data[index]
        image = Image.open(path).convert("RGB")

        if self._transform is not None:
            image = self._transform(image)

        return image, label

    def __repr__(self) -> str:
        return "\n".join(
            [
                "ImageNet1K(",
                f"  split={self._split},",
                f"  n_samples={self.__len__()},",
                f"  transform={self._transform}",
                ")",
            ]
        )


class ImageMultiCaptionDataset(Dataset):
    r"""Generic class for image-caption datasets where each image is associated with
        more than one caption.
    Methods:
    --------
    __getitem__(index): Tuple
        Return a tuple (image, captions). Both are preprocessed with
        respective transforms.
    len(): int
        Return the number of samples.
    Attributes:
    number_of_captions: int
        Return the number of captions in the dataset.
    """

    def __init__(
        self,
        root: str,
        name: str,
        split: str,
        transform: Callable = None,
    ) -> None:
        r"""Constructor for ImageMultiCaptionDataset.
        Parameters:
        -----------
        root: str
            Path where train and val split are saved.
        split: str
            The split (train or val or test).
        transform: Callable
            An image transformation.
        Returns:
        --------
        None
        """
        self._root = root
        self._name = name
        self._split = split
        self._transform = transform
        self._load_data()

    def _load_data(self) -> None:
        with open(os.path.join(self._root, f"{self._name}_{self._split}.json")) as f:
            annotations = json.load(f)
        processed_data = self._prepare_data_from_annotations(
            annotations, root=self._root
        )
        self._text = processed_data["text"]
        self._image = processed_data["image_ids"]
        self._txt2img = processed_data["txt2img"]
        self._img2txt = processed_data["img2txt"]
        self._data = processed_data["image_captions_data"]
        self._number_of_captions = int(sum(len(captions) for _, captions in self._data))

    def _prepare_data_from_annotations(self, annotations, root=None):
        text = []
        image_ids = []
        txt2img = {}
        img2txt = {}
        image_captions_data = []

        txt_id = 0
        for img_id, ann in enumerate(annotations):
            image_i = os.path.join(root, ann["image"])
            image_ids.append(image_i)
            img2txt[img_id] = []
            captions_i = []
            if isinstance(ann["caption"], str):
                ann_caption = [ann["caption"]]
            elif isinstance(ann["caption"], list):
                ann_caption = ann["caption"]
            else:
                raise TypeError("'str' or 'list' allowed for captions")

            for caption in ann_caption:
                text.append(caption)
                captions_i.append(caption)
                img2txt[img_id].append(txt_id)
                txt2img[txt_id] = img_id
                txt_id += 1

            image_captions_data.append((image_i, captions_i))

        return {
            "text": text,
            "image_ids": image_ids,
            "txt2img": txt2img,
            "img2txt": img2txt,
            "image_captions_data": image_captions_data,
        }

    def __len__(self) -> int:
        r"""Return number of images in the dataset.
        Parameters
        ----------
        None
        Returns
        -------
        len: int
            Number of images in the dataset.
        """
        return len(self._data)

    @property
    def number_of_captions(self) -> int:
        r"""Return number of captions in the dataset.
        Parameters
        ----------
        None
        Returns
        -------
        number_of_captions: int
            Number of captions in the dataset.
        """
        return self._number_of_captions

    def __getitem__(self, index: int) -> Union[Tuple, None]:
        r"""Return sample as tuple (image, index).
        We only return image and index because there could be varying number of
        captions per image and it could break default collates.
        The class implements _load_text to handle loading captions per image.
        Parameters:
        -----------
        index: int
            The index of the sample.
        Returns:
        --------
        image: array_like
            The image for the `index` sample.
        index: int
            The index of the sample.
        Exception handling:
        -------------------
        If an exception is raised during dataloading,
        this function returns `None`.
        """
        image = self._load_image(index)
        return image, index

    def _load_image(self, index: int) -> Tuple:
        path, _ = self._data[index]
        image = Image.open(path).convert("RGB")
        if self._transform is not None:
            image = self._transform(image)
        return image

    def _load_captions(self, index):
        _, captions = self._data[index]
        if isinstance(captions, str):
            captions = [captions]
        return captions

    def _load_text(self, index: int) -> Tuple:
        captions = self._load_captions(index)
        return captions, self._img2txt[index]

    def _load_item(self, index: int) -> Tuple:
        image = self._load_image(index)
        captions = self._load_captions(index)
        return image, captions
    
class ImageSingleCaptionDataset(Dataset):
    r"""Generic class for image-caption datasets where each image is associated with
        a single caption.
    Methods:
    --------
    __getitem__(index): Tuple
        Return a tuple (image, captions). Both are preprocessed with
        respective transforms.
    len(): int
        Return the number of samples.
    Attributes:
    number_of_captions: int
        Return the number of captions in the dataset.
    """

    def __init__(
        self,
        name: str,
        split: str,
        transform: Callable = None,
    ) -> None:
        r"""Constructor for ImageMultiCaptionDataset.
        Parameters:
        -----------
        split: str
            The split (train or val or test).
        transform: Callable
            An image transformation.
        Returns:
        --------
        None
        """
        self._name = name
        self._split = split
        self._transform = transform
        self._load_data()

    def fetch_single_image(self, image_url, timeout=None, retries=0):
        USER_AGENT = get_datasets_user_agent()
        for _ in range(retries + 1):
            try:
                request = urllib.request.Request(
                    image_url,
                    data=None,
                    headers={"user-agent": USER_AGENT},
                )
                with urllib.request.urlopen(request, timeout=timeout) as req:
                    image = PIL.Image.open(io.BytesIO(req.read()))
                break
            except Exception:
                image = None
        return image
    
    def fetch_images(self, batch, num_threads, timeout=None, retries=0):
        fetch_single_image_with_args = partial(self.fetch_single_image, timeout=timeout, retries=retries)
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            batch["image"] = list(executor.map(fetch_single_image_with_args, batch["image_url"]))
        
        return batch
    
    def _load_data(self) -> None:
        # Huggingface code to load the dataset
        dset = load_dataset("conceptual_captions")
        dset_val = dset[self._split].select(range(100)) # Select the first 100 examples in validation set
        
        num_threads = 20

        dset_val = dset_val.map(self.fetch_images, batched=True, batch_size=100, fn_kwargs={"num_threads": num_threads})

        processed_data = self._prepare_data_from_annotations(dset_val)
        
        self._text = processed_data["captions"]
        self._txt2img = processed_data["txt2img"]
        self._img2txt = processed_data["img2txt"]
        self._data = processed_data["image_captions_data"]
        self._number_of_captions = int(sum(len(captions) for _, captions in self._data))

    def _prepare_data_from_dict(self, dataset):
        text = []
        txt2img = []
        img2txt = []
        image_captions_data = []
        img_count = 0
        # Here since each image has only one caption, we don't need to keep track of text_id and img_id
        for index, dset_dict in enumerate(dataset):
            if dset_dict["image"] is not None:
              image_i = dset_dict["image"]
              captions_i = dset_dict["caption"]
              text.append(dset_dict["caption"])
              txt2img.append(img_count)        
              img2txt.append(img_count)
              image_captions_data.append((image_i, captions_i))
              img_count += 1
            else:
              continue

        return {
            "captions": text,
            "txt2img": txt2img,
            "img2txt": img2txt,
            "image_captions_data": image_captions_data,
        }

    def __len__(self) -> int:
        r"""Return number of images in the dataset.
        Parameters
        ----------
        None
        Returns
        -------
        len: int
            Number of images in the dataset.
        """
        return len(self._data)

    @property
    def number_of_captions(self) -> int:
        r"""Return number of captions in the dataset.
        Parameters
        ----------
        None
        Returns
        -------
        number_of_captions: int
            Number of captions in the dataset.
        """
        return self._number_of_captions

    # This method is used by the dataloader to get the data, so keep it as it is
    def __getitem__(self, index: int) -> Union[Tuple, None]:
        r"""Return sample as tuple (image, index).
        We only return image and index because the captions could be of varying size
        and it could break default collates.
        The class implements _load_caption to handle loading captions.
        Parameters:
        -----------
        index: int
            The index of the sample.
        Returns:
        --------
        image: array_like
            The image for the `index` sample.
        index: int
            The index of the sample.
        Exception handling:
        -------------------
        If an exception is raised during dataloading,
        this function returns `None`.
        """
        image = self._load_image(index)
        return image, index

    def _load_image(self, index: int) -> Tuple:
        image = self._data[index][0]
        image = image.convert("RGB")
        if self._transform is not None:
            image = self._transform(image)
        return image

    def _load_caption(self, index):
        caption = self._data[index][1]
        return caption

    # def _load_text(self, index: int) -> Tuple:
    #     captions = self._load_captions(index)
    #     return captions, self._img2txt[index]

    def _load_item(self, index: int) -> Tuple:
        image = self._load_image(index)
        captions = self._load_caption(index)
        return image, captions
    

class GCC(ImageSingleCaptionDataset):
    r"""The Google Conceptual Captions dataset.
    Webpage: https://ai.google.com/research/ConceptualCaptions/
    """

    def __init__(
        self,
        split: str,
        transform: Callable = None,
    ) -> None:
        r"""Constructor for GCC dataset.
        Parameters:
        -----------
        root: str
            Path where train and val split are saved.
        split: str
            The split (train or val or test).
        transform: Callable
            An image transformation.
        Returns:
        --------
        None
        """
        super().__init__(
            name="gcc",
            split=split,
            transform=transform,
        )

    def __repr__(self) -> str:
        return "\n".join(
            [
                "GCC(",
                f"  split={self._split},",
                f"  n_images={self.__len__()},",
                f"  n_captions={self.number_of_captions}"
                f"  transform={self._transform}",
                ")",
            ]
        )

class COCO(ImageMultiCaptionDataset):
    r"""The COCO Captions dataset.
    Webpage: https://cocodataset.org/#download
    """

    def __init__(
        self,
        root: str,
        split: str,
        transform: Callable = None,
    ) -> None:
        r"""Constructor for COCO
        Parameters:
        -----------
        root: str
            Path where train and val split are saved.
        split: str
            The split (train or val or test).
        transform: Callable
            An image transformation.
        Returns:
        --------
        None
        """
        super().__init__(
            root=root,
            name="coco",
            split=split,
            transform=transform,
        )

    def __repr__(self) -> str:
        return "\n".join(
            [
                "COCO(",
                f"  split={self._split},",
                f"  n_images={self.__len__()},",
                f"  n_captions={self.number_of_captions}"
                f"  transform={self._transform}",
                ")",
            ]
        )


class Flickr30K(ImageMultiCaptionDataset):
    r"""The Flickr30K Captions dataset.
    Webpage: https://shannon.cs.illinois.edu/DenotationGraph/
    """

    def __init__(
        self,
        root: str,
        split: str,
        transform: Callable = None,
    ) -> None:
        r"""Constructor for COCO
        Parameters:
        -----------
        root: str
            Path where train and val split are saved.
        split: str
            The split (train or val or test).
        transform: Callable
            An image transformation.
        Returns:
        --------
        None
        """
        super().__init__(
            root=root,
            name="flickr30k",
            split=split,
            transform=transform,
        )

    def __repr__(self) -> str:
        return "\n".join(
            [
                "Flickr30K(",
                f"  split={self._split},",
                f"  n_images={self.__len__()},",
                f"  n_captions={self.number_of_captions}"
                f"  transform={self._transform}",
                ")",
            ]
        )
