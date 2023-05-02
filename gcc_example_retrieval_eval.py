# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np

import torch
import torch.nn.functional as F

from diht import model_zoo
from diht.dataset import COCO, Flickr30K
from diht.dataset import GCC

from torch.utils.data import DataLoader
from tqdm import tqdm


COCO_ROOT = "<YOUR_COCO_ROOT_HERE>"  # replace with your COCO root directory
FLICKRR30K_ROOT = (
    "<YOUR_FLICKR30K_ROOT_HERE>"  # replace with your Flickr30K root directory
)
MODEL_NAME = "diht_vitl14_336px"  # replace with the model you want to evaluate
TEMPLATE = "a photo of {}."

# Updated for GCC dataset
def extract_features(model, text_tokenizer, eval_dataset, eval_dataloader):
    all_image_features = []
    all_text_features = []
    with torch.no_grad():
        for batch in tqdm(eval_dataloader):
            images, image_idxs = batch[0], batch[1]
            image_features = model.encode_image(images.cuda()) # model.encode_image(images.cuda())
            image_features = F.normalize(image_features, dim=-1)
            all_image_features.append(image_features.detach().cpu())
            for imidx in image_idxs.tolist():
                caption = eval_dataset._load_caption(imidx)
                caption = TEMPLATE.format(caption)
                caption = text_tokenizer(caption)
                text_features = model.encode_text(caption.cuda())  # model.encode_text(captions.cuda())
                text_features = F.normalize(text_features, dim=-1)
                all_text_features.append(text_features.detach().cpu())
    return (
        torch.cat(all_image_features),
        torch.cat(all_text_features),
    )


def evaluate_retrieval(image_features, text_features, txt2img, img2txt):
    # compute similarity: image2text (i2t) and text2image (t2i)
    sim_i2t = image_features @ text_features.t() # Will this work for single caption?
    sim_t2i = sim_i2t.t()
    # evaluate image-text matching (itm)
    i2t, t2i = compute_recall(
        sim_i2t.numpy(),
        sim_t2i.numpy(),
        txt2img,
        img2txt,
        k=[1],
    )
    return i2t, t2i


def compute_recall(
    scores_i2t,
    scores_t2i,
    txt2img,
    img2txt,
    k=[1, 5, 10],
):
    """Compute retrieval recall metrics."""
    # Original images-to-text
    # ranks = np.zeros(scores_i2t.shape[0])
    # for index, score in enumerate(scores_i2t):
    #     inds = np.argsort(score)[::-1]
    #     # score
    #     rank = int(1e20)
    #     for i in img2txt:
    #         tmp_arr = np.where(inds == i)[0]
    #         tmp = tmp_arr[0]
    #         if tmp < rank:
    #             rank = tmp
    #     ranks[index] = rank
    # # overall: compute text retrieval recall
    # trs = {}
    # for topk in k:
    #     trs[topk] = len(np.where(ranks < topk)[0]) / len(ranks)


    # image-to-text
    ranks = np.zeros(scores_i2t.shape[0])
    for index, score in enumerate(scores_i2t):
        inds = np.argsort(score)[::-1]
        i2t_curr_rank_arr = np.where(inds == img2txt[index])[0]
        i2t_curr_rank = i2t_curr_rank_arr[0]
        ranks[index] = i2t_curr_rank
    print(f"ranks for image-to-text: {ranks}")
    # overall: compute image retrieval recall
    trs = {}
    for topk in k:
        trs[topk] = len(np.where(ranks < topk)[0]) / len(ranks)

    # text-to-image
    ranks = np.zeros(scores_t2i.shape[0])
    for index, score in enumerate(scores_t2i):
        inds = np.argsort(score)[::-1]
        t2i_curr_rank_arr = np.where(inds == txt2img[index])[0]
        t2i_curr_rank = t2i_curr_rank_arr[0]
        ranks[index] = t2i_curr_rank
    print(f"ranks for text-to-image: {ranks}")
    # overall: compute image retrieval recall
    irs = {}
    for topk in k:
        irs[topk] = len(np.where(ranks < topk)[0]) / len(ranks)
    return trs, irs


def main():
    # get tokenizer, transform, model, eval dataset and dataloader
    print(f"Load model {MODEL_NAME} ...")
    text_tokenizer, image_transform, model = model_zoo.load_model(
        MODEL_NAME, is_train=False
    )
    coco_dataset = COCO(root=COCO_ROOT, split="test", transform=image_transform)
    
    coco_dataloader = DataLoader(
        dataset=coco_dataset,
        batch_size=32,
        num_workers=5,
    )

    gcc_dataset = GCC(split="validation", transform=image_transform)
    gcc_dataloader = DataLoader(
        dataset=gcc_dataset,
        batch_size=32,
        num_workers=5,
    )

    # move model to GPU
    model = model.cuda()

    # compute image features
    print("Extract features for COCO ...")
    (
        coco_image_features,
        coco_image_idxs,
        coco_text_features,
        coco_text_idxs,
    ) = extract_features(model, text_tokenizer, coco_dataset, coco_dataloader)
    # evaluate
    print("Compute recall for COCO ...")
    coco_i2t, coco_t2i = evaluate_retrieval(
        coco_image_features,
        coco_text_features,
        coco_dataset._txt2img,
        coco_dataset._img2txt,
    )
    print(f"COCO T2I r@1 for {MODEL_NAME}: {100*coco_t2i[1]:.1f}")
    print(f"COCO I2T r@1 for {MODEL_NAME}: {100*coco_i2t[1]:.1f}")

    # compute image features
    print("Extract features for GCC ...")
    (
        gcc_image_features,
        gcc_text_features
    ) = extract_features(model, text_tokenizer, gcc_dataset, gcc_dataloader)
    # evaluate
    print("Compute recall for GCC ...")
    gcc_i2t, gcc_t2i = evaluate_retrieval(
        gcc_image_features,
        gcc_text_features,
        gcc_dataset._txt2img,
        gcc_dataset._img2txt,
    )
    print(f"GCC T2I r@1 for {MODEL_NAME}: {100*gcc_t2i[1]:.1f}")
    print(f"GCC I2T r@1 for {MODEL_NAME}: {100*gcc_i2t[1]:.1f}")


if __name__ == "__main__":
    main()
