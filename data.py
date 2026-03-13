"""Data provider"""

import torch
import torch.utils.data as data

import os
# import nltk
import numpy as np
import json

from pytorch_pretrained_bert.tokenization import BertTokenizer


class PrecompDataset(data.Dataset):
    """
    Load precomputed captions and image features
    Possible options: f30k_precomp, coco_precomp
    """

    def __init__(self, data_path, data_split, tokenizer):

        self.tokenizer = tokenizer
        loc = data_path + '/'

        # load the raw captions
        self.captions = []
        self.tokenizer = tokenizer
        self.max_seq_len = 40

        for line in open(loc + '%s_caps.txt' % data_split, 'rb'):
            self.captions.append(line.strip())

        # self.IOU = np.load(loc + '%s_IOU_2.npy' % data_split)
        print(loc)
        print(data_split)
        print()
        # load the image features
        self.images = np.load(loc + '%s_ims.npy' % data_split, allow_pickle=True)

        # self.images = np.load(loc + '%s_feats.npy' % data_split, allow_pickle=True)

        # 如果是全图特征 (N, 2048)，则扩展为 (N, 1, 2048)
        if self.images.ndim == 2:
            self.images = self.images[:, np.newaxis, :]

        self.length = len(self.captions)

        # with open(loc + '%s_caps.json' % data_split) as f:
        #     self.depends = json.load(f)

        # rkiros glo_data has redundancy in images, we divide by 5
        if self.images.shape[0] != self.length:
            self.im_div = 5
        else:
            self.im_div = 1

        # the development set for coco is large and so validation would be slow
        if data_split == 'dev':
            self.length = 5000

    def __getitem__(self, index):
        # handle the image redundancy
        img_id = index // self.im_div
        image = torch.Tensor(self.images[img_id])

        caption = str(self.captions[index])

        # IOU = self.IOU[img_id]
        # depend = self.depends[index]

        target = self.get_text_input(caption)
        # IOU = torch.Tensor(IOU)

        return image, target, index, img_id

    def get_text_input(self, caption):
        caption_tokens = self.tokenizer.tokenize(caption)
        caption_tokens = ['[CLS]'] + caption_tokens + ['[SEP]']
        caption_ids = self.tokenizer.convert_tokens_to_ids(caption_tokens)

        ##
        # if len(caption_ids) >= self.max_seq_len:
        #     caption_ids = caption_ids[:self.max_seq_len]
        # else:
        #     caption_ids = caption_ids + [0] * (self.max_seq_len - len(caption_ids))

        # caption = torch.tensor(caption_ids)
        return caption_ids

    def __len__(self):
        return self.length


def collate_fn(data):
    """
    Build mini-batch tensors from a list of (image, caption, index, img_id) tuples.
    Args:
        data: list of (image, target, index, img_id) tuple.
            - image: torch tensor of shape (36, 2048).
            - target: torch tensor of shape (?) variable length.
    Returns:
        - images: torch tensor of shape (batch_size, 36, 2048).
        - targets: torch tensor of shape (batch_size, padded_length).
        - lengths: list; valid length for each padded caption.
    """
    # Sort a glo_data list by caption length
    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, captions, ids, img_ids\
        = zip(*data)

    # Merge images (convert tuple of 2D tensor to 3D tensor)
    images = torch.stack(images, 0)

    # Iou = torch.stack(Iou, 0)

    ###
    # Merget captions (convert tuple of 1D tensor to 2D tensor)
    # lengths = [len(cap) for cap in captions]
    # targets = torch.zeros(len(captions), max(lengths)).long()
    # for i, cap in enumerate(captions):
    #     end = lengths[i]
    #     targets[i, :end] = cap[:end]

    ##
    lengths = [len(cap) for cap in captions]
    max_seq_len = max(len(cap) for cap in captions)
    # targets = torch.zeros(len(captions), max_seq_len).long()
    # for i, caption in enumerate(captions):
    #     caption = caption + [0] * (max_seq_len - len(caption))
    #     targets[i, :max_seq_len] = caption[:max_seq_len]
    # targets = torch.tensor(targets)
    captions = [torch.tensor(caption + [0] * (max_seq_len - len(caption))).long() for caption in captions]
    targets = torch.stack(captions)

    return images, targets, lengths,ids


def get_tokenizer(bert_path):
    print(bert_path)
    tokenizer = BertTokenizer(bert_path + 'vocab.txt')
    return tokenizer


def get_precomp_loader(data_path, data_split, tokenizer=None, batch_size=32,
                       shuffle=True, num_workers=2):
    dset = PrecompDataset(data_path, data_split, tokenizer)

    data_loader = torch.utils.data.DataLoader(dataset=dset,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              pin_memory=True,
                                              collate_fn=collate_fn)
    return data_loader


def get_loaders(data_name, batch_size, workers, opt):
    # get the glo_data path
    dpath = os.path.join(opt.data_path, data_name)

    bert_path = opt.bert_path
    tokenizer = get_tokenizer(bert_path)
    # get the train_loader
    train_loader = get_precomp_loader(dpath, 'train', tokenizer,
                                      batch_size, True, workers)
    # get the val_loader
    val_loader = get_precomp_loader(dpath, 'dev', tokenizer,
                                    batch_size, False, workers)
    return train_loader, val_loader


def get_test_loader(split_name, data_name, batch_size, workers, opt):
    # get the glo_data path
    dpath = os.path.join(opt.data_path, data_name)

    bert_path = opt.bert_path
    tokenizer = get_tokenizer(bert_path)
    # get the test_loader
    test_loader = get_precomp_loader(dpath, split_name, tokenizer,
                                     batch_size, False, workers)
    return test_loader
