import re
from pathlib import Path

import click
import numpy as np
import torch
from stylegan2-pytoch.model import Generator

import pickle


def convert_to_rgb(state_ros, state_nv, ros_name, nv_name):
    state_nv[f"{nv_name}.torgb.weight"]= state_ros[f"{ros_name}.conv.weight"][0]
    state_nv[f"{nv_name}.torgb.bias"] = torch.squeeze(torch.squeeze(torch.squeeze(state_ros[f"{ros_name}.bias"], 0), -1), -1)
    state_nv[f"{nv_name}.torgb.affine.weight"] = state_ros[f"{ros_name}.conv.modulation.weight"]
    state_nv[f"{nv_name}.torgb.affine.bias"] = state_ros[f"{ros_name}.conv.modulation.bias"]


def convert_conv(state_ros, state_nv, ros_name, nv_name):
    state_nv[f"{nv_name}.weight"] = state_ros[f"{ros_name}.conv.weight"][0]
    state_nv[f"{nv_name}.bias"] = state_ros[f"{ros_name}.activate.bias"]
    state_nv[f"{nv_name}.affine.weight"] = state_ros[f"{ros_name}.conv.modulation.weight"]
    state_nv[f"{nv_name}.affine.bias"] = state_ros[f"{ros_name}.conv.modulation.bias"]
    state_nv[f"{nv_name}.noise_strength"] = state_ros[f"{ros_name}.noise.weight"][0]


def convert_blur_kernel(state_ros, state_nv, level):
    """Not quite sure why there is a factor of 4 here"""
    # They are all the same
    state_nv["synthesis.b4.resample_filter"] = state_ros[f"convs.{2 * level}.conv.blur.kernel"] / 4
    state_nv["synthesis.b8.resample_filter"] = state_ros[f"to_rgbs.{level}.upsample.kernel"] / 4
    state_nv["synthesis.b16.resample_filter"] = state_ros[f"to_rgbs.{level}.upsample.kernel"] / 4
    state_nv["synthesis.b32.resample_filter"] = state_ros[f"to_rgbs.{level}.upsample.kernel"] / 4
    state_nv["synthesis.b64.resample_filter"] = state_ros[f"to_rgbs.{level}.upsample.kernel"] / 4
    state_nv["synthesis.b128.resample_filter"] = state_ros[f"to_rgbs.{level}.upsample.kernel"] / 4

    state_nv["synthesis.b4.conv1.resample_filter"] = state_ros[f"to_rgbs.{level}.upsample.kernel"] / 4
    state_nv["synthesis.b8.conv0.resample_filter"] = state_ros[f"to_rgbs.{level}.upsample.kernel"] / 4
    state_nv["synthesis.b8.conv1.resample_filter"] = state_ros[f"to_rgbs.{level}.upsample.kernel"] / 4
    state_nv["synthesis.b16.conv0.resample_filter"] = state_ros[f"to_rgbs.{level}.upsample.kernel"] / 4
    state_nv["synthesis.b16.conv1.resample_filter"] = state_ros[f"to_rgbs.{level}.upsample.kernel"] / 4
    state_nv["synthesis.b32.conv0.resample_filter"] = state_ros[f"to_rgbs.{level}.upsample.kernel"] / 4
    state_nv["synthesis.b32.conv1.resample_filter"] = state_ros[f"to_rgbs.{level}.upsample.kernel"] / 4
    state_nv["synthesis.b64.conv0.resample_filter"] = state_ros[f"to_rgbs.{level}.upsample.kernel"] / 4
    state_nv["synthesis.b64.conv1.resample_filter"] = state_ros[f"to_rgbs.{level}.upsample.kernel"] / 4
    state_nv["synthesis.b128.conv0.resample_filter"] = state_ros[f"to_rgbs.{level}.upsample.kernel"] / 4
    state_nv["synthesis.b128.conv1.resample_filter"] = state_ros[f"to_rgbs.{level}.upsample.kernel"] / 4


def determine_config(state_ros):
    mapping_names = [name for name in state_ros.keys() if "style" in name]
    sythesis_names = [name for name in state_ros.keys() if "to_rgbs" in name]

    n_mapping = max([int(re.findall("(\d+)", n)[0]) for n in mapping_names])
    n_layers = max([int(re.findall("(\d+)", n)[0]) for n in sythesis_names]) + 2

    return n_mapping, n_layers


def convert(network_pkl, output_file):
    res = torch.load(network_pkl)
    state_ros = res["g_ema"]
    n_mapping, n_layers = determine_config(state_ros)

    state_nv = {}

    for i in range(n_mapping):
        state_nv[f"mapping.fc{i}.weight"] = state_ros[f"style.{i + 1}.weight"]
        state_nv[f"mapping.fc{i}.bias"] = state_ros[f"style.{i + 1}.bias"]

    for i in range(int(n_layers)):
        if i > 0:
            for conv_level in range(2):
                convert_conv(state_ros, state_nv, f"convs.{2 * i - 2 + conv_level}",
                             f"synthesis.b{4 * (2 ** i)}.conv{conv_level}")
                state_nv[
                    f"synthesis.b{4 * (2 ** i)}.conv{conv_level}.noise_const"] = state_ros[f"noises.noise_{2 * i - 1 + conv_level}"][0][0]

            convert_to_rgb(state_ros, state_nv, f"to_rgbs.{i - 1}", f"synthesis.b{4 * (2 ** i)}")
            convert_blur_kernel(state_ros, state_nv, i - 1)

        else:
            state_nv[f"synthesis.b{4 * (2 ** i)}.const"] = state_ros[f"input.input"][0]
            convert_conv(state_ros, state_nv, "conv1", f"synthesis.b{4 * (2 ** i)}.conv1")
            state_nv[f"synthesis.b{4 * (2 ** i)}.conv1.noise_const"] = state_ros[f"noises.noise_{2 * i}"][0][0]
            convert_to_rgb(state_ros, state_nv, "to_rgb1", f"synthesis.b{4 * (2 ** i)}")

    # https://github.com/yuval-alaluf/restyle-encoder/issues/1#issuecomment-828354736
    state_nv['mapping.w_avg'] = torch.zeros([512], dtype=torch.float64)
   # state_nv['mapping.w_avg'] =
    with open(output_file, 'wb') as f:
        pickle.dump(state_nv, f)

convert('/content/drive/MyDrive/diploma/checkpointLOGAN/237000.pt', "/content/drive/MyDrive/diploma/out.pkl")