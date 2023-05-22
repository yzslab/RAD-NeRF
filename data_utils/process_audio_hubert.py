from argparse import ArgumentParser
from transformers import Wav2Vec2Processor, HubertModel
import torch.nn.functional as F
import soundfile as sf
import numpy as np
import torch

print("Loading the Wav2Vec2 Processor...")
wav2vec2_processor = Wav2Vec2Processor.from_pretrained("facebook/hubert-large-ls960-ft")
print("Loading the HuBERT Model...")
hubert_model = HubertModel.from_pretrained("facebook/hubert-large-ls960-ft")


def get_hubert_from_16k_wav(wav_16k_name):
    speech_16k, _ = sf.read(wav_16k_name)
    hubert = get_hubert_from_16k_speech(speech_16k)
    return hubert


@torch.no_grad()
def get_hubert_from_16k_speech(speech, device="cuda:0"):
    global hubert_model
    hubert_model = hubert_model.to(device)
    if speech.ndim == 2:
        speech = speech[:, 0]  # [T, 2] ==> [T,]
    input_values_all = wav2vec2_processor(speech, return_tensors="pt", sampling_rate=16000).input_values  # [1, T]
    input_values_all = input_values_all.to(device)
    # For long audio sequence, due to the memory limitation, we cannot process them in one run
    # HuBERT process the wav with a CNN of stride [5,2,2,2,2,2], making a stride of 320
    # Besides, the kernel is [10,3,3,3,3,2,2], making 400 a fundamental unit to get 1 time step.
    # So the CNN is euqal to a big Conv1D with kernel k=400 and stride s=320
    # We have the equation to calculate out time step: T = floor((t-k)/s)
    # To prevent overlap, we set each clip length of (K+S*(N-1)), where N is the expected length T of this clip
    # The start point of next clip should roll back with a length of (kernel-stride) so it is stride * N
    kernel = 400
    stride = 320
    clip_length = stride * 1000
    num_iter = input_values_all.shape[1] // clip_length
    expected_T = (input_values_all.shape[1] - (kernel - stride)) // stride
    res_lst = []
    for i in range(num_iter):
        if i == 0:
            start_idx = 0
            end_idx = clip_length - stride + kernel
        else:
            start_idx = clip_length * i
            end_idx = start_idx + (clip_length - stride + kernel)
        input_values = input_values_all[:, start_idx: end_idx]
        hidden_states = hubert_model.forward(input_values).last_hidden_state  # [B=1, T=pts//320, hid=1024]
        res_lst.append(hidden_states[0])
    if num_iter > 0:
        input_values = input_values_all[:, clip_length * num_iter:]
    else:
        input_values = input_values_all
    # if input_values.shape[1] != 0:
    if input_values.shape[1] >= kernel:  # if the last batch is shorter than kernel_size, skip it
        hidden_states = hubert_model(input_values).last_hidden_state  # [B=1, T=pts//320, hid=1024]
        res_lst.append(hidden_states[0])
    ret = torch.cat(res_lst, dim=0).cpu()  # [T, 1024]
    # assert ret.shape[0] == expected_T
    assert abs(ret.shape[0] - expected_T) <= 1
    if ret.shape[0] < expected_T:
        ret = torch.nn.functional.pad(ret, (0, 0, 0, expected_T - ret.shape[0]))
    else:
        ret = ret[:expected_T]
    return ret


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--wav', type=str, required=True)
    args = parser.parse_args()

    hubert_npy_name = args.wav.replace(".wav", "_hubert.npy")
    speech_16k, _ = sf.read(args.wav)
    hubert_hidden = get_hubert_from_16k_speech(speech_16k)

    feats = hubert_hidden
    window_size = 16
    audio_dim = 1024
    padding = window_size // 2
    feats = feats.view(-1, audio_dim).permute(1, 0).contiguous()  # [C, M]
    feats = feats.view(1, audio_dim, -1, 1)  # [1, C, M, 1]
    unfold_feats = F.unfold(feats, kernel_size=(window_size, 1), padding=(padding, 0),
                            stride=(2, 1))  # [1, C * window_size, M / 2 + 1]
    unfold_feats = unfold_feats.view(audio_dim, window_size, -1).permute(2, 1,
                                                                              0).contiguous()  # [C, window_size, M / 2 + 1]

    np.save(hubert_npy_name, unfold_feats.detach().numpy())
    print(f"Hubert extracted at {hubert_npy_name}")
