import torch
def wav_segmentation(in_sig, framesamp=320, hopsamp=160, windows = 1):
    ch, sigLength = in_sig.shape
    M = (sigLength - framesamp) // hopsamp + 1
    a = torch.zeros((ch, M, framesamp), dtype=torch.float32)
    startpoint = 0
    for m in range(M):
        a[:, m, :] = in_sig[:, startpoint:startpoint+framesamp]
        startpoint = startpoint + hopsamp
    return a * windows