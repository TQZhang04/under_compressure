import os
import numpy as np
import librosa
import jiwer
import soundfile as sf


def compress_decompress(filepaths,
                        codec,
                        filetypes,
                        bitrates=[128],
                        folder_override=""):
    """Compress and then decompress all files in a list of filepaths.

    Args:
        filepaths (list): list of paths
        codec (str): codec to use. must be compatible with ffmpeg
        filetypes (tuple): tuple containing:
            - filetypes[0] = original filetype, ex: 'wav'
            - filetypes[1] = it's being converted to, ex: 'mp3'
        options (str, optional): other options for ffmpeg. 
            Defaults to "".
        bitrates (list, optional): list of bitrates to compress at. 
            Defaults to [256, 192, 160, 128, 96, 64, 32, 16, 8].
    """
    filetype_fr, filetype_to = filetypes
    for fp in filepaths:
        filename = fp.split("/")[-1][:-4]
        for bitrate in bitrates:
            new_filename = f"{filename}+{bitrate}kbps"
            new_path = f"audio/{filetype_to}{folder_override}/{new_filename}"

            arg1 = "--yes" if os.path.isfile(
                f"{new_path}.{filetype_to}") else ""
            arg2 = "--yes" if os.path.isfile(
                f"{new_path}.{filetype_fr}") else ""
            if arg1 or arg2:
                print(f"File {new_path}.{filetype_to} already exists")
                continue

            try:
                os.system(
                    f'ffmpeg -i {fp} -c:a {codec} -b:a {bitrate}k {new_path}.{filetype_to} {arg1}')
                os.system(
                    f'ffmpeg -i {new_path}.{filetype_to} -vn {new_path}.{filetype_fr} {arg2}')
            except BaseException as e:
                print("Error with file at", fp)
                print("Exception:", e)
                continue


def get_noise_multiplier(signal, noise, snr=20.):
    """Function to get a, the multiplier for noise based on a given 
    signal-noise-ratio

    Args:
        signal (iterable): signal
        noise (iterable): noise
        snr (float, optional): signal-to-noise ratio. Defaults to 20..

    Returns:
        float: multiplier for noise
    """
    signal = np.array(signal)
    noise = np.array(noise)

    MS_sig = (signal ** 2.).sum() / len(signal)
    MS_noise = (noise ** 2.).sum() / len(noise)

    temp = 10. ** (snr / 10.)
    a = MS_sig / (MS_noise * temp)
    return a


def add_noise(signal, noise, snr=20, sr=44100, len_sec=None, filepath=None, randomseed=None):
    """Adds noise to signal

    Args:
        signal (ndarray): signal
        noise (ndarray): noise. should be at least as long as signal
        snr (int, optional): signal-to-noise-ratio in decibels. Defaults to 20.
        sr (int, optional): sample rate in Hz. Defaults to 44100.
        len_sec (int, optional): length of the output in seconds. If not 
            specified, truncated to length of signal.
        filepath (str, optional): path to write the output to. If None, the 
            file is not written to disk and only returned. Defaults to None.
        randomseed (int, optional): random seed for reproducability. If None, 
            no seed is used.

    Returns:
        ndarray: signal with noise added
    """
    if randomseed:
        np.random.seed(randomseed)
    a = get_noise_multiplier(signal, noise, snr=snr)
    num_samples = len_sec * sr if len_sec else len(signal)
    noise_start = np.random.randint(0, len(noise) - num_samples)
    to_add = noise[noise_start: noise_start + num_samples]
    result = signal[:num_samples] + a * to_add
    if filepath:
        sf.write(filepath, result, samplerate=sr)
    return result


def wer(ref, hyp):
    """ word error rate function built with code from this article: 
        https://medium.com/@johnidouglasmarangon/how-to-calculate-the-word-error-rate-in-python-ce0751a46052

    Args:
        ref (str): reference, ground truth
        hyp (str): hypothesis, model output/prediction

    Returns:
        float: word error rate as a float between 0 and 1
    """
    transforms = jiwer.Compose(
        [
            jiwer.ExpandCommonEnglishContractions(),
            jiwer.RemoveEmptyStrings(),
            jiwer.ToLowerCase(),
            jiwer.RemoveMultipleSpaces(),
            jiwer.Strip(),
            jiwer.RemovePunctuation(),
            jiwer.ReduceToListOfListOfWords(),
        ]
    )
    wer = jiwer.wer(
        ref,
        hyp,
        truth_transform=transforms,
        hypothesis_transform=transforms,
    )
    return wer
