import onnxruntime
import numpy as np

from scipy.io import wavfile

AUDIO_MAXLEN = 100000


def _normalize(x):
    """
    You must call this before padding.
    Code from https://github.com/vasudevgupta7/gsoc-wav2vec2/blob/main/src/wav2vec2/processor.py#L101
    Fork TF to numpy
    """

    # -> (1, seqlen)
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)

    return np.squeeze((x - mean) / np.sqrt(var + 1e-5))


def remove_adjacent(item):
    """
    Code from https://stackoverflow.com/a/3460423
    """

    nums = list(item)
    a = nums[:1]
    for item in nums[1:]:
        if item != a[-1]:
            a.append(item)
    return ''.join(a)


def fix(s):
    """
    Replace spaces and strip text.
    """

    return s.replace('|', ' ').strip()


def asr(ort_s, vocab_dict, path):
    """
    Code from https://github.com/vasudevgupta7/gsoc-wav2vec2/blob/main/notebooks/wav2vec2_onnx.ipynb
    """

    _, data = wavfile.read(path)
    speech = np.array(data, dtype=np.float32)
    speech = _normalize(speech)[None]
    padding = np.zeros((speech.shape[0], AUDIO_MAXLEN - speech.shape[1]))
    speech = np.concatenate([speech, padding], axis=-1).astype(np.float32)

    ort_inputs = {"modelInput": speech}
    ort_outs = ort_s.run(None, ort_inputs)
    prediction = np.argmax(ort_outs, axis=-1)

    # Text post-processing
    t1 = ''.join([vocab_dict[i] for i in list(prediction[0][0])])

    return fix(''.join([remove_adjacent(j) for j in t1.split("[PAD]")]))


if __name__ == '__main__':
    # Load the vocabulary
    with open("vocab.json", "r", encoding="utf-8-sig") as f:
        d = eval(f.read())
        vocab = dict((v, k) for k, v in d.items())

    # Load the onnx model
    ort_session = onnxruntime.InferenceSession('onnx-uk-1b/asr.onnx')

    # Recognize a file
    transcription = asr(ort_session, vocab, 'files/sound2.wav')
    print(transcription)  # it will be: аня сполучені штати над важливий стратегічний
