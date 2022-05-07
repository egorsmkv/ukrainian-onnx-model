import tensorflow as tf
import keras
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


def asr(m, vocab_dict, path):
    """
    Code from https://github.com/vasudevgupta7/gsoc-wav2vec2/blob/main/notebooks/wav2vec2_onnx.ipynb
    """

    _, data = wavfile.read(path)
    speech = np.array(data, dtype=np.float32)
    speech = _normalize(speech)[None]
    padding = np.zeros((speech.shape[0], AUDIO_MAXLEN - speech.shape[1]))
    speech = np.concatenate([speech, padding], axis=-1).astype(np.float32)

    infer = m.signatures["serving_default"]
    out = infer(tf.constant(speech))['modelOutput']
    prediction = np.argmax(out, axis=-1)

    # Text post-processing
    t1 = ''.join([vocab_dict[i] for i in list(prediction[0])])

    return fix(''.join([remove_adjacent(j) for j in t1.split("[PAD]")]))


if __name__ == '__main__':
    # Load the vocabulary
    with open("vocab.json", "r", encoding="utf-8-sig") as f:
        d = eval(f.read())
        vocab = dict((v, k) for k, v in d.items())

    # Load the model
    model_dir = './tf_uk_300m_model'
    m = tf.saved_model.load(model_dir, tags=None, options=None)

    # Recognize a file
    transcription = asr(m, vocab, 'files/sound.wav')
    print(transcription)
