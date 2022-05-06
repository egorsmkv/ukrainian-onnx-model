import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType

model_fp32 = 'asr.onnx'
model_quant = 'asr.quant.onnx'

quantized_model = quantize_dynamic(model_fp32, model_quant, weight_type=QuantType.QUInt8)

print('Finished')
