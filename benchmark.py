from sdgclassification.benchmark import Benchmark
from bert_inference_binary import predict_sdgs

benchmark = Benchmark(predict_sdgs)
benchmark.run()