# DL4J Tester

This is a simple Scala project that trains a neural network on MNIST 
and measures the performance for training and inference.  The run.sh
script shows various GC tuning parameters you can use to limit the
memory usage and increase performance.

The script outputs the timing in a results.csv file for histogram 
visualizations.

## Requirements

- Java 8, 9, 11, or 12
- Maven 3.3.9 or greater

## Building

Run `build.sh`

By default a CPU-only version will be built.  To build a GPU version 
uncomment the commented out `nd4j-cuda-9.2-platform` and 
`deeplearning4j-cuda-9.2` dependencies in pom.xml and change the 
cuda version if necessary from 9.2 to either 9.0 or 10.0.


## Running

Run `run.sh`

You will see a log every time the GC kicks in and also see output 
like the following (Taken with jdk12 on a 2.6GHz Intel i7-8850H):

```
14:27:50.661 [main] INFO org.nd4j.nativeblas.NativeOpsHolder - Number of threads used for NativeOps: 12
14:27:50.939 [main] INFO org.nd4j.nativeblas.Nd4jBlas - Number of threads used for BLAS: 12
14:27:50.945 [main] INFO org.nd4j.linalg.api.ops.executioner.DefaultOpExecutioner - Backend used: [CPU]; OS: [Mac OS X]
14:27:50.945 [main] INFO org.nd4j.linalg.api.ops.executioner.DefaultOpExecutioner - Cores: [12]; Memory: [0.3GB];
14:27:50.945 [main] INFO org.nd4j.linalg.api.ops.executioner.DefaultOpExecutioner - Blas vendor: [MKL]
14:27:51.063 [main] INFO org.deeplearning4j.nn.multilayer.MultiLayerNetwork - Starting MultiLayerNetwork with WorkspaceModes set to [training: ENABLED; inference: ENABLED], cacheMode set to [NONE]
Using model:

=======================================================================
LayerName (LayerType)   nIn,nOut   TotalParams   ParamsShape           
=======================================================================
layer0 (DenseLayer)     784,768    602880        W:{784,768}, b:{1,768}
layer1 (DenseLayer)     768,256    196864        W:{768,256}, b:{1,256}
layer2 (DenseLayer)     256,10     2570          W:{256,10}, b:{1,10}  
layer3 (LossLayer)      -,-        0             -                     
-----------------------------------------------------------------------
            Total Parameters:  802314
        Trainable Parameters:  802314
           Frozen Parameters:  0
=======================================================================

Measuring training performance...

... GC logs here ...

Training took 10 seconds
(3.3333333333333335 per epoch)
Measuring inference performance...

... GC logs here ...

Measuring inference performance...
Min (0.043304 ms per sample)
Mean (0.057155 ms per sample)
Median (0.053707 ms per sample)
90 Percentile (0.068776 ms per sample)
99 Percentile (0.11381 ms per sample)
99.9 Percentile (0.19192 ms per sample)
99.99 Percentile (0.568692 ms per sample)
99.999 Percentile (0.729292 ms per sample)
99.9999 Percentile (1.130841 ms per sample)
Max (1.130841 ms per sample)
Average 17496.282040066486 inferences per second
Took 57 seconds for 1000000 inferences (17543.859649122805 inferences per second)

```

A python script (`measure_keras.py`) is also provided that does more/less the same thing for comparison:

 ```
 Measuring training performance...
 
 ...
 
 Epoch 1/3
 2019-06-16 22:01:06.268263: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
 60000/60000 [==============================] - 4s 59us/step - loss: 0.2379 - acc: 0.9260
 Epoch 2/3
 60000/60000 [==============================] - 3s 54us/step - loss: 0.1128 - acc: 0.9660
 Epoch 3/3
 60000/60000 [==============================] - 3s 53us/step - loss: 0.0847 - acc: 0.9737
 Training took 10.215353036066517 seconds
 (3.405117678688839 per epoch)

 Measuring inference performance...
 Min (0.215769 ms per sample)
 Mean (0.289890 ms per sample)
 Median (0.283223 ms per sample)
 90 Percentile (0.328217 ms per sample)
 99 Percentile (0.381326 ms per sample)
 99.9 Percentile (0.642981 ms per sample)
 99.99 Percentile (1.000424 ms per sample)
 99.999 Percentile (1.208193 ms per sample)
 99.9999 Percentile (1.795472 ms per sample)
 Max (1.795472 ms per sample)
 Average 3449.585665 inference per second
 Took 291.459642 seconds for 1000000.000000 inferences (3431.006758 inferences per second)

 
 ```
 