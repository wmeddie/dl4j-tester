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

Min (0.041094 ms per sample)
Mean (0.05349 ms per sample)
Median (0.050756 ms per sample)
90 Percentile (0.061588 ms per sample)
99 Percentile (0.099401 ms per sample)
99.9 Percentile (0.171543 ms per sample)
99.99 Percentile (1.084526 ms per sample)
99.999 Percentile (2.312762 ms per sample)
99.9999 Percentile (2.891824 ms per sample)
Max (2.891824 ms per sample)

```
