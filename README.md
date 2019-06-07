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

==============================================================================
LayerName (LayerType)         nIn,nOut   TotalParams   ParamsShape            
==============================================================================
cnn1 (ConvolutionLayer)       1,20       520           b:{1,20}, W:{20,1,5,5} 
maxpool1 (SubsamplingLayer)   -,-        0             -                      
cnn2 (ConvolutionLayer)       20,50      25050         b:{1,50}, W:{50,20,5,5}
maxpool2 (SubsamplingLayer)   -,-        0             -                      
ffn1 (DenseLayer)             800,500    400500        W:{800,500}, b:{1,500} 
output (OutputLayer)          500,10     5010          W:{500,10}, b:{1,10}   
------------------------------------------------------------------------------
            Total Parameters:  431080
        Trainable Parameters:  431080
           Frozen Parameters:  0
==============================================================================

... GC logs here ...

Training took 24 seconds
(24.0 per epoch)

... GC logs here ...

Min (795 µs per sample)
Mean (1563 µs per sample)
Median (1200 µs per sample)
90 Percentile (2229 µs per sample)
99 Percentile (5175 µs per sample)
99.9 Percentile (24037 µs per sample)
99.99 Percentile (83160 µs per sample)
99.999 Percentile (128223 µs per sample)
99.9999 Percentile (128223 µs per sample)
99.99999 Percentile (128223 µs per sample)
Max (128223 µs per sample)
```
