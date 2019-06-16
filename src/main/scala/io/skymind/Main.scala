package io.skymind

import java.io.PrintWriter
import java.util.concurrent.TimeUnit

import org.apache.commons.lang3.time.StopWatch
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator
import org.deeplearning4j.nn.api.OutputAdapter
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.inputs.InputType
import org.deeplearning4j.nn.conf.layers.{SubsamplingLayer, _}
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.learning.config.AMSGrad
import org.nd4j.linalg.lossfunctions.LossFunctions
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction

object Main {
  val epochs = 3
  val batchSize = 256
  val nClasses = 10
  val inferenceCount = 1000000

  def main(args: Array[String]): Unit = {
    // This is great when using Dl4j in a notebook, but not useful here.
    Nd4j.getMemoryManager.togglePeriodicGc(false)

    //val model = makeLeNetDl4JModel()
    val model = makeSimpleDl4JModel()

    println("Using model:")
    println(model.summary)
    println("Measuring training performance...")

    val trainTime = measureTraining(model)

    println(s"Training took $trainTime seconds")
    println(s"(${trainTime.toDouble / epochs.toDouble} per epoch)")

    println("Measuring inference performance...")

    val timer = StopWatch.createStarted()
    val inferenceTimes = measureInference(model)
    val measurementTime = timer.getTime(TimeUnit.SECONDS)

    val sortedTimes = inferenceTimes.sorted
    val inferenceMean = sortedTimes.sum / sortedTimes.length
    val inferenceMin = sortedTimes.min
    val inferenceMax = sortedTimes.max
    val oneNine = percentile(90, sortedTimes)
    val twoNines = percentile(99, sortedTimes)
    val threeNines = percentile(99.9, sortedTimes)
    val fourNines = percentile(99.99, sortedTimes)
    val fiveNines = percentile(99.999, sortedTimes)
    val sixNines = percentile(99.9999, sortedTimes)
    val ns2ms = 1000000.0
    val ns2s = ns2ms * 1000.0

    println(s"Min (${inferenceMin / ns2ms} ms per sample)")
    println(s"Mean (${inferenceMean / ns2ms} ms per sample)")
    println(s"Median (${sortedTimes(inferenceTimes.length / 2) / ns2ms} ms per sample)")
    println(s"90 Percentile (${oneNine / ns2ms} ms per sample)")
    println(s"99 Percentile (${twoNines / ns2ms} ms per sample)")
    println(s"99.9 Percentile (${threeNines / ns2ms} ms per sample)")
    println(s"99.99 Percentile (${fourNines / ns2ms} ms per sample)")
    println(s"99.999 Percentile (${fiveNines / ns2ms} ms per sample)")
    println(s"99.9999 Percentile (${sixNines / ns2ms} ms per sample)")
    println(s"Max (${inferenceMax / ns2ms} ms per sample)")
    println(s"Average ${ns2s / inferenceMean} inferences per second")
    println(s"Took $measurementTime seconds for $inferenceCount inferences (${inferenceCount.toDouble / measurementTime} inferences per second)")

    new PrintWriter("results.csv") {
      write(sortedTimes.mkString("\n"))
      close()
    }
  }

  private def measureTraining(model: MultiLayerNetwork): Long = {
    val trainData = new MnistDataSetIterator(batchSize, true, 42)
    //val testData = new MnistDataSetIterator(batchSize, false, 42)

    val stopWatch: StopWatch = StopWatch.createStarted()
    model.fit(trainData, epochs)

    stopWatch.getTime(TimeUnit.SECONDS)
  }

  private def measureInference(model: MultiLayerNetwork): Array[Long] = {
    val stopWatch = new StopWatch()
    val inferenceTimes = new Array[Long](inferenceCount)
    val outputAdapter: OutputAdapter[Int] = new SimpleArgMaxAdapter()
    val array = Nd4j.zeros(1L, 28 * 28L)

    // warmup some caches
    for (_ <- 0 until 100) {
      model.output(array, null, null, outputAdapter)
    }

    for (i <- 0 until inferenceCount) {
      stopWatch.reset()

      stopWatch.start()
      val result = model.output(array, null, null, outputAdapter)
      stopWatch.stop()

      assert(result >= 0)

      inferenceTimes(i) = stopWatch.getNanoTime
    }

    inferenceTimes
  }


  //noinspection ScalaUnusedSymbol
  private def makeLeNetDl4JModel(): MultiLayerNetwork = {
    val conf = new NeuralNetConfiguration.Builder()
      .seed(42)
      .activation(Activation.IDENTITY)
      .weightInit(WeightInit.XAVIER)
      .l2(5e-4)
      .updater(new AMSGrad(0.004))
      .list(
      new ConvolutionLayer.Builder(Array[Int](5, 5), Array[Int](1, 1))
        .name("cnn1")
        .nIn(1)
        .nOut(20)
        .build(),
      new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX, Array[Int](2, 2), Array[Int](2, 2))
        .name("maxpool1")
        .build(),
      new ConvolutionLayer.Builder(Array[Int](5, 5), Array[Int](1, 1))
        .name("cnn2")
        .nOut(50)
        .build(),
      new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX, Array[Int](2, 2), Array[Int](2, 2))
        .name("maxpool2")
        .build(),
      new DenseLayer.Builder()
        .name("ffn1")
        .activation(Activation.RELU)
        .nOut(500)
        .build(),
      new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
        .name("output")
        .nOut(nClasses)
        .activation(Activation.SOFTMAX) // radial basis function required
        .build())
      .setInputType(InputType.convolutionalFlat(28, 28, 1))
      .build()

    val model = new MultiLayerNetwork(conf)
    model.init()

    model
  }

  private def makeSimpleDl4JModel(): MultiLayerNetwork = {
    val conf = new NeuralNetConfiguration.Builder()
      .seed(42) // include a random seed for reproducibility
      .updater(new AMSGrad(0.004)) //specify the updating method and learning rate.
      .list(
      new DenseLayer.Builder()
        .nIn(28 * 28)
        .nOut(768)
        .activation(Activation.RELU)
        .weightInit(WeightInit.RELU)
        .dropOut(1 - 0.2)
        .build(),
      new DenseLayer.Builder()
        .nIn(768)
        .nOut(256)
        .activation(Activation.RELU)
        .weightInit(WeightInit.RELU)
        .dropOut(1 - 0.2)
        .build(),
      new DenseLayer.Builder()
        .nIn(256)
        .nOut(nClasses)
        .activation(Activation.SOFTMAX)
        .weightInit(WeightInit.XAVIER)
        .build(),
      new LossLayer.Builder(LossFunction.MCXENT).build()
    ).build()


    val model = new MultiLayerNetwork(conf)
    model.init()

    model
  }

  private def percentile(p: Double, seq: Seq[Long]) = {
    assert(seq.min == seq.head && seq.max == seq.last)

    val k = math.ceil((seq.length - 1) * (p / 100.0)).toInt

    seq(k)
  }

  class SimpleArgMaxAdapter extends OutputAdapter[Int] {
    override def apply(outputs: INDArray*): Int = {

      //Nd4j.argMax(outputs(0), 0).getDoubleUnsafe(0).toInt

      // Note: the argMax above creates a intermediate 10 element INDArray
      // outside of a workspace which adds a small amount of garbage that
      // is easy to avoid with the loop below.

      val array = outputs(0)
      var max = -1.0
      var maxIndex = 0L

      for (i <- 0 until 10) {
        val x = array.getDoubleUnsafe(i)
        if (x > max) {
          max = x
          maxIndex = i
        }
      }
      maxIndex.toInt
    }
  }
}