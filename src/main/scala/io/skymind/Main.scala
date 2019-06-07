package io.skymind

import java.io.{File, PrintWriter}
import java.util.concurrent.TimeUnit

import org.apache.commons.lang3.time.StopWatch
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator
import org.deeplearning4j.nn.adapters.ArgmaxAdapter
import org.deeplearning4j.nn.api.OutputAdapter
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.inputs.InputType
import org.deeplearning4j.nn.conf.layers.{SubsamplingLayer, _}
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.learning.config.AMSGrad
import org.nd4j.linalg.lossfunctions.LossFunctions
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction

import scala.io.Source

object Main extends App {
  val batchSize = 256
  val nClasses = 10
  val trainData = new MnistDataSetIterator(batchSize, true, 42)
  val testData = new MnistDataSetIterator(batchSize, false, 42)


  val model = makeLeNetDl4JModel()

  println(model.summary)

  val sample = Nd4j.zeros(1L, 28L * 28L)
  model.output(sample) // warmup

  println("Training...")
  val stopWatch: StopWatch = StopWatch.createStarted()

  val epochs = 1
  for (i <- 0 until epochs) {
    println(s"Epoch $i")
    model.fit(trainData)
  }

  stopWatch.stop()

  println(s"Training took ${stopWatch.getTime(TimeUnit.SECONDS)} seconds")
  println(s"(${stopWatch.getTime(TimeUnit.SECONDS).toDouble / epochs.toDouble} per epoch)")

  //private val inference: ParallelInference = new ParallelInference.Builder(model).build()

  val inferenceCount = 100000
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

    inferenceTimes(i) = stopWatch.getNanoTime

    assert(result > 0)
  }

  val sortedTimes = inferenceTimes.sorted

  val inferenceMean = sortedTimes.sum / sortedTimes.length
  val inferenceMin = sortedTimes.min
  val inferenceMax = sortedTimes.max

  println(s"Min (${inferenceMin / 1000} µs per sample)")
  println(s"Mean (${inferenceMean / 1000} µs per sample)")
  println(s"Median (${sortedTimes(inferenceTimes.length / 2) / 1000} µs per sample)")
  println(s"90 Percentile (${percentile(90)(sortedTimes) / 1000} µs per sample)")
  println(s"99 Percentile (${percentile(99)(sortedTimes) / 1000} µs per sample)")
  println(s"99.9 Percentile (${percentile(99.9)(sortedTimes) / 1000} µs per sample)")
  println(s"99.99 Percentile (${percentile(99.99)(sortedTimes) / 1000} µs per sample)")
  println(s"99.999 Percentile (${percentile(99.999)(sortedTimes) / 1000} µs per sample)")
  println(s"99.9999 Percentile (${percentile(99.9999)(sortedTimes) / 1000} µs per sample)")
  println(s"99.99999 Percentile (${percentile(99.99999)(sortedTimes) / 1000} µs per sample)")
  println(s"Max (${inferenceMax / 1000} µs per sample)")

  new PrintWriter("results.csv") { write(sortedTimes.mkString("\n")); close() }

  def makeLeNetDl4JModel() = {
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

  def makeSimpleDl4JModel() = {
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
          .nOut(nClasses)
          .activation(Activation.SOFTMAX)
          .weightInit(WeightInit.XAVIER)
          .build(),
        new LossLayer.Builder(LossFunction.MCXENT).build()
      ).build()


    val model = new MultiLayerNetwork(conf)
    //model.setListeners(new PerformanceListener(1000, true))
    model.init()


    model
  }

  def collectToArray(dataSetIterator: DataSetIterator): Array[DataSet] = {
    var ret: List[DataSet] = Nil

    while (dataSetIterator.hasNext) {
      ret ::= dataSetIterator.next()
    }

    ret.reverse.toArray
  }

  def percentile(p: Double)(seq: Seq[Long]) = {
    assert(seq.min == seq.head && seq.max == seq.last)

    val k = math.ceil((seq.length - 1) * (p / 100.0)).toInt
    seq(k)
  }

  class SimpleArgMaxAdapter extends OutputAdapter[Int] {

    override def apply(outputs: INDArray*): Int = {
        val array = outputs(0)
        Nd4j.argMax(array, Integer.MAX_VALUE).getDouble(0L).toInt
    }
  }
}