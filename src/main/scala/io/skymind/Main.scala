package io.skymind

import java.util.concurrent.TimeUnit

import org.apache.commons.lang3.time.StopWatch
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator
import org.deeplearning4j.nn.adapters.ArgmaxAdapter
import org.deeplearning4j.nn.api.OutputAdapter
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.layers._
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.optimize.listeners.PerformanceListener
import org.deeplearning4j.parallelism.{ParallelInference, ParallelWrapper}
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.api.memory.conf.WorkspaceConfiguration
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.learning.config.AMSGrad
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction

object Main extends App {
  val batchSize = 256
  val nClasses = 10
  val trainData = new MnistDataSetIterator(batchSize, true, 42)
  val testData = new MnistDataSetIterator(batchSize, false, 42)


  val model = makeDl4JModel()

  //model.addListeners(new PerformanceListener(10, true))

  println(model.summary)

  val sample = Nd4j.zeros(1L, 28L * 28L)
  model.output(sample) // warmup

  val stopWatch: StopWatch = StopWatch.createStarted()
  model.output(sample)
  stopWatch.stop()

  println(s"Time for inference: ${stopWatch.getTime(TimeUnit.MICROSECONDS)} microseconds")

  println("Training...")
  val start = System.nanoTime()

  val epochs = 3
  for (i <- 0 to epochs) {
    println(s"Epoch $i")
    model.fit(trainData)
  }

  val end = System.nanoTime()

  val seconds = (end - start) / 1000000000.0
  println(s"Training took $seconds seconds")
  println(s"(${seconds.toDouble / epochs.toDouble} per epoch)")

  //private val inference: ParallelInference = new ParallelInference.Builder(model).build()

  val inferenceStart = System.nanoTime()
  val inferenceTimes = new Array[Long](1000000)

  val outputAdapter: OutputAdapter[Array[Int]] = new ArgmaxAdapter()

  for (i <- 0 until 1000000) {
    val array = Nd4j.rand(Array(1, 28 * 28))

    val outputStart = System.nanoTime()
    val result = model.output(array, null, null, outputAdapter)
    val outputEnd = System.nanoTime()

    inferenceTimes(i) = outputEnd - outputStart

    assert(result != null)
  }

  val sortedTimes = inferenceTimes.sorted

  val inferenceEnd = System.nanoTime()
  val inferenceSeconds = (inferenceEnd - inferenceStart) / 1000000000.0
  println(s"Inference took $inferenceSeconds seconds")

  val inferenceMean = sortedTimes.sum / sortedTimes.length
  val inferenceMin = sortedTimes.min
  val inferenceMax = sortedTimes.max

  println(s"Min (${inferenceMin / 1000} µs per sample)")
  println(s"Mean (${inferenceMean / 1000} µs per sample)")
  println(s"Median (${sortedTimes(inferenceTimes.length / 2) / 1000} µs per sample")
  println(s"90 Percentile (${percentile(90)(sortedTimes) / 1000} µs per sample)")
  println(s"99 Percentile (${percentile(99)(sortedTimes) / 1000} µs per sample)")
  println(s"99.9 Percentile (${percentile(99.9)(sortedTimes) / 1000} µs per sample)")
  println(s"99.99 Percentile (${percentile(99.99)(sortedTimes) / 1000} µs per sample)")
  println(s"99.999 Percentile (${percentile(99.999)(sortedTimes) / 1000} µs per sample)")
  println(s"99.9999 Percentile (${percentile(99.9999)(sortedTimes) / 1000} µs per sample)")
  println(s"99.99999 Percentile (${percentile(99.99999)(sortedTimes) / 1000} µs per sample)")
  println(s"Max (${inferenceMax / 1000} µs per sample)")

  /*
  for (i <- 0 to 25) {
    val start = System.nanoTime()

    trainData.foreach { ds =>
      model.fit(ds)
    }

    /*val evaluation = new Evaluation()
    testData.foreach { ds =>
      evaluation.eval(ds.getLabels, model.output(ds.getFeatures))
    }
    println("Eval ACC: " + evaluation.accuracy())
    */
    val end = System.nanoTime()

    println("Took " + ((end - start) / 1000000000.0) + " seconds")
  }
  */

  def makeDl4JModel() = {
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
          .nOut(10)
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
}