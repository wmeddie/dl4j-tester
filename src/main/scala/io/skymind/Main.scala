package io.skymind

import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator
import org.deeplearning4j.eval.Evaluation
import org.deeplearning4j.nn.conf.{MultiLayerConfiguration, NeuralNetConfiguration}
import org.deeplearning4j.nn.conf.inputs.InputType
import org.deeplearning4j.nn.conf.layers._
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.optimize.listeners.PerformanceListener
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.learning.config.AMSGrad
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction

object Main extends App {
  val batchSize = 256
  val nClasses = 10

  val conf = new NeuralNetConfiguration.Builder()
    .updater(new AMSGrad(0.01))
    .l2(5e-4)
    .weightInit(WeightInit.XAVIER)
    .activation(Activation.RELU).list(
      new ConvolutionLayer.Builder(5, 5).nOut(20).build(),
      new SubsamplingLayer.Builder(PoolingType.MAX).kernelSize(2, 2).build(),
      new ConvolutionLayer.Builder(5, 5).nOut(50).build(),
      new SubsamplingLayer.Builder(PoolingType.MAX).kernelSize(2, 2).build(),
      new DenseLayer.Builder().nOut(500).activation(Activation.RELU).build(),
      new OutputLayer.Builder(LossFunction.NEGATIVELOGLIKELIHOOD).nOut(nClasses).activation(Activation.SOFTMAX).build()
    )
    .setInputType(InputType.convolutionalFlat(28, 28, 1))
    .build()

  val model = new MultiLayerNetwork(conf)
  model.setListeners(new PerformanceListener(10, true))
  model.init()

  val trainData = new MnistDataSetIterator(batchSize, true, 42)

  println(model.summary)

  val start = System.nanoTime()
  model.fit(trainData, 3)
  val end = System.nanoTime()

  val testData = new MnistDataSetIterator(batchSize, false, 42)
  val eval = model.evaluate(testData)
  println(eval)

  println("Took " + ((end - start) / 1000000000) + " seconds")
}