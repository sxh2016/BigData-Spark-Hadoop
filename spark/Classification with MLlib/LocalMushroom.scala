import breeze.linalg.{Vector, DenseVector, squaredDistance}
import org.apache.spark.mllib.classification.LogisticRegressionWithSGD
import org.apache.spark.mllib.classification.SVMWithSGD
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.classification.NaiveBayes
import org.apache.spark.mllib.tree.DecisionTree
import org.apache.spark.mllib.tree.configuration.Algo
import org.apache.spark.mllib.tree.impurity.Entropy
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.SparkContext._
object LocalMushroom {
  def main(args : Array[String]){
    //val sparkConf = new SparkConf().setAppName("Scala-Spark")
    val sc = new SparkContext("local[2]","Scala-Spark")
    //Read data from local
    val mushroom = sc.textFile("data/agaricus-lepiota.data")
    val data = mushroom.map(line => line.split(','))
    val record = data.map(line => line.flatMap(x => x.map(_.toDouble -'a'.toDouble)))
    
    //preprocess the data and get the input
    val Inputdata = record.map { line =>
      val mrmdata = line.map(_.toDouble)
      val label = if(mrmdata(0) == 15.0) 1.0 else 0.0
      val feature = mrmdata.slice(1,line.size).map(d => if(d <0) 0.0 else d.toDouble)
      LabeledPoint(label,Vectors.dense(feature))
    }
    println("the label and feature are ready")
    
    //split the whole dataset into trainingset and testingset
    val splitset = Inputdata.randomSplit(Array(0.8,0.2), seed = 11L)
    val trainingset = splitset(0)
    val testingset = splitset(1)
    println("training set and testing set are ready, the radio is 8:2")
    
    //train the Naive-Bayes model
    val NBmodel = NaiveBayes.train(trainingset, lambda = 1.0, modelType = "multinomial")
    val NBPredictionAndLabel = testingset.map(p => (NBmodel.predict(p.features),p.label))
    //show a sample output
    NBPredictionAndLabel.take(10).foreach(println)
    println("Naive Bayes model training and predict")
    
    //train the Decision Tree model
    val maxTreeDepth = 5
    val dtModel = DecisionTree.train(Inputdata, Algo.Classification, Entropy,maxTreeDepth)
    val DTPredictionAndLabel = testingset.map(p => (dtModel.predict(p.features),p.label))
    println("Decition Tree model training and predict")
    
    //train the linear model
    val numIterations = 10
    val lrModel = LogisticRegressionWithSGD.train(Inputdata, numIterations)
    val LRPredictionAndLabel = lrModel.predict(Inputdata.map(p => p.features))
    
    //train the support vector machine model
    val svmModel = SVMWithSGD.train(Inputdata, numIterations)
    val SVMPredictionAndLabel = svmModel.predict(Inputdata.map(p => p.features))
    
    //calculate the accuracy of prediction
    //val NBaccuracy = 1.0 * NBPredictionAndLabel.filter(x => x._1 == x._2).count()/testingset.count()
    val nbTotalCorrect = testingset.map { point =>
    	if (NBmodel.predict(point.features) == point.label) 1 else 0
    }.sum
    val nbAccuracy = nbTotalCorrect / testingset.count()
    //val DTaccuracy = 1.0 * DTPredictionAndLabel.filter(x => x._1 == x._2).count()/testingset.count()
    val dtTotalCorrect = testingset.map { point =>
    	val score = dtModel.predict(point.features)
    	val predicted = if (score > 0.5) 1 else 0
    	if (predicted == point.label) 1 else 0
    }.sum
    val dtAccuracy = dtTotalCorrect / testingset.count()
    
    val lrTotalCorrect = testingset.map { point =>
    	if (lrModel.predict(point.features) == point.label) 1 else 0
    }.sum
    val lrAccuracy = lrTotalCorrect / testingset.count()
    
    val svmTotalCorrect = testingset.map { point =>
    	if (svmModel.predict(point.features) == point.label) 1 else 0
    }.sum
    val svmAccuracy = svmTotalCorrect / testingset.count()
    
    //show the accuracy of predictions of each models
    println("The Naive-Bayes accuracy is :" + nbAccuracy)
    println("The Dicesion-Tree accuracy is :" + dtAccuracy)
    println("The Leanier Regresion accuracy is :" + lrAccuracy)
    println("The Support Vector Machine accuracy is :" + svmAccuracy)
    sc.stop()
  }
}