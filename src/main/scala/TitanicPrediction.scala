import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.sql.functions._
import org.apache.spark.ml.classification.RandomForestClassifier
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.Pipeline
import org.apache.spark.sql.functions._


import org.apache.spark.{SparkContext, SparkConf }


object TitanicPrediction {


    def main(args: Array[String]): Unit = {

      val conf= new SparkConf().setMaster("local[*]").setAppName("TitanicPredictionSurvival")
      val sc= new SparkContext(conf)


      // Initialize a Spark session
      val spark = SparkSession.builder().config(conf).getOrCreate()

      // Load the training and testing datasets
      val trainData = spark.read.option("header", "true").csv("/Users/mariagloriaraquelobono/Fall2023/TitanicPrediction/src/main/scala/train (1).csv")
      val testData = spark.read.option("header", "true").csv("/Users/mariagloriaraquelobono/Fall2023/TitanicPrediction/src/test/scala/test.csv")

      // Data Preprocessing
      val data = trainData
        .withColumn("Pclass", trainData("Pclass").cast("int"))
        .withColumn("Age", trainData("Age").cast("int"))
        .withColumn("Fare", trainData("Fare").cast("int"))
        .withColumn("Survived", trainData("Survived").cast("int")) // Convert Survived to int

      // Handle missing values (you can use other strategies as needed)
      val dataCleaned = data.na.fill(0.0, Seq("Age", "Fare"))


      // Feature Engineering
      val trainDataWithFeatures = dataCleaned
      //val trainDataWithFeatures = trainData
        // Create a new feature 'FamilySize' by summing 'SibSp' and 'Parch'
        .withColumn("FamilySize", col("SibSp") + col("Parch"))
        // Extract the title from the 'Name' column
        .withColumn("Title", regexp_extract(col("Name"), "(Mrs\\.|Mr\\.|Miss\\.|Master\\.|Rev\\.|Dr\\.|Major\\.|Capt\\.|Col\\.|Lady\\.|Sir\\.|Jonkheer\\.|Don\\.)", 0))

        // Use StringIndexer to convert categorical 'Sex' and 'Embarked' columns to numerical
        .transform { df =>
          val sexIndexer = new StringIndexer()
            .setInputCol("Sex")
            .setOutputCol("SexIndex")
            .setHandleInvalid("skip") // Skip rows with invalid values
            .fit(df)

          val embarkedIndexer = new StringIndexer()
            .setInputCol("Embarked")
            .setOutputCol("EmbarkedIndex")
            .setHandleInvalid("skip") // Skip rows with invalid values
            .fit(df)

          val titleIndexer = new StringIndexer()
            .setInputCol("Title")
            .setOutputCol("TitleIndex")
            .setHandleInvalid("skip") // Skip rows with invalid values
            .fit(df)


          //sexIndexer.transform(embarkedIndexer.transform(df))
          titleIndexer.transform(embarkedIndexer.transform(sexIndexer.transform(df)))
        }
        // Select only the relevant feature columns
        .select("PassengerId", "Pclass", "Age", "FamilySize", "TitleIndex", "SexIndex", "Fare", "EmbarkedIndex", "Survived")

      // Remove unnecessary columns
      val selectedFeatures = trainDataWithFeatures.drop("Name", "SibSp", "Parch", "Ticket", "Cabin")


      // Show the resulting dataset
      selectedFeatures.show()



      // Define the machine learning pipeline

      // Define feature columns (including the ones from feature engineering)
      val featureColumns = Array("Pclass", "Age", "FamilySize", "SexIndex", "Fare", "EmbarkedIndex")

      // Assemble the feature columns into a single vector
      val assembler = new VectorAssembler()
        .setInputCols(featureColumns)
        .setOutputCol("features")

      // Create a Random Forest Classifier
      val randomForest = new RandomForestClassifier()
        .setLabelCol("Survived")
        .setFeaturesCol("features")
        .setNumTrees(1)

      // Create a pipeline
      val pipeline = new Pipeline().setStages(Array(assembler, randomForest))

      // Split the data into training and validation sets
      val Array(trainingData, validationData) = selectedFeatures.randomSplit(Array(0.3, 0.7))

      // Train the model on the training data
      val model = pipeline.fit(trainingData)

      // Make predictions on the validation data
      val predictions = model.transform(validationData)

      // Evaluate the model's performance using a BinaryClassificationEvaluator
      val evaluator = new BinaryClassificationEvaluator()
        .setLabelCol("Survived")
        .setRawPredictionCol("rawPrediction")
        .setMetricName("areaUnderROC")

      val accuracy = evaluator.evaluate(predictions) * 100
      println(s"Accuracy: $accuracy%")

      // Make predictions on the test data
      val testFeatures = testData
        .withColumn("Pclass", testData("Pclass").cast("int"))
        .withColumn("Age", testData("Age").cast("int"))
        .withColumn("Fare", testData("Fare").cast("int"))
        .na.fill(0.0, Seq("Age", "Fare"))
        .withColumn("FamilySize", col("SibSp") + col("Parch"))
        .withColumn("Title", regexp_extract(col("Name"), "(Mrs\\.|Mr\\.|Miss\\.|Master\\.|Rev\\.|Dr\\.|Major\\.|Capt\\.|Col\\.|Lady\\.|Sir\\.|Jonkheer\\.|Don\\.)", 0))
        .transform { df =>
          val sexIndexer = new StringIndexer()
            .setInputCol("Sex")
            .setOutputCol("SexIndex")
            .setHandleInvalid("skip")
            .fit(df)

          val embarkedIndexer = new StringIndexer()
            .setInputCol("Embarked")
            .setOutputCol("EmbarkedIndex")
            .setHandleInvalid("skip")
            .fit(df)

          val titleIndexer = new StringIndexer()
            .setInputCol("Title")
            .setOutputCol("TitleIndex")
            .setHandleInvalid("skip")
            .fit(df)

          titleIndexer.transform(embarkedIndexer.transform(sexIndexer.transform(df)))
        }
        .select("PassengerId", "Pclass", "Age", "FamilySize", "TitleIndex", "SexIndex", "Fare", "EmbarkedIndex")

      val testPredictions = model.transform(testFeatures)

      // Save or display the test predictions as needed
      testPredictions.show()


      // Stop the Spark session
      sc.stop()
    }

}
