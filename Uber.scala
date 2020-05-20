package TheDefinitiveGuide

import org.apache.spark.ml.clustering.KMeans
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types._

object Uber {
  def main(args:Array[String]):Unit={
    val spark:SparkSession = SparkSession.builder().master("local[*]").appName("App").getOrCreate()
    val schema = StructType(Array(
      StructField("dt",TimestampType,true),
      StructField("lat",DoubleType,true),
      StructField("lon",DoubleType,true),
      StructField("base",StringType,true)
    ))
    val df = spark.read.option("header","false").schema(schema).csv("C:/Users/vishal rana/Desktop/spark-ml-kmeans-uber-master/data/uber.csv")
   // df.printSchema()
    df.cache()
   // df.show()
    val featureCols = Array("lat","lon")
    val assembler = new VectorAssembler().setInputCols(featureCols).setOutputCol("features")
    val df2 = assembler.transform(df)
   // df2.show()
    val Array(train,test) = df2.randomSplit(Array(0.7,0.3),5043)
    val kmeans = new KMeans().setK(8).setFeaturesCol("features").setPredictionCol("Prediction")
    val model = kmeans.fit(train)
    println("Final Centers:")
    model.clusterCenters.foreach(println)
    val categories = model.transform(test)
    //categories.show()
    categories.groupBy("Prediction").count().show()
    model.write.overwrite().save("C:/Users/vishal rana/Desktop/spark-ml-kmeans-uber-master/data/savemodel")
  }
}
