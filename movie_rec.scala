import algebra.instances.all.catsKernelStdOrderForChar
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.recommendation.ALS
import org.apache.spark.sql.{Row, SparkSession}
import org.apache.spark.sql.functions.{array_position, col, struct}
import org.apache.spark.sql.types.{FloatType, IntegerType, LongType, StructField, StructType}
import org.apache.spark.sql.catalyst.plans.Inner
import org.apache.spark.SparkContext._
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.ml.feature.IndexToString
import spire.implicits._
import spire.math.Polynomial.x
import sun.misc.MessageUtils.where


object movie_rec {
  def main(args: Array[String]): Unit = {

    val spark = SparkSession
      .builder()
      .master("local[*]")
      .getOrCreate()

    spark.sparkContext.setLogLevel("WARN")

    val dataspath = "D:\\data_files\\movielens_ratings.txt"
    val dataspath_movies = "D:\\data_files\\movies.csv"


    val textDF =spark.read.option("header","false")
      .option ("inferSchema",false)
      .text(dataspath)

    val textDF_new =spark.read.option("header","true")
      .option ("inferSchema",true)
      .csv(dataspath_movies)
    textDF_new.createOrReplaceTempView("new_df")

    textDF.printSchema()
    textDF.show()

    textDF_new.printSchema()
    textDF_new.show()

/*
+-------+--------------------+--------------------+
|movieId|               title|              genres|
+-------+--------------------+--------------------+
|      1|    Toy Story (1995)|Adventure|Animati...|
|      2|      Jumanji (1995)|Adventure|Childre...|
|      3|Grumpier Old Men ...|      Comedy|Romance|
|      4|Waiting to Exhale...|Comedy|Drama|Romance|
|      5|Father of the Bri...|              Comedy|
 */

    import  spark.implicits._
    val textDF_split = textDF.map( f => {val elements = f.getString(0).split("::")
      (elements(0),elements(1),elements(2),elements(3))})

    textDF_split.printSchema()

    /*
    root
     |-- _1: string (nullable = true)
     |-- _2: string (nullable = true)
     |-- _3: string (nullable = true)
     |-- _4: string (nullable = true)

     */

    textDF_split.show(5,false)

    /*
    +---+---+---+----------+
    |_1 |_2 |_3 |_4        |
    +---+---+---+----------+
    |0  |2  |3  |1424380312|
    |0  |3  |1  |1424380312|
    |0  |5  |2  |1424380312|
    |0  |9  |4  |1424380312|
    |0  |11 |1  |1424380312|
    +---+---+---+----------+
     */

    val text_renamed  = textDF_split

      .withColumnRenamed("_1","userid")
      .withColumnRenamed("_2","movieid")
      .withColumnRenamed("_3","rating")
      .withColumnRenamed("_4","timestamp")

    text_renamed.printSchema()
    text_renamed.show(5,false)
    text_renamed.createOrReplaceTempView("mainDF")


    val text_datatype_chg =  text_renamed
      .withColumn("userid", col ( "userid").cast(IntegerType))
      .withColumn("movieId", col ( "movieId").cast(IntegerType))
      .withColumn("rating", col ( "rating").cast(IntegerType))
      .withColumn("timestamp", col ( "timestamp").cast(LongType))

    text_datatype_chg.printSchema()
    text_datatype_chg.describe().show()

    println("joined df")
    val joined_df = text_datatype_chg
      .join(textDF_new,text_datatype_chg("movieid") === textDF_new("movieId"),"inner")

    joined_df.show()
    joined_df.createOrReplaceTempView("for_pre_df")

/*
+------+-------+------+----------+-------+----------------+--------------------+
|userid|movieId|rating| timestamp|movieId|           title|              genres|
+------+-------+------+----------+-------+----------------+--------------------+
|    28|      1|     1|1424380312|      1|Toy Story (1995)|Adventure|Animati...|
|    26|      1|     1|1424380312|      1|Toy Story (1995)|Adventure|Animati...|
|    25|      1|     3|1424380312|      1|Toy Story (1995)|Adventure|Animati...|
|    20|      1|     1|1424380312|      1|Toy Story (1995)|Adventure|Animati...|
|    19|      1|     1|1424380312|      1|Toy Story (1995)|Adventure|Animati...|
|    18|      1|     1|1424380312|      1|Toy Story (1995)|Adventure|Animati...|
|    15|      1|     4|1424380312|      1|Toy Story (1995)|Adventure|Animati...|
|    14|      1|     1|1424380312|      1|Toy Story (1995)|Adventure|Animati...|
|     7|      1|     1|1424380312|      1|Toy Story (1995)|Adventure|Animati...|
 */
    println("df for prediction")
    val df_for_pre = spark.sql("select userid,title,rating from for_pre_df order by userid")
     df_for_pre.show()

/*
+------+--------------------+------+
|userid|               title|rating|
+------+--------------------+------+
|     0|    Assassins (1995)|     1|
|     0|       Friday (1995)|     1|
|     0|      Othello (1995)|     3|
|     0|Dangerous Minds (...|     1|
|     0|French Twist (Gaz...|     1|
|     0|         Babe (1995)|     1|
|     0|Father of the Bri...|     2|
|     0|  Richard III (1995)|     2|
|     0| Sudden Death (1995)|     4|
 */


    val indexer = new StringIndexer()
      .setInputCol("title")
      .setOutputCol("label")

    val for_pre_label = indexer.setHandleInvalid("keep")
      .fit( df_for_pre)
      .transform( df_for_pre)
    for_pre_label.printSchema()
    for_pre_label.show()
    for_pre_label.createOrReplaceTempView("labeled_view")
/*
+------+--------------------+------+-----+
|userid|               title|rating|label|
+------+--------------------+------+-----+
|     0|      Jumanji (1995)|     3|  8.0|
|     0|Grumpier Old Men ...|     1| 64.0|
|     0|Father of the Bri...|     2| 62.0|
|     0| Sudden Death (1995)|     4| 35.0|
|     0|American Presiden...|     1| 70.0|
|     0|Dracula: Dead and...|     2| 20.0|
|     0|Cutthroat Island ...|     1|  5.0|
|     0|Sense and Sensibi...|     1| 68.0|
|     0|Ace Ventura: When...|     1| 15.0|
|     0|   Get Shorty (1995)|     1| 21.0|
|     0|    Assassins (1995)|     1| 38.0|
 */

    println("labeled_view")
    val labeled_title = spark.sql("select distinct title,label from labeled_view ")
    labeled_title.show(10,false)
    labeled_title.createOrReplaceTempView("labeled_title")

/*
+----------------------------------+-----+
|title                             |label|
+----------------------------------+-----+
|Heat (1995)                       |2.0  |
|Persuasion (1995)                 |75.0 |
|Balto (1995)                      |24.0 |
|Two if by Sea (1996)              |51.0 |
|Usual Suspects, The (1995)        |3.0  |
|Eye for an Eye (1996)             |41.0 |
 */


    val Array(trainData,testData)=for_pre_label.randomSplit(Array(0.70,0.30))

//build recommended model using ALS on the training data

  val als = new ALS()
    .setMaxIter(5)
    .setRegParam(0.01)
    .setUserCol("userid")
    .setItemCol("label")
    .setRatingCol("rating")
    

//Create model
    val model = als.fit(trainData)

//Evaluate the model by computing the RMSE on the test data

//Note: we set cold start strategy to 'drop' to ensure we don't get NaN evaluvation metrics

    model.setColdStartStrategy("drop")

    val predictions = model.transform(testData)

    println("prediction show before ")
    predictions.show(5)



    val evaluator = new RegressionEvaluator()
      .setMetricName("rmse")
      .setLabelCol("rating")
      .setPredictionCol("prediction")

    val rmse = evaluator.evaluate(predictions)
    println(s"root mean square error = $rmse")
/*
+------+-----------------------------------------------------------------------------------+
|userid|recommendations                                                                    |
+------+-----------------------------------------------------------------------------------+
|20    |[{1, 4.7586555}, {72, 4.075051}, {47, 3.8743427}, {4, 3.7565405}, {32, 3.502046}]  |
|10    |[{74, 5.848626}, {48, 4.3290176}, {85, 4.292591}, {76, 4.065938}, {64, 3.9447196}] |
|0     |[{32, 4.2374535}, {8, 2.8006315}, {58, 2.699373}, {1, 2.6562421}, {81, 2.6498687}] |
|1     |[{47, 4.703966}, {32, 4.1265044}, {6, 4.033287}, {72, 3.8867211}, {61, 3.8714652}] |
|21    |[{74, 5.03284}, {8, 3.3334682}, {50, 3.1279092}, {49, 3.029807}, {37, 2.8379664}]  |
|11    |[{45, 8.117554}, {63, 5.144524}, {42, 5.0784497}, {48, 5.0101185}, {29, 4.947463}] |
|12    |[{48, 5.019181}, {68, 4.883958}, {51, 4.772387}, {83, 4.391696}, {76, 4.3426356}]  |
|22    |[{71, 4.960453}, {72, 4.9053664}, {61, 4.782325}, {38, 4.533542}, {56, 4.2273355}] |
|2     |[{87, 5.0282216}, {85, 4.964202}, {25, 4.7969513}, {32, 4.5606003}, {80, 4.140167}]|
|13    |[{45, 4.6021504}, {85, 4.032349}, {59, 3.536076}, {61, 3.5202436}, {38, 3.431693}] |
+------+-----------------------------------------------------------------------------------+
 */

    println("prediction")
    val userrec = model.recommendForAllUsers(5)
    userrec.printSchema()
    userrec.show(10,false)
/*
    val indexer_2 = new StringIndexer()
      .setInputCol("recommendations")
      .setOutputCol("title")

    val final_out = indexer_2.setHandleInvalid("keep")
      .fit( userrec)
      .transform( userrec)
    final_out.printSchema()
    final_out.show()
    */
    //userrec.select($"recommendations"(1)).show

    //val arr_pos_df = userrec.withColumn("recommendations", array_position($"recommendations", 0))
    //arr_pos_df.show()

   val struc_label  = userrec
    .select(col("recommendations.label"),col("userid"))
    struc_label.show(5,false)
    struc_label.printSchema()
    struc_label.createOrReplaceTempView("userid_view")

/*
+-------------------+------+
|label              |userid|
+-------------------+------+
|[72, 0, 4, 38, 68] |20    |
|[76, 42, 85, 4, 71]|10    |
|[60, 7, 37, 55, 35]|0     |
|[60, 56, 32, 6, 72]|1     |
|[0, 74, 71, 8, 56] |21    |
+-------------------+------+

root
 |-- label: array (nullable = true)
 |    |-- element: integer (containsNull = true)
 |-- userid: integer (nullable = false)

 */
    println("userid_view")
    val userid_df= spark.sql("select distinct userid from userid_view")
    userid_df.show()

/*
+------+
|userid|
+------+
|    20|
|    10|
|     0|
|     1|
|    21|
|    11|
|    12|
|    22|
|     2|
|    13|
|     3|
|    23|
 */

    val struc_label_new = struc_label
      .select($"userid",$"label"(0),$"label"(1),$"label"(2),$"label"(3),$"label"(4))
    struc_label_new.show(5,false)

    struc_label_new.createOrReplaceTempView("title_labels")

/*
+------+--------+--------+--------+--------+--------+
|userid|label[0]|label[1]|label[2]|label[3]|label[4]|
+------+--------+--------+--------+--------+--------+
|20    |1       |72      |47      |4       |32      |
|10    |74      |48      |85      |76      |64      |
|0     |32      |8       |58      |1       |81      |
|1     |47      |32      |6       |72      |61      |
|21    |74      |8       |50      |49      |37      |
+------+--------+--------+--------+--------+--------+
 */

/*
    val converter = new IndexToString()
      .setInputCol("label[0]")
      .setOutputCol("title")
    val converted = converter.transform(struc_label_new)

    converted.show(5,false)
*/

    //struc_label_new.withColumn("label[0]", col("label[0]").cast("string") ).write.csv("D:\\checking_log")
    println("title_view")
    val title_view =struc_label_new

      .withColumnRenamed("label[0]","labels")
      .withColumnRenamed("label[1]","labels_2")
      .withColumnRenamed("label[2]","labels_3")
      .withColumnRenamed("label[3]","labels_4")
      .withColumnRenamed("label[4]","labels_5")
   title_view.show()
    title_view.createOrReplaceTempView("ti_fi_view")

/*
+------+------+--------+--------+--------+--------+
|userid|labels|labels_2|labels_3|labels_4|labels_5|
+------+------+--------+--------+--------+--------+
|    20|    77|      34|      72|      47|      38|
|    10|     9|       0|      48|      63|      76|
|     0|    42|      25|      35|      57|      21|
|     1|    77|      72|      60|      27|      51|
|    21|    74|       0|      61|      63|      76|
|    11|    37|      33|      63|      38|      29|

 */
/*
    val converter = new IndexToString()
      .setInputCol("labels")
      .setOutputCol("title")
    val converted = converter.transform(title_view)

    converted.show(5,false)

 */
/*
    val joined_final = title_view
      .join(labeled_title,title_view("labels") === labeled_title("label"),"inner")
    joined_final.show()
*/
    println("joined final")
    val joined_final = spark.sql("select ti_fi_view.userid,labeled_title.title FROM ti_fi_view JOIN labeled_title ON ti_fi_view.labels = labeled_title.label," +
                                                                  "labeled_title.title FROM ti_fi_view JOIN labeled_title ON ti_fi_view.labels_2 = labeled_title.label" +
                                                                  "labeled_title.title FROM ti_fi_view JOIN labeled_title ON ti_fi_view.labels_3 = labeled_title.label" +
                                                                  "labeled_title.title FROM ti_fi_view JOIN labeled_title ON ti_fi_view.labels_4 = labeled_title.label" +
                                                                  "labeled_title.title FROM ti_fi_view JOIN labeled_title ON ti_fi_view.labels_5 = labeled_title.label")


    joined_final.show(5,false)
/*
    val final_new = joined_final
      .select(col("userid"),col("title"))
    final_new.show(5,false)
    final_new.createOrReplaceTempView("final_out")

    spark.sql("select userid,labels,label[1],label[2],label[3],label[4] where userid = 20").show()
*/


  }

}
