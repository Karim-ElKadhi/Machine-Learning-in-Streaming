from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, IntegerType
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression

spark = SparkSession.builder \
    .appName("RealTimeMLInference") \
    .getOrCreate()

# Define the schema matching your CSV
schema = StructType([
    StructField("transaction_id", StringType(), True),
    StructField("user_id", StringType(), True),
    StructField("amount", StringType(), True),
    StructField("timestamp", StringType(), True)
])

# Read from Kafka
df = spark.readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "localhost:9092") \
    .option("subscribe", "topic_csv") \
    .option("startingOffsets", "latest") \
    .load()

# Deserialize the JSON value
json_df = df.selectExpr("CAST(value AS STRING) as json")
events_df = json_df.select(from_json(col("json"), schema).alias("data")).select("data.*")

from pyspark.sql.functions import col

# Convertir la colonne amount en DoubleType
events_df = events_df.withColumn("amount", col("amount").cast("double"))

# Préparer les features
assembler = VectorAssembler(inputCols=["amount"], outputCol="features")
features_df = assembler.transform(events_df)

#  Linear Regression
lr = LinearRegression(featuresCol="features", labelCol="amount")

def train_model(batch_df, batch_id):
    if batch_df.count() > 0:
        model = lr.fit(batch_df)
        predictions = model.transform(batch_df)
        predictions.show() 

query = features_df.writeStream.foreachBatch(train_model).start()
query.awaitTermination()
