from pyspark.sql import SparkSession
from pyspark.sql.functions import split
from pyspark.sql.types import StructType, StructField, IntegerType, StringType
from pyspark import SparkContext
from pyspark.sql.types import *

spark = SparkSession \
    .builder \
    .master("local[*]") \
    .appName("PySpark ETL") \
    .getOrCreate()

sc = spark.sparkContext
sc

######################### EXTRACTION #################################################################

def extract():
    spark = SparkSession.builder \
        .appName("PassagesLogsExtraction") \
        .getOrCreate()
    
    passages = spark.read.text("passages.logs")

    df = passages.withColumn("data", split(passages["value"], " / ")) \
                 .selectExpr("data[0] as identifiant",
                             "split(data[1], ' ')[0] as date",
                             "split(data[1], ' ')[1] as heure",
                             "data[2] as immatriculation",
                             "data[3] as vitesse")

    df = df.dropDuplicates()
    df = df.dropna()

    amendes = spark.read.csv("amendes.csv", header=True)

    schema_amendes = StructType([
        StructField("_0", IntegerType(), nullable=True),
        StructField("borne_basse", IntegerType(), nullable=True),
        StructField("borne_haute", IntegerType(), nullable=True),
        StructField("amende", IntegerType(), nullable=True),
        StructField("points", IntegerType(), nullable=True)
    ])

    amendes_with_schema = spark.read \
        .format("csv") \
        .option("header", True) \
        .schema(schema_amendes) \
        .load("amendes.csv")

    schema_voitures = StructType([
        StructField("immatriculation", StringType()),
        StructField("name", StringType()),
        StructField("location", StructType([
            StructField("address", StringType()),
            StructField("city", StringType()),
            StructField("state", StringType()),
            StructField("zipcode", StringType()),
        ])),
        StructField("phone", StringType()),
        StructField("points", IntegerType()),
    ])

    voitures_df = spark.read \
        .json("voitures.json", schema=schema_voitures, multiLine=True) \
        .select(
            "immatriculation",
            "name",
            "location.address",
            "location.city",
            "location.state",
            "location.zipcode",
            "phone",
            "points"
        )

    return df, amendes_with_schema, voitures_df

passages_logs_df, amendes_df, voitures_df = extract()

# Supprimer les doublons et les valeurs manquantes dans chaque DataFrame
passages_logs_df = passages_logs_df.dropDuplicates().dropna()
amendes_df = amendes_df.dropDuplicates().dropna()
voitures_df = voitures_df.dropDuplicates().dropna()

print("passages_logs :")
passages_logs_df.show()
print("amendes_df :")
amendes_df.show()
print("voitures_df :")
voitures_df.show()

######################### TRANSFORMATION #################################################################

from pyspark.sql.functions import expr, avg as spark_avg

#Vérification avec un exemple par rapport au montant des amendes et le nombre de points retiré
passages_logs_df.filter(passages_logs_df["immatriculation"] == "XC630RP").show()
amendes_df.show()

def transform(passages_logs_df, amendes_df, voitures_df):
    voitures_df = voitures_df.withColumnRenamed("points", "voitures_points")

    joined_df = passages_logs_df.join(voitures_df, "immatriculation", "inner")

    # Jointure entre joined_df et amendes_df
    joined_df = joined_df.join(
        amendes_df,
        (joined_df["vitesse"] > amendes_df["borne_basse"]) & (joined_df["vitesse"] <= amendes_df["borne_haute"]),
        "left"
    )

    to_pay_df = joined_df.groupBy("immatriculation", "voitures_points").agg(
        spark_sum(expr("CASE WHEN amende IS NULL THEN 0 ELSE amende END")).alias("to_pay"),
        spark_sum(expr("CASE WHEN points IS NULL THEN 0 ELSE points END")).alias("points_perdus")
    )

    avg_speed_df = passages_logs_df.groupBy("immatriculation").agg(
        spark_avg("vitesse").alias("vitesse_moyenne") #rajout perso pour vérification
    )

    final_df = to_pay_df.join(avg_speed_df, "immatriculation", "left")

    final_df = final_df.withColumn("retrait_permis", final_df["voitures_points"] <= final_df["points_perdus"])

    return final_df

final_df = transform(passages_logs_df, amendes_df, voitures_df)

final_df.show()
