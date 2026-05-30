import polars as pl
import elfen
from scipy import stats

df=pl.read_csv("Random_English_Sentences.csv")

'''print(df.head())
print(df.shape)'''

extractor = elfen.Extractor(
    data=df,
    language="en",
    text_column="text",
)
print("Extractor created successfully.")
'''''extractor.extract("ttr")
print(extractor.data.head())'''''

extractor.extract_feature_group("readability")
''''print(
    extractor.data.select([
        "text",
        "flesch_reading_ease",
        "flesch_kincaid_grade",
        "smog",
        "gunning_fog"
    ]).head()
)
print(extractor.data.head())
print(extractor.data.columns)'''''''''

'''''print(
    extractor.data.select([
        "text",
        "flesch_reading_ease"
    ]).sort("flesch_reading_ease").head(10)
)'''''''''

''''extractor.data["flesch_reading_ease"].describe()
print(stats.describe(extractor.data["flesch_reading_ease"]))'''''
'''''print(
    extractor.data.select([
        "text",
        "n_tokens",
        "n_sentences",
        "n_syllables",
        "flesch_reading_ease"
    ]).head(10)
)'''''''''

'''''print(
    extractor.data.select([
        "flesch_kincaid_grade",
        "gunning_fog",
        "smog"
    ]).head()
)'''''''''

extractor.extract_features()

print(len(extractor.data.columns))





               