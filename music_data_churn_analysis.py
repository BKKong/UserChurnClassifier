import collections
import functools
import io
import os
import logging
import numpy as np
import pandas as pd
import tarfile
import requests
from pyspark.sql.types import *
from datetime import datetime, date, time, timedelta

PlayEntry = collections.namedtuple('PlayEntry', 
                                  ('date', 'uid', 'device', 'song_id', 'song_type',
                                   'song_name', 'singer', 'play_time', 'song_length',
                                   'paid_flag'))

DownEntry = collections.namedtuple('DownEntry',
                                   ('date', 'uid', 'device', 'song_id', 'song_name',
                                    'singer', 'some_flag'))

PlayFeatures = collections.namedtuple('PlayFeatures', 
                                  ('play_freq', 'avg_play_percent', 'num_songs_played',
                                   'num_singers_played', 'last_play', 'play_sum'))

DownFeatures = collections.namedtuple('DownFeatures', 
                                      ('down_freq', 'num_singers_downed', 'last_down'))

DfPlayFeatures = collections.namedtuple('DfPlayFeatures', 
                                    ('play_freq_1st', 'play_perc_1st', 'play_songs_1st',
                                     'play_singers_1st', 'play_sum_1st', 'play_freq_2nd',
                                     'play_perc_2nd', 'play_songs_2nd', 'play_singers_2nd',
                                     'play_sum_2nd', 'play_freq_3rd', 'play_perc_3rd', 
                                     'play_songs_3rd', 'play_singers_3rd', 'play_sum_3rd',
                                     'days_from_lastplay', 'play_label'))
DfDownFeatures = collections.namedtuple('DfDownFeatures', 
                                    ('down_freq_1st', 'down_singers_1st', 'down_freq_2nd', 
                                     'down_singers_2nd', 'down_freq_3rd', 'down_singers_3rd', 
                                     'days_from_lastdown','down_label'))


def FilterByDate(begin_date, end_date, url):
    date = os.path.basename(url)[:8]
    if begin_date <= date <= end_date:
        yield url


def DownloadLog(url):
    logging.info('Downloading %s', url)
    date = os.path.basename(url)[:8]
    try:
        content = requests.get(url).content
    except IOError as e:
        logging.exception(str(e))
        return
    yield date, content


def ExtractLines(date_and_content):
    date, content = date_and_content
    try:
        with tarfile.open(mode='r:gz', fileobj=io.BytesIO(content)) as archive:
            filename = archive.getnames()[0]
            for line in archive.extractfile(filename):
                try:
                    yield date, line.decode('utf-8')
                except UnicodeDecodeError as e:
                    logging.exception(str(e))
    except (IOError, IndexError, EOFError) as e:
        logging.exception(str(e))


def ParsePlayLogLine(date_and_content):
    date, content = date_and_content
    fields = content.split('\t')
    if len(fields) < 9:
        logging.error('%r contains only %d fields', content, len(fields))
        return 
    try:
        uid = int(fields[0].strip())
        device = fields[1].strip()
        song_id = int(fields[2].strip())
        song_type = int(fields[3].strip())
        song_name = ' '.join(fields[4:-4]).strip()
        singer = fields[-4].strip()
        play_time = int(fields[-3].strip())
        song_length = int(fields[-2].strip())
        paid_flag = int(fields[-1].strip())
    except ValueError as e:
        logging.exception(str(e))
        return
    if song_length <= 0 or play_time < 0:
        return
    yield PlayEntry(date=date, uid=uid, device=device, song_id=song_id, 
                   song_type=song_type, song_name=song_name, singer=singer,
                   play_time=play_time, song_length=song_length, paid_flag=paid_flag)   

    
def ParseDownLogLine(date_and_content):
    date, content = date_and_content
    fields = content.split('\t')
    if len(fields) < 6:
        logging.error('%r contains only %d fields', content, len(fields))
        return 
    try:
        uid = int(fields[0].strip())
        device = fields[1].strip()
        song_id = int(fields[2].strip())
        song_name = ' '.join(fields[3:-2]).strip()
        singer = fields[-2].strip()
        some_flag = int(fields[-1].strip())
    except ValueError as e:
        logging.exception(str(e))
        return
    
    yield DownEntry(date=date, uid=uid, device=device, song_id=song_id, 
                   song_name=song_name, singer=singer, some_flag=some_flag)   


def ExtractPlayFeatures(uid_and_log_entries):
    uid, log_entries = uid_and_log_entries
    play_percs = []
    song_lengths = []
    songs_played = set()
    singers_played = set()
    last_play = '20170301'
    play_sum = 0.0
    for log_entry in log_entries:
        play_percs.append(min(float(log_entry.play_time)/float(log_entry.song_length), 1.0))
        song_lengths.append(log_entry.song_length)
        songs_played.add(log_entry.song_id)
        singers_played.add(log_entry.singer)
        last_play = max(last_play, log_entry.date)
        play_sum = play_sum + float(log_entry.play_time)
        
    play_freq = len(log_entries)    
    avg_play_percent = float(np.mean(play_percs))
    num_songs_played = len(songs_played)
    num_singers_played = len(singers_played)
    return uid, PlayFeatures(play_freq=play_freq, avg_play_percent=avg_play_percent, 
                             num_songs_played=num_songs_played, num_singers_played=num_singers_played,
                             last_play=last_play, play_sum=play_sum)

def ExtractDownFeatures(uid_and_log_entries):
    uid, down_entries = uid_and_log_entries
    singers_downed = set()
    last_down = '20170301'
    for down_entry in down_entries:
        singers_downed.add(down_entry.singer)
        last_down = max(last_down, down_entry.date)
        
    down_freq = len(down_entries)    
    num_singers_downed = len(singers_downed)
    return uid, DownFeatures(down_freq=down_freq, num_singers_downed=num_singers_downed,
                             last_down=last_down)

def PlayPipeline(url_list, begin_date, end_date):
    return (sc.parallelize(url_list)
              .filter(lambda url: '_play' in url)
              .flatMap(lambda url: FilterByDate(begin_date, end_date, url))
              .flatMap(DownloadLog)
              .flatMap(ExtractLines)
              .flatMap(ParsePlayLogLine)
              .map(lambda play_entry: (play_entry.uid, play_entry))
              .groupByKey()
              .mapValues(list)
              .map(ExtractPlayFeatures)
           )


def DownPipeline(url_list, begin_date, end_date):
    return (sc.parallelize(url_list)
              .filter(lambda url: '_down' in url)
              .flatMap(lambda url: FilterByDate(begin_date, end_date, url))
              .flatMap(DownloadLog)
              .flatMap(ExtractLines)
              .flatMap(ParseDownLogLine)
              .map(lambda down_entry: (down_entry.uid, down_entry))
              .groupByKey()
              .mapValues(list)
              .map(ExtractDownFeatures)
           )


def FlattenJoinResults(join_results):
    if type(join_results) is tuple:
        for elem in join_results:
            yield from FlattenJoinResults(elem)
    else:
        yield join_results


def CombinePlayFeatures(four_features):
    features1, features2, features3, features4= four_features    
    none_to_zero = lambda x: float(x) if not np.isnan(x) else 0.0
    if features3:
        recent_play = features3.last_play
    elif features2:
        recent_play = features2.last_play
    elif features1:
        recent_play = features1.last_play
    else:
        recent_play = '20170301'
    features = DfPlayFeatures(
                          play_freq_1st=none_to_zero(features1.play_freq) if features1 else 0.0, 
                          play_perc_1st=none_to_zero(features1.avg_play_percent) if features1 else 0.0, 
                          play_songs_1st=none_to_zero(features1.num_songs_played) if features1 else 0.0,
                          play_singers_1st=none_to_zero(features1.num_singers_played) if features1 else 0.0, 
                          play_sum_1st=none_to_zero(features1.play_sum) if features1 else 0.0,
        
                          play_freq_2nd=none_to_zero(features2.play_freq) if features2 else 0.0, 
                          play_perc_2nd=none_to_zero(features2.avg_play_percent) if features2 else 0.0,
                          play_songs_2nd=none_to_zero(features2.num_songs_played) if features2 else 0.0, 
                          play_singers_2nd=none_to_zero(features2.num_singers_played) if features2 else 0.0,
                          play_sum_2nd=none_to_zero(features2.play_sum) if features2 else 0.0,
        
                          play_freq_3rd=none_to_zero(features3.play_freq) if features3 else 0.0, 
                          play_perc_3rd=none_to_zero(features3.avg_play_percent) if features3 else 0.0,
                          play_songs_3rd=none_to_zero(features3.num_songs_played) if features3 else 0.0, 
                          play_singers_3rd=none_to_zero(features3.num_singers_played) if features3 else 0.0,
                          play_sum_3rd=none_to_zero(features3.play_sum) if features3 else 0.0,
        
                          days_from_lastplay=(StringtoDatetime('20170501')-StringtoDatetime(recent_play))/timedelta(days=1),
                          play_label=features4 is None
                     )
    assert set(map(type, features[:-1])) == {float}
    return features

def CombineDownFeatures(four_features):
    features1, features2, features3, features4= four_features 
    none_to_zero = lambda x: float(x) if not np.isnan(x) else 0.0
    if features3:
        recent_down = features3.last_down
    elif features2:
        recent_down = features2.last_down
    elif features1:
        recent_down = features1.last_down
    else:
        recent_down = '20170301'
    return DfDownFeatures(down_freq_1st=none_to_zero(features1.down_freq) if features1 else 0.0,          
                          down_singers_1st=none_to_zero(features1.num_singers_downed) if features1 else 0.0,    
                          down_freq_2nd=none_to_zero(features2.down_freq) if features2 else 0.0,             
                          down_singers_2nd=none_to_zero(features2.num_singers_downed) if features2 else 0.0,
                          down_freq_3rd=none_to_zero(features3.down_freq) if features3 else 0.0, 
                          down_singers_3rd=none_to_zero(features3.num_singers_downed) if features3 else 0.0,
                          days_from_lastdown=(StringtoDatetime('20170501')-StringtoDatetime(recent_down))/timedelta(days=1),
                          down_label=features4 is None
                     )

url_list = []
with open('url_list.txt', 'r') as url_input:
    for line in url_input:
        url_list.append(line.strip())

play_wk_1 = PlayPipeline(url_list, '20170329', '20170408')
play_wk_2 = PlayPipeline(url_list, '20170409', '20170419')
play_wk_3 = PlayPipeline(url_list, '20170420', '20170430')
play_wk_4 = PlayPipeline(url_list, '20170501', '20170512')

rdd_play_features = (play_wk_1.fullOuterJoin(play_wk_2)
                    .fullOuterJoin(play_wk_3)
                    .mapValues(lambda join_results:
                        join_results if join_results[0] else ((None, None), join_results[1]))
                    .leftOuterJoin(play_wk_4)
                    .mapValues(lambda join_results: tuple(FlattenJoinResults(join_results)))
                    .mapValues(CombinePlayFeatures)
                    .map(lambda uid_and_features: (uid_and_features[0],) + uid_and_features[1]))

schema_play = StructType([StructField("uid", IntegerType(), True), 
                     StructField("play_freq_1st", FloatType(), True),
                     StructField("play_perc_1st", FloatType(), True),
                     StructField("play_songs_1st", FloatType(), True),
                     StructField("play_singers_1st", FloatType(), True),
                     StructField("play_sum_1st", FloatType(), True),     
                          
                     StructField("play_freq_2nd", FloatType(), True),
                     StructField("play_perc_2nd", FloatType(), True),
                     StructField("play_songs_2nd", FloatType(), True),
                     StructField("play_singers_2nd", FloatType(), True),
                     StructField("play_sum_2nd", FloatType(), True),     
                          
                     StructField("play_freq_3rd", FloatType(), True),
                     StructField("play_perc_3rd", FloatType(), True),
                     StructField("play_songs_3rd", FloatType(), True),
                     StructField("play_singers_3rd", FloatType(), True),
                     StructField("play_sum_3rd", FloatType(), True),     

                     StructField("days_from_lastplay", FloatType(), True),
                     StructField("play_label", BooleanType(), True)])


df_play_features = spark.createDataFrame(rdd_play_features, schema_play)
df_play_features.toPandas()

down_wk_1 = DownPipeline(url_list, '20170329', '20170408')
down_wk_2 = DownPipeline(url_list, '20170409', '20170419')
down_wk_3 = DownPipeline(url_list, '20170420', '20170430')
down_wk_4 = DownPipeline(url_list, '20170501', '20170512')

rdd_down_features = (down_wk_1.fullOuterJoin(down_wk_2)
                    .fullOuterJoin(down_wk_3)
                    .mapValues(lambda join_results:
                        join_results if join_results[0] else ((None, None), join_results[1]))
                    .leftOuterJoin(down_wk_4)
                    .mapValues(lambda join_results: tuple(FlattenJoinResults(join_results)))
                    .mapValues(CombineDownFeatures)
                    .map(lambda uid_and_features: (uid_and_features[0],) + uid_and_features[1]))

schema_down = StructType([StructField("uid", IntegerType(), True), 
                     StructField("down_freq_1st", FloatType(), True),
                     StructField("down_singers_1st", FloatType(), True),
                     StructField("down_freq_2nd", FloatType(), True),
                     StructField("down_singers_2nd", FloatType(), True),
                     StructField("down_freq_3rd", FloatType(), True),
                     StructField("down_singers_3rd", FloatType(), True),
                     StructField("days_from_lastdown", FloatType(), True),
                     StructField("down_label", BooleanType(), True)])

df_down_features = spark.createDataFrame(rdd_down_features, schema_down)
df_down_features.toPandas()


## Joining the play features dataframe and download features dataframe

df_play_features.registerTempTable('play_df')
df_down_features.registerTempTable('down_df')

query = """
    SELECT
        A.uid AS uid,
        A.play_freq_1st,
        A.play_perc_1st,
        A.play_songs_1st,
        A.play_singers_1st,
        A.play_sum_1st,
        A.play_freq_2nd,
        A.play_perc_2nd,
        A.play_songs_2nd,
        A.play_singers_2nd,
        A.play_sum_2nd,
        A.play_freq_3rd,
        A.play_perc_3rd,
        A.play_songs_3rd,
        A.play_singers_3rd,
        A.play_sum_3rd,
        A.days_from_lastplay,
        COALESCE(B.down_freq_1st, 0)    AS down_freq_1st,
        COALESCE(B.down_singers_1st, 0) AS down_singers_1st,
        COALESCE(B.down_freq_2nd, 0)    AS down_freq_2nd,
        COALESCE(B.down_singers_2nd, 0) AS down_singers_2nd,
        COALESCE(B.down_freq_3rd, 0)    AS down_freq_3rd,
        COALESCE(B.down_singers_3rd, 0) AS down_singers_3rd,
        COALESCE(B.days_from_lastdown, 60) AS days_from_lastdown,
        A.play_label AND COALESCE(B.down_label, True) AS label
        
    FROM play_df A
    LEFT JOIN down_df B
    ON A.uid = B.uid
    """
df_data = spark.sql(query).toPandas()

## Drop the values that are larger than 99% percentile

df_stats = df_data.describe(percentiles=[0.01, 0.025, 0.25, 0.5, 0.75, 0.975, 0.99])
df_stats = df_stats.iloc[:, 1:]

df_ex_outlier = df_data.iloc[:,1:-1]
df_ex_outlier = df_ex_outlier.apply(lambda x: x[(x < df_stats.loc['99%', x.name])], axis=0)


# bring back the 'uid'
df_ex_outlier = pd.concat([df_data.loc[:,'uid'], df_ex_outlier, df_data.loc[:,'label']], axis=1)
df_ex_outlier = df_ex_outlier.dropna()

# save the dataframe to csv file
df_ex_outlier.to_csv("churn_features.csv")
