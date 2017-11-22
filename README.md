# User Churn Classifier

## Purpose of Project

Two datasets of user activity log data from [kuwo.cn Music Box](http://kuwo.cn/)
was made available for analysis. The datasets reflect users' music playing and
music downloading activities from March 29, 2017 to May 12, 2017. In this
project, we parsed relevant features indicating user engagement from the raw
log data, and analyzed the contributing factors leading to user voluntary
churn (i.e., users stop using the Music Box App after 3 time windows) using
classification machine learning models.

## Data Preprocessing & Feature Extraction with Apache Spark

Each record of user music play log contains 10 fields: `PlayDate`, `UserID`,
`Device`, `SongID`, `SongType`, `SongName`, `Artist`, `UserPlayTime`,
`SongLength`, `PlayFlag`. 

Each record of user music download log contains 7 fields: `DownloadDate`,
`UserID`, `Device`, `SongID`, `SongName`, `Artist`, `DownloadFlag`.

Apache Spark was used to parse the raw log data since the total size of the
play and download log datasets was greater than 14 gigabytes. The play and
download log data were aggregated according to UserID for four time windows
(see below for details), and 23 new features were extracted. 

**_New features extracted_**:

* `play_freq`: The number of times a user listened to music in a time window.
* `play_perc`: The average percentage of song length that the user listened to
  in a time window.
* `play_songs`: The number of distinct songs a user listened to in a time
   window.
* `play_singers`: The number of distinct singers a user listened to in a time
   window.
* `play_sum`: The sum of music time (in seconds) a user listened to in a time
  window.
* `days_from_last_play`: The number of days passed since the user played music
   last time.
* `down_freq`: The number of times a user downloaded songs in a time window.
* `down_singers`: The number of distinct singers a user downloaded music from in
   a time window.
* `days_from_last_down`: The number of days passed since the user downloaded
   music last time.
* `play_label`: 1 if user had no music playing activity in the fourth time window
   (see below), 0 otherwise.
* `down_label`: 1 if user had no music downloading activity in the fourth time
   window, 0 otherwise.

**_Time windows_**:

The days in which the data was available were divided into four time windows.
The first three time windows were used to observe users' music playing and
downloading activities. The fourth time window was used to define whether a
user had become churned or not. Churned users are defined as users who neither
played nor downloaded activities in the fourth window.

**_Joining data_**:

After the features were extracted from both the music play and download
datasets, and from the three observation windows as well as the fourth churned
user definition window, Spark SQL was used to join the features from different
tables together. Churned labels were assigned if users had 1s in both
`play_label` and `down_label`. This large dataframe will be used for machine
learning modeling purpose.

## Classification Machine Learning Models

After the large dataframe was ready, we built and tested different
classification models on this dataframe. Models used include: Logistic
Regression, Naive Bayes Classifier, Random Forest, and XGBoost Classifier.
Hyperparameters for these models were tuned using grid search, and models were
retrained with the tuned parameters. 

Model performance were compared by checking the accuracy, precision, recall, F-1
score, and Area Under RoC (AUC score) measures.
