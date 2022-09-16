
# HDFS commands

## List and give details about an HDFS directory

Give the size of files on Kb(or KB?).

```bash
hdfs dfs -du $PATH
```

Give the size of files in a human readable way.

```bash
hdfs dfs -du -h $PATH
```

As ls on linux

```bash
hdfs dfs -ls $PATH
```

## Rename

```bash
hdfs dfs -mv $PATH $PATH_WITH_NEW_NAME
```

## Delete

### Delete directory

```bash
hdfs dfs -rm -r $PATH
```

### Delete many files using pattern

```bash
hdfs dfs -rm -r $PATH/part-001*
```

## Copy

### Copy from HDFS to local filesystem

```bash
hdfs dfs -get $PATH_HDFS $PATH_LOCAL
```

### Copy from local to HDFS

```bash
hdfs dfs -put $PATH_LOCAL $PATH_HDFS
```

```bash
hdfs dfs -moveFromLocal $PATH_LOCAL $PATH_HDFS
```

### Copy inside HDFS

```bash
hdfs dfs -cp $PATH1 $PATH1
```

To force

```bash
hdfs dfs -cp -f $PATH1 $PATH1
```

### Read file

hdfs dfs -ls $PATH