# Hive

## Interactive Commands (after executing "hive" in command line)

show databases;
show tables in dbname
show partitions dbname.tablename

### Repair tables

msck repair table dbname.tablename

ALTER TABLE dbname.tablename ADD COLUMNS (new_col type);
ALTER TABLE dbname.tablename PARTITION (col_partition type) SET LOCATION "hdfs://hdp1234/path";
ALTER TABLE dbname.tablename REPLACE COLUMNS (col1 type, col2 type) CASCADE;
CREATE EXTERNAL TABLE IF NOT EXISTS dbname.tablename (col_name1 type, col_name2 type) PARTITIONED BY (col_partition type) STORED AS ORC location 'path';
DELETE FROM dbname.tablename WHERE col = 'value';
DESCRIBE dbname.tablename
DESCRIBE FORMATTED dbname.tablename
DESCRIBE FORMATTED dbname.tablename (col_partition='value');
DROP TABLE IF EXISTS dbname.tablename;
INSERT INTO dbname.tablename values ('col1_value_1', "col2_value_1")('col1_value_2', "col2_value_2");
LOCATION 'path';
SELECT * from dbname."tablename$partitions"
SHOW CREATE TABLE dbname.tablename
SHOW partitions dbname.tablename;

### Types

DOUBLE, STRING

## Commands

hive -e "Interactive command" ex. hive -e "SHOW databases;"
hive --orcfiledump filepath

### Execute an hive file

hive -f file.hql

### Delete

hive --orcfiledump -d path
