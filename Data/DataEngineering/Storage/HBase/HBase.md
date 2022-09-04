# Commands


| Command                                          | Description                                                           |
| -------------------------------------------------- | :---------------------------------------------------------------------- |
| list                                             | List all tables in HBase                                              |
| list_namespace                                   |                                                                       |
| describe                                         | Ex: describe 't1'                                                     |
| scan 'table'                                     | read values of table 'table'                                          |
| put 'table', 'test', 'json_content', 'value', 10 | add "'test', 'json_content', 'value', 10" as one row in table 'table' |



# Notes

To use HBase with a different language than Java, you could use Thrift as interface. To launch Thrift ready to use by HBase execute:

```shell
hbase-master/bin/hbase-daemon.sh start thrift -p {port1} --infoport {port2}
```


# References

- Commands:
  - https://sparkbyexamples.com/hbase/hbase-shell-commands-cheat-sheet/
- Data structure:
  - https://www.tutorialspoint.com/hbase/hbase_overview.htm#:~:text=What%20is%20HBase%3F,huge%20amounts%20of%20structured%20data.
