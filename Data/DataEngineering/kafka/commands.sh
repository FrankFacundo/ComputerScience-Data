/usr/local/kafka/bin/kafka-console-consumer.sh --bootstrap-server localhost:9092 --topic testTopic --from-beginning
/usr/local/kafka/bin/kafka-console-consumer.sh --bootstrap-server localhost:9092 --topic testTopic --from-beginning --security-protocol PLAINTEXTSASL
/usr/local/kafka/bin/kafka-console-consumer.sh --bootstrap-server localhost:9092 --topic testTopic --offset --security-protocol PLAINTEXTSASL
/usr/local/kafka/bin/kafka-console-consumer.sh --bootstrap-server localhost:9092 --topic testTopic --security-protocol 