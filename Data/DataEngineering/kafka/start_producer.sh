set -x

# Install: https://tecadmin.net/how-to-install-apache-kafka-on-ubuntu-20-04/

sudo systemctl start zookeeper
sudo systemctl start kafka
# sudo systemctl status kafka

/usr/local/kafka/bin/kafka-topics.sh --create --bootstrap-server localhost:9092 --replication-factor 1 --partitions 1 --topic testTopic
# to check list of topics:
# bin/kafka-topics.sh --list --zookeeper localhost:9092

/usr/local/kafka/bin/kafka-console-producer.sh --broker-list localhost:9092 --topic testTopic