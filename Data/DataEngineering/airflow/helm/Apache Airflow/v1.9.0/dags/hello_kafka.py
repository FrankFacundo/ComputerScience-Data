"""
DAG Airflow for Kafka
To get group.id read file /usr/local/kafka/config/consumer.properties
"""

import json
import logging
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator


from airflow_provider_kafka.operators.consume_from_topic import ConsumeFromTopicOperator

default_args = {
    "owner": "airflow",
    "depend_on_past": False,
    "start_date": datetime(2021, 7, 20),
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


consumer_logger = logging.getLogger("airflow")
def consumer_function(message, prefix=None):
    key = json.loads(message.key())
    value = json.loads(message.value())
    consumer_logger.info(f"{prefix} {message.topic()} @ {message.offset()}; {key} : {value}")
    return


def hello_kafka():
    print("Hello Kafka !")
    return

with DAG(
    "kafka-example",
    default_args=default_args,
    description="Examples of Kafka Operators",
    schedule_interval=timedelta(days=1),
    start_date=datetime(2021, 1, 1),
    catchup=False,
    tags=["example"],
) as dag:


    t2 = ConsumeFromTopicOperator(
        task_id="consume_from_topic",
        topics=["testTopic"],
        apply_function="hello_kafka.consumer_function",
        apply_function_kwargs={"prefix": "consumed:::"},
        consumer_config={
            "bootstrap.servers": "127.0.0.1:9092",
            "group.id": "test-consumer_group",
            "enable.auto.commit": False,
            "auto.offset.reset": "beginning",
        },
        commit_cadence="end_of_batch",
        max_messages=10,
        max_batch_size=2,
    )

    t2.doc_md = 'Reads a series of messages from the `test_1` topic, and processes them with a consumer function with a keyword argument.'


    t6 = PythonOperator(
        task_id='hello_kafka',
        python_callable=hello_kafka
    )

    t6.doc_md = 'The task that is executed after the deferable task returns for execution.'
    
    t2 >> t6