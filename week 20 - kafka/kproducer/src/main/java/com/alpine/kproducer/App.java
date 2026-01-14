package com.alpine.kproducer;

import com.alpine.Unchecked;
import org.apache.kafka.clients.producer.ProducerConfig;
import org.apache.kafka.common.serialization.StringSerializer;

import java.util.Optional;
import java.util.Properties;

public class App {
    public static void main(String[] args) throws InterruptedException {
        Optional<String> kafkaBroker = Optional.ofNullable(System.getenv("KAFKA_BROKER"));

        Properties props = new Properties();
        props.put(ProducerConfig.BOOTSTRAP_SERVERS_CONFIG, kafkaBroker.orElse("localhost:9094"));
        props.put(ProducerConfig.KEY_SERIALIZER_CLASS_CONFIG, StringSerializer.class.getName());
        props.put(ProducerConfig.VALUE_SERIALIZER_CLASS_CONFIG, StringSerializer.class.getName());

        props.put(ProducerConfig.ACKS_CONFIG, "all"); // Ensures records are safely stored in Kafka
        props.put(ProducerConfig.RETRIES_CONFIG, 3); // If a send fails with a retriable error (e.g. leader election, temporary network issue), Kafka will retry automatically.
        props.put(ProducerConfig.ENABLE_IDEMPOTENCE_CONFIG, true); // Guarantees that retries will not create duplicates.
        props.put(ProducerConfig.LINGER_MS_CONFIG, 5); // The producer will wait up to 5 ms before sending a batch, hoping more records arrive so they can be sent together.
        props.put(ProducerConfig.BATCH_SIZE_CONFIG, 32_768); // The maximum size of a batch per partition before it’s sent immediately.

        Producer producer = new Producer(props);
        producer.start();

        while (true) {
            Unchecked.threadSleep(1000);
        }
    }
}
