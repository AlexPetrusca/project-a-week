package com.alpine.kconsumer;

import com.alpine.Unchecked;
import org.apache.kafka.clients.consumer.ConsumerConfig;
import org.apache.kafka.common.serialization.StringDeserializer;

import java.util.Optional;
import java.util.Properties;

public class App {
    public static void main(String[] args) {
        Optional<String> kafkaBroker = Optional.ofNullable(System.getenv("KAFKA_BROKER"));
        Optional<String> consumerGroup = Optional.ofNullable(System.getenv("KAFKA_CONSUMER_GROUP"));

        Properties props = new Properties();
        props.put(ConsumerConfig.BOOTSTRAP_SERVERS_CONFIG, kafkaBroker.orElse("localhost:9094"));
        props.put(ConsumerConfig.GROUP_ID_CONFIG, consumerGroup.orElse("default-group"));
        props.put(ConsumerConfig.KEY_DESERIALIZER_CLASS_CONFIG, StringDeserializer.class.getName());
        props.put(ConsumerConfig.VALUE_DESERIALIZER_CLASS_CONFIG, StringDeserializer.class.getName());

        props.put(ConsumerConfig.AUTO_OFFSET_RESET_CONFIG, "earliest"); // start from beginning if no offset
        props.put(ConsumerConfig.ENABLE_AUTO_COMMIT_CONFIG, true); // auto commit offsets (pick up where it left off)

        Consumer consumer = new Consumer(props, "my-topic");
        consumer.start();

        while (true) {
            Unchecked.threadSleep(1000);
        }
    }
}
