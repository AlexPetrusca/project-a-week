package com.alpine.kconsumer;

import com.alpine.Unchecked;
import org.apache.kafka.clients.consumer.ConsumerConfig;
import org.apache.kafka.clients.consumer.ConsumerRecord;
import org.apache.kafka.clients.consumer.ConsumerRecords;
import org.apache.kafka.clients.consumer.KafkaConsumer;
import org.apache.kafka.common.serialization.StringDeserializer;

import java.time.Duration;
import java.util.Collections;
import java.util.Properties;

public class Consumer {
    private final KafkaConsumer<String, String> consumer;
    private volatile boolean running;
    private Thread workerThread;

    public Consumer(Properties props, String topic) {
        running = false;
        this.consumer = new KafkaConsumer<>(props);
        consumer.subscribe(Collections.singletonList(topic));
    }

    public synchronized void start() {
        if (running) return;

        running = true;
        workerThread = new Thread(this::runLoop, "kafka-consumer-thread");
        workerThread.start();
    }

    public synchronized void stop() {
        if (!running) return;

        running = false;
        workerThread.interrupt(); // terminate workerThread
        Unchecked.threadJoin(workerThread); // wait for it to terminate
    }

    public synchronized void close() {
        stop();
        consumer.close();
    }

    private void runLoop() {
        while (running) {
            ConsumerRecords<String, String> records = consumer.poll(Duration.ofMillis(1000));
            for (ConsumerRecord<String, String> record : records) {
                System.out.printf("Consumed message: key=%s, value=%s, partition=%d, offset=%d%n",
                        record.key(), record.value(), record.partition(), record.offset());
            }
            Unchecked.threadSleep(1000);
        }
    }

    public static void main(String[] args) {
        Properties props = new Properties();
        props.put(ConsumerConfig.BOOTSTRAP_SERVERS_CONFIG, "localhost:9092");
        props.put(ConsumerConfig.GROUP_ID_CONFIG, "my-group");
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
