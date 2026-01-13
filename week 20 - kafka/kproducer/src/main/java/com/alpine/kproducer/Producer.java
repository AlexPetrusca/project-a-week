package com.alpine.kproducer;

import com.alpine.Unchecked;
import org.apache.kafka.clients.producer.KafkaProducer;
import org.apache.kafka.clients.producer.ProducerConfig;
import org.apache.kafka.clients.producer.ProducerRecord;
import org.apache.kafka.common.serialization.StringSerializer;

import java.util.Properties;

public class Producer {
    private final KafkaProducer<String, String> producer;
    private Thread workerThread;
    private volatile boolean running;

    public Producer(Properties kafkaProps) {
        running = false;
        producer = new KafkaProducer<>(kafkaProps);
    }

    public synchronized void start() {
        if (running) return;

        running = true;
        workerThread = new Thread(this::runLoop, "kafka-producer-thread");
        workerThread.start(); // start workerThread
    }

    public synchronized void stop() {
        if (!running) return;

        running = false;
        workerThread.interrupt(); // terminate workerThread
        Unchecked.threadJoin(workerThread); // wait for it to terminate
    }

    public synchronized void close() {
        stop();
        producer.close();
    }

    private void runLoop() {
        while (running) {
            ProducerRecord<String, String> record = new ProducerRecord<>("my-topic", "key1", "Hello Kafka");
            producer.send(record, (metadata, exception) -> {
                if (exception != null) {
                    exception.printStackTrace();
                } else {
                    System.out.printf("Sent to topic=%s partition=%d offset=%d%n",
                            metadata.topic(), metadata.partition(), metadata.offset());
                }
            });
            Unchecked.threadSleep(1000);
        }
    }

    public static void main(String[] args) throws InterruptedException {
        Properties props = new Properties();
        props.put(ProducerConfig.BOOTSTRAP_SERVERS_CONFIG, "localhost:9092");
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
