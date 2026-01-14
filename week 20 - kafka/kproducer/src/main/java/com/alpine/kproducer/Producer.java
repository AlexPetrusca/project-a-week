package com.alpine.kproducer;

import com.alpine.Unchecked;
import org.apache.kafka.clients.producer.KafkaProducer;
import org.apache.kafka.clients.producer.ProducerRecord;

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
}
