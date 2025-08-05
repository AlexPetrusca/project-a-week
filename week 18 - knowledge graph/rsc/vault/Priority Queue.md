A **priority queue** is an abstract data structure, similar to a queue, in which each element has an associated priority and elements with high priority are served before elements with low priority.

![[priority_queue_overview.png|500]]

Priority queues are commonly implemented using [[Heap|heaps]], giving $O(\log n)$ performance for inserts and removals, and $O(n)$ to build the heap initially from a set of $n$ elements.

### Operations

Priority queues support the following operations:

Basic
- *enqueue*: add an element to the queue with an associated priority.
- *dequeue*: remove the highest priority element from the queue, and return it.
- *delete*: remove an element from the queue.
- *peek*: return the highest priority element from the queue.

Inspection
- *size*: return the number of elements in the queue.
- *is_empty*: check whether the queue has no elements.

### Equivalence of priority queues and sorting algorithms

The semantics of priority queues naturally suggest a sorting method:
 1. insert all the elements to be sorted into a priority queue
 2. sequentially remove them; they will come out in sorted order.

| Name               | Priority Queue Implementation     | Best             | Average          | Worst            |
| ------------------ | --------------------------------- | ---------------- | ---------------- | ---------------- |
| [[Heapsort]]       | Heap                              | $n \log n$        | $n \log n$        | $n \log n$         |
| Smoothsort         | Leonardo Heap                     | $n$              | $n \log n$        | $n \log n$         |
| Selection sort     | Unordered Array                   | {.bad-cell}$n^2$ | {.bad-cell}$n^2$  | {.bad-cell}$n^2$  |
| [[Insertion sort]] | Ordered Array                     | $n$              | {.bad-cell}$n^2$  | {.bad-cell}$n^2$  |
| Tree sort          | Self-balancing binary search tree | $n \log n$        | $n \log n$        | $n \log n$         |

A sorting algorithm can also be used to implement a priority queue. If we can sort up to $n$ keys in $S(n)$ time per key, then there is a priority queue supporting *enqueue*, *dequeue*, and *delete* in $O(S(n))$ time and *peek* in constant time. 
- For example, if one has an $O(n \log n)$ sort algorithm, one can create a priority queue with $O(1)$ polling and $O(\log n)$ insertion.

### Applications

Generally, priority queues share the same applications as heaps (see [[Heap#Applications]]).