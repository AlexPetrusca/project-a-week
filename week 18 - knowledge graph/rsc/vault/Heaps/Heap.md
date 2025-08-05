A **heap** is a tree-based data structure that satisfies the heap property:
- *max-heap*: for any parent node $P$ and child node $C$, $P \geq C$.
- *min-heap*: $P \leq C$.

![[heap_overview.png]]

The heap is one maximally efficient implementation of a [[Priority Queue]], where elements with higher (or lower) priority values are served before those with lower (or higher) priority values. The highest (or lowest) priority element is always stored at the root of the heap.

Heaps are often implemented as arrays, where the parent-child relationships are defined by the positions of elements in the array.

### Operations

Heaps support the following operations:

Basic
- *find-max* (or *find-min*): return the maximum (or minimum) item of the heap (a.k.a. *peek*)
- *insert*: add a new key to the heap (a.k.a., *push*).
- *extract-max* (or extract-min): return the maximum (or minimum) item after removing it from the heap (a.k.a., *pop*).
- *delete-max* (or *delete-min*): remove the root node of the heap.

Creation
- *heapify*: create a heap out of given array of elements.
- *meld*: join two heaps to form a valid new heap containing all the elements of both.

Inspection
- *size*: return the number of items in the heap.
- *is-empty*: return true if the heap is empty, false otherwise.

Internal
- *increase-key* or *decrease-key*: update a key within the heap.
- *delete*: delete an arbitrary node.
- *heapify-up*: move a node up in the tree, as long as needed; used to restore heap condition after insertion.
- *heapify-down*: move a node down in the tree, similar to heapify-up; used to restore heap condition after deletion or replacement.

### Applications

Heaps are commonly used in the following applications:
- **Sorting**: Efficient, general-purpose sorting algorithms can be implemented through the construction of in-place heaps; examples include [[heapsort]] and smoothsort.
- **Priority queue**: A heap is a maximally efficient implementation of a priority queue.
- **Graph algorithms**: Efficient implementations of Prim's minimal-spanning-tree algorithm and Dijkstra's shortest-path algorithm use heaps as internal traversal data structures.
- **K-way merge**: A heap data structure is useful to merge many already-sorted input streams into a single sorted output stream.
- **Best-first search**: Algorithms like A\* use priority queues (heaps) to keep track of unexplored routes and to select the most promising route to expand.