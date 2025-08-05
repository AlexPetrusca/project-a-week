A **binary heap** is a [[heap]] that is implemented as a binary tree. A binary heap is a common way of implementing a [[priority queue]] and is the data structure behind [[heapsort]].

![[heap_overview.png|center]]

A binary heap is defined as a binary tree with two additional constraints:
- **Shape Property**: a binary heap is a complete binary tree.
- **Heap Property**: for any parent node $P$ and child node $C$, $P \geq C$ (or $P \leq C$).

### Complexity Analysis

| Operation               | Average     | Worst Case  |
| ----------------------- | ----------- | ----------- |
| Search                  | $Θ(n)$       | $O(n)$      |
| Insert                  | $Θ(1)$       | $O(\log n)$  |
| Find-Min (Find-Max)     | $Θ(1)$       | $O(1)$      |
| Delete-Min (Delete-Max) | $Θ(\log n)$   | $O(\log n)$ |
| **Space**               | $Θ(n)$       | $O(n)$      |

### Structure

Because a binary heap is always a complete binary tree, it can be stored compactly in an array. No space is required for pointers; instead, the parent and children of each node can be found by arithmetic on array indices.

For an element at index $i$ in the array, its children and parent are at indices:
$$i_{parent} = \frac{i − 1}{2}$$
$$i_{left-child} = 2i + 1$$
$$i_{right-child} = 2i + 2$$

### Operations

A binary heap supports all general operations of a heap (see [[Heap#Operations]]).

##### Heapify (Construct)

The `heapify` operation refers to constructing a heap from an array of $n$ elements. If done naively ($n$ inserts into an empty heap), heap construction takes $O(n \log n)$ time. A faster method, generally referred to as Floyd's heap construction algorithm or "bottom-up heap construction", can construct a heap in optimal $O(n)$ time.

For a binary heap, Floyd's method works as follows:
1. Begin with the last non-leaf node in the array, which is at index $i = \frac{n}{2} - 1$.
2. For each element $e$ from indices $[i..0]$, perform `heapify-down` on $e$.

This process essentially treats the input array as a binary tree and "heapifies" all subtrees, starting from bottom to top, to build the final heap.

##### Heapify-Up

The `heapify-up` operation refers to moving a violating element $e$ up in the tree, as long as needed, until the heap condition is restored.

In a min-heap, this is done by repeatedly swapping $e$ with its parent $p$ until $e > p$. In a max-heap, repeatedly swap $e$ with its parent until $e < p$.

##### Heapify-Down

The `heapify-down` operation refers to moving a violating element $e$ down in the tree, as long as needed, until the heap condition is restored.

In a min-heap, this is done by repeatedly swapping $e$ with its smaller child until $e$'s children are both larger than $e$. In a max-heap, repeatedly swap $e$ with its larger child until $e$'s children are both smaller than $e$.

##### Find-Min (or Find-Max)

Simply return the root of the heap.

##### Extract-Min (or Extract-Max)

To extract the root of the heap $e$:
1. Replace the root of the heap with the last element in the array $e$ 
2. Perform `heapify-down` on $e$ 
3. Return the extracted min (or max)

![[binary_heap_extract1.png]]
![[binary_heap_extract2.png]]
![[binary_heap_extract3.png]]
 { .image-group }

##### Insert

To insert a new element $e$ into the heap:
1. Append $e$ to the end of the array.
2. Perform `heapify-up` on $e$ 

![[binary_heap_insert1.png]]
![[binary_heap_insert2.png]]
![[binary_heap_insert3.png]]
 { .image-group }
 
Insertions take $O(\log n)$ time in the worst case, but take $O(1)$ on average due to key characteristics of the heap[^1].

##### Delete

To delete for an arbitrary element $e$:
1. Replace the target element with the last element in the array $e$ 
2. Perform `heapify-down` on $e$ 

##### Search

To search for an arbitrary element $e$:
1. Start from the root.
2. Recursively search children; prune the search at a child node $c$ if:
    - in a max-heap: $c < e$, since children of $c$ can only be smaller than $e$
    - in a min-heap: $c > e$, since children of $c$ can only be larger than $e$
3. Return the found element.

### Applications

In practice, binary heaps are used to implement the following:
- **Priority Queues**: A queue where items are dequeued based on their priority.
- **Heapsort**: Heapsort works by building an in-place, implicit binary heap and repeatedly extracting the root of the heap to grow a sorted region until the whole array is sorted.


[^1]: https://stackoverflow.com/questions/39514469/argument-for-o1-average-case-complexity-of-heap-insertion
