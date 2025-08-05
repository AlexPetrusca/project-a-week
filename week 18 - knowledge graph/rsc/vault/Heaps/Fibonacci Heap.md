A **fibonacci heap** is a [[heap]] data structure consisting of a collection of heap-ordered trees. Fibonacci heaps are named after the [[Fibonacci Sequence|Fibonacci numbers]], which are used in their running time analysis.

![[fibonacci_heap_overview.png]]

### Complexity Analysis

| Operation                   | Average    |
| --------------------------- | ---------- |
| Insert                      | $Θ(1)$      |
| Find-Min (Find-Max)         | $Θ(1)$      |
| Delete-Min (Delete-Max)     | $Θ(\log n)$  |
| Decrease-Key (Increase-Key) | $Θ(1)$      |
| Merge                       | $Θ(1)$      | 

### Structure

The structure of a Fibonacci heap differs significantly from that of a binary heap:
- There are multiple root nodes
    - Every root node is linked using a circular doubly linked list
    - Every tree associated with a root satisfies the heap property
    - Maintain a pointer to the min (or max) root
- Every node can have >2 children

A Fibonacci heap is basically a collection of heap trees. The trees do not have a prescribed shape and, in the extreme case, the heap can have every element in a separate tree. The structure of a Fibonacci heap is therefore highly flexible.

The tree is generally considered unordered until the delete-min operation is processed, at which point a "clean up" algorithm merges heap trees in such a way that:
1. The number of root nodes is bounded to $O(\log n)$
2. The maximum number of direct children for a root (its *degree*) is bounded to $O(\log n)$

This "clean up" job ensures that the Fibonacci heap achieves its desired running time. Specifically, it guarantees that, in the worst case, the size of a subtree rooted in a node of degree $k$ is at least $F_{k+2}$, where $F_{k}$ is the $k$-th Fibonacci number. 
- Since Fibonacci numbers increase exponentially with factor $\phi$ (i.e. $\phi^{k}$), the number of roots will be no greater than $O(\log n)$. 
- Similarly, the degree of each root will be no greater than $O(\log n)$.

![[fibonacci_heap_fibonacci_numbers.png]]

### Operations

The key to the Fibonacci heap's amortized performance is *lazy execution* - postponing work for later operations.

##### Find-Min

This is trivial because we keep a pointer to the min-node. Simply return the node. 

This pointer must be updated to always point to the correct node:
- After every delete-min operation, search through all the roots to find the next min-node
- After every insert and decrease-key operation, compare the current min-node with the updated node and swap the two if necessary

##### Insert

Add the node as a new root.

##### Merge

Concatenate the lists of tree roots of the two Fibonacci heaps.

##### Delete-Min

Delete-min operates in three phases:
1. *Remove Minimum*: Remove the direct children of min-node and insert them as new roots.
2. *Clean Up*: Merge roots with the same degree until all roots have unique degrees
    1. Create an array of size $D$, where $D$ is the maximum possible degree
    2. Iterate over the root list, for each root $r$ with degree $d$:
        1. Insert $r$ at index $d$
        2. If there is already a root $s$ at index $d$:
            1. Merge $r$ and $s$ to create a new root $t$ with degree $d + 1$
            2. Delete $r$ and $s$ at index $d$
            3. Insert $t$ at index $d + 1$; recursively merge if necessary
    3. The array now has at most one root at each index (i.e. each root has a different degree)
3. *Rebuild Heap*: Recreate the heap from the intermediate array from the last step
    - Simply add each root $r$ in the array to the Fibonacci heap's list of roots

To merge two sub-trees of a Fibonacci heap, $a$ and $b$:
- If $a > b$, add $a$ as a child of $b$
- If $a < b$, add $b$ as a child of $a$

Note: the *clean up* phase takes longer to process depending on how many prior calls to insert and decrease-key there were. This cost scales linearly with the number of calls. For performance analysis purposes, the cost can be written off and treated as a constant overhead applied to each insert and decrease-key that "we pay for later". This is an example of *amortization*.

##### Decrease-Key

Decrease-key on a node $x$ has two cases:
1. The decrease does NOT violate the heap property
    - Do nothing
2. The decrease violates the heap property
    - Cut $x$ out of its sub-tree and insert it as a new root (let's call this operation "cut out")
    - Mark $x$'s parent $p$ to indicate it has lost a child
        - If a direct child of $p$ is subsequently "cut out":
            - "Cut out" $p$ as well
            - Unmark $p$

![[fibonacci_heap_decrease_key.png|500]]

### Analysis of Amortized Performance

As you can see, the decrease-key operation is very particular, but this is for a good reason: it is the crux of the Fibonacci heap's amortized performance.

At first glance, handling the marking logic seems unnecessary. A simpler method would appear to work: simply cut $x$ out of its subtree and add it to the root list; don't worry about marking its parent $p$. However, this solution can lead to degenerate cases where heap trees have too few children given their degree. This can, in turn, violate one of the properties of Fibonacci heaps, namely that the number of root nodes should be bounded to $O(\log n)$. Without this property, we lose the theoretical performance guarantees of Fibonacci heaps. 

Adding the mark mechanism reduces the impact of these degenerate cases. In the worst case (where a heap tree is full and decrease-key is called until every non-terminal node is marked), a node $x$ with degree $d$ will have a size of at least $F(d + 2)$ nodes, where $F(k)$ is the $k$-th Fibonacci number. For example, a tree of degree $d = 4$ will have at least $F(d + 2) = F(6) = \{1, 1, 2, 3, 5, 8\} =$ 8 nodes. 

This guarantee on a minimal tree size can be used to show that Delete-Min takes at worst $O(\log n)$ time. The amortized performance of a Fibonacci heap depends on the degree of any tree root being $O(\log n)$, where $n$ is the size of the heap. Since Fibonacci numbers scale exponentially and a given tree of degree $d$ is guaranteed to have a size of at least $F(d+2)$, the maximum degree of any tree must be less than $F^{-1}(n) - 2$, which is of the order $O(\log n)$. 

(Note: $F^{-1}(k)$ is some function that is the inverse of the fibonacci function; that is to say: given a number, it will return the index of the closest fibonacci number.)

To demonstrate this, consider the proof by contradiction. Assume this limit does not hold: 
1. Let us define a tree of degree $d > F^{-1}(n) - 2$, where $n$ is the size of the Fibonacci heap.
2. The tree would have a minimum size of $s = F(d + 2)$.
3. Plugging in for $d$, we get:
    1. $s > F(F^{-1}(n) - 2 + 2)$
    2. $s > F(F^{-1}(n))$
    3. $s > n$
4. The size of the tree $s$ is greater than the total number of elements in the fibonacci heap. *Contradiction!*

Having proved the degree of any tree root is at most $O(\log n)$, the runtime of the fibonacci heap's Delete-Min must be of order $O(\log n)$. *Q.E.D.*

### Applications

Theoretically, the fibonacci heap has a better amortized running time than many other priority queue data structures, including the [[binary heap]] and binomial heap. Using Fibonacci heaps as priority queues improves the asymptotic running time of important algorithms, such as [[Dijkstra's Algorithm]] for computing the shortest path between two nodes in a graph.

Fibonacci heaps are rarely used in practice. They are slow in real world applications due to large memory consumption per node and high constant factors on all operations.