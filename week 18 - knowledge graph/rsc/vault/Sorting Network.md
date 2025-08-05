**Sorting networks** are comparison networks that always sort their input. They can achieve sub-linear performance through parallelization, but can only handle fixed-size inputs.

![[sorting_network_overview.png]]

A sorting network consists of two types of items:
- **Wires**: The wires are thought of as running from left to right, carrying values (one per wire) that traverse the network all at the same time.
- **Comparators**: Each comparator connects two wires. When a pair of values, traveling through a pair of wires, encounter a comparator, the comparator swaps the values if and only if the top wire's value is greater or equal to the bottom wire's value (i.e. $top \geq bottom$).

Sorting networks can be implemented either in hardware or in software.

### Zero-One Principle

The *zero-one principle* says that if a sorting network works correctly when each input is drawn from the set $\{0, 1\}$, then it works correctly on arbitrary input numbers. 

### Parallelism

Sorting networks can perform certain comparisons in parallel. In the graphical notation, this is represented by comparators that lie on the same vertical line or close to one another. Each group of comparisons that can be performed in parallel is referred to as a *time step*.

The parallelization of sorting networks, particularly [[Sorting Network#Bitonic Mergesort|bitonic sorting networks]], makes them a popular choice for sorting sorting large numbers of elements on the GPU.

### Complexity Analysis

The efficiency of a sorting network can be measured by:
- **Total Size**: the number of comparators in the network.
- **Depth**: the largest number of comparators that any input value can encounter on its way through the network; or equivalently, the number of time steps in the network.

The space complexity of a sorting network is given by its total size. Its time complexity is given by its depth.

### Insertion / Bubble Sort Network

Insertion sort (or equivalently, bubble sort) can easily be represented as a sorting network.

![[sorting_network_insertion_sort.png|400]]

The insertion sort network (or equivalently, bubble sort network) has a depth of $2n - 3$, so its runtime performance is $O(n)$. It has a total size of $\sum_{k=1}^{n} k = {\frac {n(n+1)}{2}}$, so its space complexity is $O(n^2)$.

### Bitonic Mergesort

*Bitonic mergesort* is a sorting network that works by merging the output of bitonic sorters.
- A *bitonic sorter* is a network that sorts bitonic sequences. 
- A *bitonic sequence* is a sequence that either monotonically increases and then monotonically decreases, or else monotonically decreases and then monotonically increases.

![[sorting_network_bitonic_sort_directional.png]]

In the above diagram of a bitonic mergesort network:
- the blue boxes output fully sorted sequences.
- the green boxes output fully sorted sequences in the opposite order.
- the red boxes output bitonic sequences.

Note that every green and every blue box is a bitonic sorter. By concatenating the outputs of a "green" and a "blue" $n$-input bitonic sorter, we get a bitonic sequence of length $2n$ that can then be fed to a $2n$-input bitonic sorter. We can then repeat the same process for the output of a "green" and a "blue" $2n$-input bitonic sorter and so on. 

By recursively composing bitonic sorters in this way, we can construct a bitonic mergesort network that sorts any input whose length is a power of two.

![[sorting_network_bitonic_sort.png]]

The diagram above is equivalent, but refactored to remove directional comparators:
- the blue boxes output fully sorted sequences.
- the red boxes output bitonic sequences.
- the orange boxes are equivalent to red boxes where the sequence order is reversed for the bottom half of its inputs and the bottom half of its outputs.

Bitonic mergesort networks have a $O(\log ^{2} n)$ worst-case performance in parallel time and a $O(n \log ^{2} n)$ space complexity.

### Batcher's Odd–Even Mergesort

*Batcher odd-even mergesort* is a sorting network that works by merging sorted outputs separately on odd keys and even keys.

![[sorting_network_batcher_odd_even_mergesort.png|400]]

In the diagram above, note that the red and green sections are also composed of odd-even mergesort networks. We can recursively compose networks in this way to construct an odd-even mergesort network that sorts any input whose length is a power of two.

Batcher odd-even mergesort networks have a $O(\log ^{2} n)$ worst-case performance in parallel time and a $O(n \log ^{2} n)$ space complexity.

Compared to bitonic mergesort, Batcher's odd-even mergesort takes fewer comparisons to sort input. However, in practice, bitonic mergesort typically performs better (2-3x faster) than Batcher's odd-even mergesort due to better locality of reference.

### Applications

Sorting networks, particularly bitonic sorting networks, are commonly employed to take advantage of the parallel nature of GPU architectures.

The second GPU Gems book popularized Batcher odd–even mergesort and bitonic mergesort, as easy ways of doing reasonably efficient sorts on graphics-processing hardware.