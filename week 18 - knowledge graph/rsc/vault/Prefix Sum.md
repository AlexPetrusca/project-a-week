A **prefix sum** (also known as a *running sum*) is a technique used in computer science and mathematics to precompute the sum of elements in an array up to a given index. 

This allows for efficient computation of range sums or other aggregate queries over a sequence of numbers. The prefix sum array is constructed such that each element at index $i$ represents the sum of all elements from the start of the original array up to index $i$.

### Complexity Analysis

| Operation   | Time Complexity | Space Complexity |
| ----------- | --------------- | ---------------- |
| Construct   | $O(n)$          | $O(n)$           |
| Query Range | $O(1)$          | $O(n)$           |
| Update      | $O(n)$          | $O(n)$           |

### Structure

For an array $A$, the prefix sum array $P$ is defined as:

$$
\begin{align}
    P[0] &= A[0] \\
    P[1] &= A[0] + A[1] \\
    P[2] &= A[0] + A[1] + A[2] \\
    \ldots \\
    P[i] &= A[0] + A[1] + \ldots + A[i] \\
\end{align}
$$

### Operation

##### Construct

To build the prefix sum array $P$ from an array $A$:
1. Set $P[0] = A[0]$.
2. For each subsequent index $i$ from 1 to $n-1$, compute $P[i] = P[i-1] + A[i]$.

##### Query Range

To find the sum of elements in $A$ from index $i$ to $j$ (inclusive):
- If $i = 0$, the sum is simply $P[j]$.
- If $i > 0$, the sum is $P[j] - P[i-1]$.

#### Update

If an element $A[k]$ is updated, all prefix sums from $P[k]$ to $P[n-1]$ must be recalculated, which takes $O(n)$ time. 

For dynamic updates, alternative data structures like [[Computer Science/Algorithms & Data Structures/Data Structures/Trees/Segment Tree|Segment Trees]] or [[Fenwick Tree|Fenwick Tree]] are preferred.

### Variants

- **[[Integral Image|2D Prefix Sum]]**: Extends the concept to 2D arrays (matrices) for computing rectangular range sums. For a matrix $M$, the 2D prefix sum at $(i, j)$ is the sum of all elements in the submatrix from $(0, 0)$ to $(i, j)$.
- **Modular Prefix Sum**: Used in problems involving modular arithmetic, where each prefix sum is computed modulo some value.
- **Prefix \***: The prefix sum can be easily generalized to operations other than addition. For example, prefix product for multiplication and prefix xor.