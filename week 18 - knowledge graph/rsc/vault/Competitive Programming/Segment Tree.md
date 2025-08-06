A **[[Computer Science/Algorithms & Data Structures/Data Structures/Trees/Segment Tree|Segment Tree]]** is a way to store array data in a tree structure, where each node represents a range of the array. It is used to efficiently handle range queries and updates on an array.

![[segment_tree_overview.png|center|pad10]]

Imagine you have an array of numbers and want to quickly find the sum (or min, max, etc.) of any range of elements, like from index 3 to 7, while also being able to update individual elements. Doing this directly on the array would be slow for large ranges or frequent updates.

A segment tree solves this by organizing the array into a binary tree. Each node in the tree represents a segment (or range) of the array:
- The root represents the entire array.
- Each child node covers half of its parent's range, splitting it into left and right segments.
- The leaves represent individual array elements.

When you query a range (e.g., sum from index 1 to 3), the tree combines only the relevant segments, avoiding unnecessary work. Updates (e.g., changing one element) propagate up the tree, adjusting the affected nodes. This makes both queries and updates fast - typically O(log n) time, where n is the array size - compared to O(n) for a plain array.

%% #todo #refactor %%

### What Are They Useful For?

- **Fast range queries** (sum, min, max, GCD, etc.) in **O(log n)**.
- **Efficient point updates** in **O(log n)**.
- **[[Computer Science/Algorithms & Data Structures/Data Structures/Trees/Segment Tree#Lazy Propagation|Lazy propagation]]** for **fast range updates**.

### What Problems Are They Used to Solve?

- **Range sum/min/max queries** where updates occur dynamically.
- **Range updates with lazy propagation** (e.g., incrementing a subarray).
- **Finding the first/last occurrence** of an element within a range.
- **Handling dynamic constraints** where other structures (like prefix sums) would be inefficient.

### LeetCode Examples

- **307. Range Sum Query - Mutable**
    - **Problem:** Support `update(i, val)` and `sumRange(left, right)`.
    - **Solution:** Use a segment tree to store cumulative sums. Queries and updates both run in **O(log n)**.

- **315. Count of Smaller Numbers After Self**
    - **Problem:** Given an array, count how many smaller elements appear after each index.
    - **Solution:** Use a segment tree to maintain a frequency array and perform efficient range queries.

- **327. Count of Range Sum**
    - **Problem:** Count subarrays whose sums fall within a given range `[lower, upper]`.
    - **Solution:** Use a segment tree to efficiently count prefix sums in a given range.

- **699. Falling Squares**
    - **Problem:** Given squares falling on a 1D plane, determine the maximum height at each step.
    - **Solution:** Use a segment tree with lazy propagation to track height changes efficiently.

- **1202. Smallest String With Swaps**
    - **Problem:** Given pairs of indices, find the lexicographically smallest string possible after swaps.
    - **Solution:** A segment tree helps efficiently track character counts and reconstruct the answer.