**Multisets** (also called bags) are data structures that allow elements to appear multiple times, unlike sets where duplicates are not allowed. Under the hood, they are typically implemented as [[Binary Search Tree|binary search trees]], specifically [[Red-Black Tree| red-black trees]].

In C++, they are available through `std::multiset`.

%% #todo #refactor %%

### What Are They Useful For?

- **Efficient ordered storage** of duplicate elements.
- **Logarithmic time complexity** for insertions, deletions, and searches.
- **Automatic element sorting**, allowing quick access to the smallest/largest values.

### What Problems Are They Used to Solve?

- **Efficient frequency tracking** while maintaining element order.
- **Quick retrieval/removal** of min/max elements.
- **Sliding window computations** where order matters.
- **Simulation of priority queue behavior** with more control over element erasure.

### LeetCode Examples

- **295. Find Median from Data Stream**
    - **Problem:** Continuously find the median of a growing list of numbers.
    - **Solution:** Maintain two multisets (or priority queues)—one for the lower half and one for the upper half. Balance them so that the median is easily retrievable.

- **480. Sliding Window Median**
    - **Problem:** Find the median of every sliding window of size `k`.
    - **Solution:** Use a multiset to store the current window. Maintain balance between the lower and upper halves while adding/removing elements efficiently.

- **502. IPO**
    - **Problem:** Maximize capital given a list of projects with associated costs and profits.
    - **Solution:** Use a multiset to efficiently track affordable projects based on available capital.

- **846. Hand of Straights**
    - **Problem:** Determine if a deck of cards can be split into consecutive groups.
    - **Solution:** Use a multiset to track and greedily remove the smallest available elements while forming sequences.

- **1675. Minimize Deviation in Array**
    - **Problem:** Minimize the difference between the maximum and minimum of a modified array.
    - **Solution:** Use a multiset to dynamically track changes and maintain the current deviation.