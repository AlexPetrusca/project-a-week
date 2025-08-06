A **monotonic stack** is a stack that maintains elements in either strictly increasing or decreasing order. It is useful for efficiently solving problems involving **next/previous greater/smaller elements** in **O(n) time**.

![[algorithms_monotonic_stack.png]]

In C++, they are typically implemented using `std::stack` or `std::vector` with manual pop operations.

%% #todo #refactor %%

### What Are They Useful For?

- **Efficient range queries** for nearest greater/smaller elements.
- **Reducing nested loops** in problems that require comparisons across indices.
- **Optimizing dynamic programming transitions** when decisions depend on previous elements.

### What Problems Are They Used to Solve?

- **Finding Next/Previous Greater or Smaller Elements.**
- **Histogram problems** where rectangle areas need to be computed efficiently.
- **Stock span and temperature-based queries.**
- **Trapping rainwater problems** where boundary constraints are needed.

### LeetCode Examples

- **739. Daily Temperatures**
    - **Problem:** Given an array of temperatures, find how many days until a warmer temperature appears.
    - **Solution:** Use a **monotonic decreasing stack** to keep track of indices. Pop when a higher temperature is found to determine wait days.

- **496. Next Greater Element I**
    - **Problem:** For each element in `nums1`, find its next greater element in `nums2`.
    - **Solution:** Use a **monotonic decreasing stack** to preprocess next greater elements in `nums2` in **O(n)**, then query in **O(1)** using a hash map.

- **503. Next Greater Element II**
    - **Problem:** Like 496, but `nums2` is circular.
    - **Solution:** Use a **monotonic stack**, iterate twice to simulate a circular array.

- **84. Largest Rectangle in Histogram**
    - **Problem:** Given a histogram, find the largest rectangular area.
    - **Solution:** Use a **monotonic increasing stack** to track increasing heights. Process each bar’s **left and right boundaries** efficiently.

- **42. Trapping Rain Water**
    - **Problem:** Compute the amount of trapped rainwater given elevation heights.
    - **Solution:** Use a **monotonic decreasing stack** to track boundaries and compute trapped water efficiently.

- **901. Online Stock Span**
    - **Problem:** Given a stream of stock prices, return the number of consecutive days the price was less than or equal to today.
    - **Solution:** Use a **monotonic decreasing stack** storing (price, span) pairs to efficiently compute spans.
