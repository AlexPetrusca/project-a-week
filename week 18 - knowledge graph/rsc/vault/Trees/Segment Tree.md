A segment tree is a **binary tree** where
- Each leaf node represents a single element of the original array.
- Each internal node represents the result of merging its two child nodes (e.g., sum, min, max).
- The root stores the result of merging all elements in the array.

![[segment_tree_overview.png|center|pad10]]

It is particularly useful for *efficiently processing range queries* when dealing with problems that involve querying the sum, minimum, maximum, greatest common divisor (GCD), or other associative operations over a subarray.

Think of a segment tree as a [[Prefix Sum|prefix sum]] which supports updates efficiently.

### Complexity Analysis

| Operation | Time Complexity |
| --------- | --------------- |
| Build     | $O(n)$          |
| Query     | $O(\log n)$     |
| Update    | $O(\log n)$     |
| **Space** | $O(n)$          |

### Structure

A segment tree is typically represented as an array where:

- The root node is at index `1`.
- The left child of a node at index `i` is at `2i`, and the right child is at `2i + 1`.
- Leaf nodes (representing single elements) are found in the second half of the array.

### Operations

##### Construct

To construct a segment tree:
1. Start with an array of size `n`.
2. Recursively divide it into two halves until reaching single elements.
3. Merge results from child nodes into parent nodes.

![[segment_tree_construct.png|center]] { caption="Construction for A ＝ \[1, 3, -2, 8, -7\]." }

##### Query

A query retrieves the result of an operation (sum, min, max, etc.) over a given range `[L, R]`:
1. Start from the root.
2. If the current node's range is entirely inside `[L, R]`, return its value.
3. If it's outside, return a neutral value (e.g., `0` for sum, `∞` for min).
4. Otherwise, recursively check left and right children and combine results.

![[segment_tree_query.png|center]] { caption="Query the sum of range \[2, 4\] in A. Result is -2 + 1 ＝ -1." }

##### Update

If an element at index `i` is updated:
1. Locate the corresponding leaf node.
2. Update its value.
3. Propagate changes upwards to maintain correct values in parent nodes.

![[segment_tree_update.png|center]] { caption="Update A\[2\] ＝ 3. The change propagates up the tree." }

### Variants and Improvements

##### Lazy Propagation

Normally, updating a range in a segment tree (like adding a value to all elements in a range) requires changing many nodes, which can be slow. **Lazy propagation** makes it faster by delaying updates. Instead of updating all affected nodes right away, you:
1. Store the update (e.g., "add 5") in a "lazy" tag at the parent node of the range.
2. Only apply the update to child nodes when you need to access or query them later.

This "laziness" avoids unnecessary work, spreading the update cost over time. For example:
- If you update a range [2, 5], the root node gets a lazy tag.
- When querying [3, 4], you push the lazy update down to the relevant nodes, apply it, and clear the tag.

It’s like procrastinating chores until you _have_ to do them, saving effort when ranges are updated often but queried less frequently. This keeps both updates and queries efficient, typically $O(\log n)$ time each.

##### 2D Segment Tree

We can generalize the segment tree to higher dimensions:
1. Make a first-dimensional pseudo segment tree (X tree).
2. For each node in the X tree, connect it to a second-dimensional pseudo segment tree (Y tree).

The basic idea is *"tree within tree"*. Whereas a 1D segment tree is represented as an array of ints, a 2D segment tree is represented as an array of 1D segment trees (i.e. array of arrays of ints).

![[segment_tree_2d.png|center|pad10]]

In this example, the primary X tree segments by row and the secondary Y subtrees segment by column.

### Implementation

```jupyter
class SegmentTree:
    def __init__(self, arr):
        self.n = len(arr)
        # The segment tree size is ~4 times the size of the input array
        self.tree = [0] * (4 * self.n)
        self.build_tree(arr, 0, 0, self.n - 1)
    
    def build_tree(self, arr, node, start, end):
        """
        Build the segment tree recursively
        node: current node index in the segment tree
        start, end: range of the original array this node represents
        """
        # Leaf node case
        if start == end:
            self.tree[node] = arr[start]
            return
        
        # Find the middle point and recurse on left and right children
        mid = (start + end) // 2
        left_child = 2 * node + 1
        right_child = 2 * node + 2
        
        # Recursively build left and right subtrees
        self.build_tree(arr, left_child, start, mid)
        self.build_tree(arr, right_child, mid + 1, end)
        
        # Update current node with the sum of its children
        self.tree[node] = self.tree[left_child] + self.tree[right_child]
    
    def update(self, index, new_value, node=0, start=0, end=None):
        """
        Update the value at a specific index and propagate the changes
        index: index in the original array to update
        new_value: new value to set at the index
        node, start, end: current node and its range
        """
        if end is None:
            end = self.n - 1
            
        # If we've gone outside the range we're looking for
        if index < start or index > end:
            return
            
        # Leaf node case - we found the index to update
        if start == end:
            self.tree[node] = new_value
            return
        
        # Otherwise, recurse down to the right node
        mid = (start + end) // 2
        left_child = 2 * node + 1
        right_child = 2 * node + 2
        
        if index <= mid:
            # Update left child
            self.update(index, new_value, left_child, start, mid)
        else:
            # Update right child
            self.update(index, new_value, right_child, mid + 1, end)
            
        # Update the current node based on its children
        self.tree[node] = self.tree[left_child] + self.tree[right_child]
    
    def query_sum(self, query_start, query_end, node=0, start=0, end=None):
        """
        Query the sum within a range
        query_start, query_end: range to calculate sum for
        node, start, end: current node and its range
        """
        if end is None:
            end = self.n - 1
            
        # If the range is completely outside our segment
        if query_end < start or query_start > end:
            return 0
            
        # If the range completely encompasses our segment
        if query_start <= start and query_end >= end:
            return self.tree[node]
            
        # Otherwise, we need to look at both children
        mid = (start + end) // 2
        left_child = 2 * node + 1
        right_child = 2 * node + 2
        
        # Calculate sum by querying left and right children and adding results
        l = self.query_sum(query_start, query_end, left_child, start, mid)
        r = self.query_sum(query_start, query_end, right_child, mid + 1, end)
        return l + r


# Example usage
if __name__ == "__main__":
    # Initial array
    arr = [1, 3, 5, 7, 9, 11]
    print(f"Original array: {arr}")
    
    # Create the segment tree
    seg_tree = SegmentTree(arr)
    
    # Perform some range sum queries
    print("\nInitial range sum queries:")
    # Should be 1+3+5 = 9
    print(f"Sum of range [0, 2]: {seg_tree.query_sum(0, 2)}")
    # Should be 3+5+7+9 = 24
    print(f"Sum of range [1, 4]: {seg_tree.query_sum(1, 4)}")
    # Should be 1+3+5+7+9+11 = 36
    print(f"Sum of range [0, 5]: {seg_tree.query_sum(0, 5)}")
    
    # Update some values in the array
    print("\nUpdating array values:")
    seg_tree.update(1, 10)  # Change 3 to 10
    print(f"Updated index 1 from 3 to 10")
    seg_tree.update(4, 20)  # Change a 9 to 20
    print(f"Updated index 4 from 9 to 20")
    
    # Modified array would be [1, 10, 5, 7, 20, 11]
    
    # Perform the same range sum queries after updates
    print("\nRange sum queries after updates:")
    # Should be 1+10+5 = 16
    print(f"Sum of range [0, 2]: {seg_tree.query_sum(0, 2)}")
    # Should be 10+5+7+20 = 42
    print(f"Sum of range [1, 4]: {seg_tree.query_sum(1, 4)}")
    # Should be 1+10+5+7+20+11 = 54
    print(f"Sum of range [0, 5]: {seg_tree.query_sum(0, 5)}")
```

