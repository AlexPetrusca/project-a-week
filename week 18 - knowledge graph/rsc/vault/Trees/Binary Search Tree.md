A **binary search tree (BST)** is a binary tree that stores and organizes data in a sorted order. The key property of a BST is that for each node $n$:

- All the elements in the left subtree of $n$ are less than $n$.
- All the elements in the right subtree of $n$ are greater than $n$.

![[bst_overview.png|center|500]]

### Complexity Analysis

| Operation | Average     | Worst Case |
| --------- | ----------- | ---------- |
| Search    | $O(\log n)$  | $O(n)$     | 
| Insert    | $O(\log n)$  | $O(n)$     |
| Delete    | $O(\log n)$  | $O(n)$     |
| **Space** | $O(n)$      | $O(n)$     |

### Structure

A binary search tree consists of nodes where each node contains:

- **Key**: A value that helps maintain the order property.
- **Left Child**: A reference to a left node that holds a value less than the current node.
- **Right Child**: A reference to a right node that holds a value greater than the current node.

### Operations

A binary search tree supports several basic operations:

##### Search

To search for an element $e$ in a BST:

1. Start from the root.
2. If $e$ is equal to the key of the current node, return the node.
3. If $e$ is less than the key of the current node, move to the left child.
4. If $e$ is greater than the key of the current node, move to the right child.
5. Repeat the process until the element is found or a leaf node is reached.

##### Insert

To insert a new element $e$ into a BST:

1. Start from the root.
2. If $e$ is less than the current node, move to the left child; if $e$ is greater, move to the right child.
3. Continue this process recursively until you find a null spot (i.e., a leaf position).
4. Insert the new node as a left or right child, depending on whether $e$ is smaller or larger than the parent node.d.

##### Delete

To delete a node $n$ from the BST:

1. If the node to be deleted has no children (a leaf), simply remove it.
2. If the node has one child, replace it with its child.
3. If the node has two children, find the minimum value node in the right subtree (or maximum in the left subtree), replace the node to be deleted with that node, and remove the original node.

### Balancing

A key limitation of the basic BST is that its operations can degrade to linear time if the tree becomes unbalanced (e.g., a degenerate tree where each node has only one child). 

To maintain efficient operations, balanced trees like **AVL trees** or **[[Red-Black Tree|Red-Black trees]]** are often used in practice. These trees ensure that the height remains logarithmic by performing rotations during insertions and deletions.

### Applications

- **Searchable Databases**: Fast lookups and range queries.
- **Autocompletion**: Efficient search for prefixes in large datasets.
- **Symbol Tables**: Used in compilers for managing variables and functions.
- **Set Operations**: Union, intersection, and difference operations on sets can be implemented efficiently using BSTs.