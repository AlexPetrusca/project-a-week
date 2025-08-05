A **union–find**, also known as a disjoint set, is a data structure that stores a collection of disjoint (non-overlapping) sets. It provides operations for adding new sets, merging sets, and finding the set an element belongs to.

Union-find is both asymptotically and practically optimal for certain tasks, such as computing the [[Minimum Spanning Tree]] of a weighted graph (see [[Kruskal's Algorithm]]) and identifying the [[Connected Component|connected components]] of a graph.

### Complexity Analysis

| Operation | Average        | Worst Case     |
| --------- | -------------- | -------------- |
| Search    | $Θ(\alpha(n))$   | $O(\alpha(n))$  |
| Insert    | $Θ(1)$          | $O(1)$         | 
| **Space** | $Θ(n)$          | $O(n)$         |

where $\alpha(n)$ is the [[Ackermann Function#Inverse|inverse Ackermann function]] of $n$. For most practical purposes, $\alpha(n)$ can be treated as a constant since it grows *extremely* slowly with respect to reasonable values of $n$.

### Structure

The main idea of a union-find data structure is to represent each disjoint set as a rooted tree:
- Every node maintains a link to its parent.
- A root node is the "representative" of the corresponding set.
    - The "representative" node is defined as the node whose parent is itself.

![[union_find_structure.png|500]]

### Operation

Union-find, as the name implies, supports two operations:
- **Union**: merge two sets.
- **Find**: find the representative of a set given a member of that set.

![[union_find_operations.png]]

### Implementation

A basic implementation, where each vertex is labeled from `0` to `n - 1` (inclusive), is as follows:

```cpp
vector<int> parent(n);

// make each node a representative of itself
void init() {
    for (int i = 0; i < n; i++) {
        parent[i] = i;
    }
}

// find the representative of a set
int find(int x) {
    while (x != parent[x]) {
        x = parent[x];
    }
    return x;
}

// merge two sets
void union(int x, int y) {
    parent[find(y)] = find(x);
}
```

##### Path Compression

Path compression makes the trees shallower every time $find()$ is called:
- Without it, the trees can become too deep in the worst case.
    - This slows down future operations.
- We don’t care how a tree looks like as long as the root stays the same.
    - After $find(x)$ returns the root, backtrack to $x$ and reroute all the links to the root.

```cpp
// recursive implementation
int find(int x) {
    if (x == parent[x]) return x;
    int root = find(parent[x]);
    parent[x] = root;
    return root;
}
```

```cpp
// iterative implementation
int find(int x) {
    int root = x;;
    while (root != parent[root]) {
        root = parent[root];
    }
    while (x != root) {
        int next = parent[x];
        parent[x] = root;
        x = next;
    }
    return root;
}
```

##### Path Halving

Same motivation as for path compression. Reduce the depth of the tree by replacing every other parent pointer with its grandparent (on the path from node to root):

```cpp
int find(int x) {
    while (x != parent[x]) {
        parent[x] = parent[parent[x]];
        x = parent[x];
    }
    return x;
}
```

Path halving retains the same worst-case complexity as path compression, but is more efficient in practice.

##### Union by Size and by Rank

In an efficient implementation, tree height is controlled using **union by size** or **union by rank**.
- Both of these require a node to store information besides just its parent pointer. 
    - This information is used to decide which root becomes the new parent. 
- Both strategies ensure that trees do not become too deep.

In union by size:
- Each representative stores the size of its set.
- On union, the set with more descendants becomes the representative for the merged set.

In union by tank:
- Each representative stores its _rank_, which is an upper bound for its height
- On union, the set with the higher rank becomes the representative for the merged set.