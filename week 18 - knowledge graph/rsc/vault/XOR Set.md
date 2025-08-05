Basically, its a strange sort of set, which we can only read when it contains exactly 1 element:
- **Initialize**: keep track of a running xor and a size.
- **Insert**: increment size and add to running xor.
- **Delete**: decrement size and add to running xor.
- **Read**: return running xor (this value is gibberish unless the set's size is exactly 1).

XOR sets are great for telling us when a node in a graph is a leaf node, after a certain number of edge insertions and deletions are performed.

XOR sets are useful for efficiently implementing [[topological sort]] using BFS.

### Example

https://leetcode.com/problems/minimum-height-trees/

```cpp
class XORSet {
    size_t sz;
    int xr;
public:
    XORSet(): sz(0), xr(0) { }
    void insert(const int val) {
        ++sz;
        xr ^= val;
    }
    void erase(const int val) {
        --sz;
        xr ^= val;
    }
    size_t size() const { return sz; }
    int value() const { return xr; }
};

class Solution {
public:
    vector<int> findMinHeightTrees(int n, vector<vector<int>>& edges) {
        // represent the graph as a list of XOR sets.
        std::vector<XORSet> conns(n);
        for (std::vector<int> edge : edges) {
            conns[edge[0]].insert(edge[1]);
            conns[edge[1]].insert(edge[0]);
        }

        // queue will only ever contain leaf nodes.
        std::queue<int> que;
        for (int i = 0; i < conns.size(); i++) {
            if (conns[i].size() <= 1) {
                que.push(i);
            }
        }

        // progressively remove layers of leafs until there are two
        // or less nodes remaining
        while (n > 2) {
            int size = que.size();
            for (int i = 0; i < size; i++) {
                int from = que.front();
                // we can read the value for any node on the queue
                // from our set, since we know that it must be a leaf.
                int to = conns[from].value();
                conns[to].erase(from);
                // if the `to` node is now a leaf, add it to the queue
                if (conns[to].size() == 1) {
                    que.push(to); 
                }
                que.pop();
            }
            n -= size;
        }

        // output the nodes remaining on the queue
        std::vector<int> minTrees;
        while (!que.empty()) {
            minTrees.push_back(que.front());
            que.pop();
        }
        return minTrees;
    }
};
```

%% 
#stub 
%%