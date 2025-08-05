Separate vertex labels from graph representation ⭐️
- To track vertex labels:
    - Keep an indexed list $V$ of all vertex labels
- To represent graph:
    - reference vertices by their index into $V$ (not by their label)

### Edge List

Represent the graph as a list of all edges:
- Each edge is represented as a tuple --> $(startVertex, endVertex)$
- Can be easily extended to represent weighted graph --> $(startVertex, endVertex, weight)$

### Adjacency Matrix

Represent the graph as a $v$ x $v$ matrix:
- Each entry $boolean[i][j]$ represents whether or not there is an edge between the $i$-th vertex and $j$-th vertex
- Can be easily extended to represent weighted graph --> $int[i][j]$ represents weight of edge (0 if no edge)
- Good for dense graphs

### Adjacency List

Represent the graph as a list of $v$ lists:
- Each entry $j$ in $list[i]$ represents a vertex that is reachable from the $i$-th vertex
- Good for sparse graphs

### Adjacency Sets

Represent the graph as a list of $v$ sets:
- Same idea as adjacency list
