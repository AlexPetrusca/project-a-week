A **hash map**, or *hash table*, is a data structure that implements an associative array (a.k.a. dictionary) which is an abstract data type that maps keys to values.

![[hash_table_overview.png]]

A hash table uses a [[Hash Function]] to compute an index, also called a hash code, into an array of buckets or slots, from which the desired value can be found. During lookup, the key is hashed and the resulting hash indicates where the corresponding value is stored.

### Complexity Analysis

| Operation | Average | Worst Case | 
| --------- | ------- | ---------- |
| Search    | $Θ(1)$  | $O(n)$      |
| Insert    | $Θ(1)$  | $O(n)$      |
| Delete    | $Θ(1)$  | $O(n)$      |
| **Space** | $Θ(n)$  | $O(n)$      |

### Collision Resolution

During insertion of a record, if the hash code indexes a full slot, some kind of collision resolution is required:
- **Chained Hashing**
    - Each slot is the head of a linked list or chain.
    - Items that collide at the slot are added to the chain. 
    - During lookup, the corresponding chain is searched linearly until either:
        - The item is located (item found)
        - The entire chain has been searched (item not found)
- **Open Address Hashing**
    - The table is probed starting from the occupied slot until an open slot is located or the entire table is probed (overflow)
    - Common probing procedures include:
        - *Linear Probing*: The interval between probes is fixed (usually 1)
        - *Quadratic Probing*: The interval between probes is increased by adding the successive outputs of a quadratic polynomial to the value given by the original hash computation.
        - *Double Hashing*: The interval between probes is computed by a secondary hash function.
    - During lookup, the table is probed using the same procedure until either:
        - The item is located (item found)
        - An open slot is found or the entire table has been searched (item not found)

### Dynamic resizing

Repeated insertions cause the number of entries in a hash table to grow, which consequently increases the load factor; to maintain the amortized $O(1)$ performance of the lookup and insertion operations, a hash table is dynamically resized.

Generally, a new hash table with a size double that of the original hash table gets allocated privately and every item in the original hash table gets moved to the newly allocated one by computing the hash values of the items followed by the insertion operation. Rehashing is simple, but computationally expensive.

### Hash Function

The better the hashing algorithm used to compute hash codes, the better the performance of a hash table.

In Java, for a class User:

```java
public class User {
    private long id;
    private String name;
    private String email;
}
```

A “standard” hash function implementation would look something like:

```java
@Override
public int hashCode() {
    int hash = (int) (id ^ (id >>> 32));
    hash = 31 * hash + (name == null ? 0 : name.hashCode());
    hash = 31 * hash + (email == null ? 0 : email.hashCode());
    return hash;
}
```

### Applications

In practice, hash tables are used to implement the following:
- **Dictionary**: data structure that stores a collection of (key, value) pairs.
- **Cache**: auxiliary data table that is used to speed up the access to data.
- **Set**: data structure that stores unique values, without any particular order.
- **Transposition Table**: cache of previously seen positions and associated evaluations in a game tree, used in tree search algorithms, such as minimax, to speed up execution.