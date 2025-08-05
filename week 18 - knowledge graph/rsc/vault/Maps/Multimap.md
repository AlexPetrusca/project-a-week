A **multimap** is a data structure that allows one key to map to multiple values. 

![[cs_ds_multimap.png]]

Unlike a standard map or dictionary, where each key is associated with a single value, a multimap can associate a single key with multiple values, typically represented as a collection (e.g., a list, set, or other container).

### Language Support 

- **C++**: The Standard Template Library (STL) provides `std::multimap`.
- **Java**: The Google Guava library provides `Multimap` interfaces and implementations like `HashMultimap`.
- **Python**: While Python's `dict` isn't a multimap, you can emulate it using collections like `defaultdict(list)`.

### Applications

1. **Database-like structures**: Storing and querying multiple records for a single key.
2. **Graph algorithms**: Representing adjacency lists.
3. **Indexing**: Grouping items by common attributes or categories.