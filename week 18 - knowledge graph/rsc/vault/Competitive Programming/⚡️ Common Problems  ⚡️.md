| Problem                                         | Solution                                             |
| ----------------------------------------------- | ---------------------------------------------------- |
| [[Find pairs of values]]                        | HashMap / HashSet                                    |
| [[Find fuzzy pairs of values]]                  | Sort + Two Pointer (Converging) **OR** Binary Search |
| [[Find duplicate values]]                       | HashSet / HashMap                                    |
| [[Find value / boundary / range]]               | Binary Search                                        |
| [[In-place array processing]]                   | Two Pointer (Read / Write)                           |
| [[Process a Tree or Graph]]                     | Recursive DFS **OR** BFS                             |
| [[Symmetry in Tree or Invert Tree]]             | Two pointer (Comparison) **OR** Double Recursion     |
| [[Linked List Middle or Nth From End or Cycle]] | Two Pointer (Slow / Fast)                            |
| [[Linked list delete node]]                     | Dummy node trick + pointer to curr and prev          |
| [[In-place array rotation]]                     | Index Mapping + Sliding Puzzle **OR** Reverse trick  |
| [[Find local extrema (peaks and troughs)]]      | Linear Scan **OR** Sliding Window                    |
| [[Detect integer overflow]]                     | Upcast **OR** Precondition Test                      |
| [[Dynamic Programming]]                         | OPT function + base case                             |
| [[Calculator Problems]]                         | Operator stack + operands stack                      |
| [[K-Sum Problem]]                               | Reduce to 2-Sum problem                              |
| [[Group By Operation]]                          | HashMap + Key Design                                 |
| [[Find Substring or Subarray]]                  | Sliding window                                       |

⭐️ **Tip**: Think out-loud while solving problems. This will help you talk during interviews (and make your problem-solving better overall).
⭐️ **Tip**: If stuck, consider the problem without constraints and then with constraints.
⭐️ **Tip**: If stuck, draw step-by-step diagrams where indices are labelled.
🪲 **Debug**: If wrong code but right idea, check all of the assumptions your code makes line by line to spot the bug.
✍️ **Memorize**: Binary Search, DFS, Partition, Merge, Backtracking
🤔 **Remember**: QuickSort, QuickSelect, BFS, Bottom-Up DP, Array Rotation
😡 **Avoid**: Over-communicating, Rambling, Not reading the problem, Starting the problem without thinking it out (CALM DOWN)

### 4 Step Plan to Solving an Interview Question

1.  Think
2.  Verify
3.  Write
4.  Revise

### 5 Problem Solving Tips for Cracking Coding Interview Questions

1.  Find a brute force solution
2.  Think of simpler version of the problem
3.  Think with simple examples —> Try noticing a pattern
4.  Use some visualization
5.  Test your solution on a few examples

### Recursion Tips

True for call stack and call tree:
- You can pass down a mutable, “collector” object reference via parameter (ArrayList reference)
- You can pass down data via parameters
- You can pass up data via returns

True exclusively for call stack:
- You can do processing going up & down the stack

True exclusively recursive call tree:
- You can compare returns from the right, left, and crossover (divide and conquer)
- You can do pre-order, in-order, & post-order processing