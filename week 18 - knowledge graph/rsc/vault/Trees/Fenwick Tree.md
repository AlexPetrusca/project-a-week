Integral Tree:
- Keep running sum (starting from root along path) for each node of tree
- Sum from node1 to node2: `integral.get(node2) - integral.get(node1) + node1.val`

%% 
#stub #todo
%%