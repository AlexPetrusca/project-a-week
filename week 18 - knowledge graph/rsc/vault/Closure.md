A closure is a pairing of:
1.  A function, and
2.  References to that function's outer scope (lexical environment)

A lexical environment is part of every execution context (**stack frame**) and is a map of all identifiers available in the current scope (outer local variables, fields, etc.).

In [[JavaScript]], an inner function always has access to the variables of the outer function due to the closure nature of the code.

---

Counter implemented via a closure:

```javascript
// Returns a counter as a closure
function createCounter() {
    let counter = 0; // counter will be kept in memory even after leaving scope
    function increment() {
        // Inner function 'increment' references its outer scope. The local 
        // variable reference 'counter' binds to the corresponding variable
        // of the same name in the outer scope. A closure is created.
        return ++counter; 
    }
    return increment; // return inner function with closure
}

let add = createCounter(); // counter = 0
add(); // counter = 1
add(); // counter = 2
add(); // counter = 3
```