Bit manipulation is the act of algorithmically manipulating bits. Computer programming tasks that require bit manipulation include **low-level device control**, **error detection algorithms**, **data compression**, **encryption algorithms**, and **code optimization**.

Bit manipulation code makes heavy use of bitwise operations (AND `&`, OR `|`, XOR `^`, NOT `~`) and bit shifts (LEFT `<<`, SIGNED RIGHT `>>`, UNSIGNED RIGHT `>>>`).

---

Check if number is power of two:

```c
(n & (n - 1)) == 0
```

---

Check if two numbers have different signs:

```c
(x ^ y) < 0
```

---

Multiply by two:

```c
x <<= 1
```

---

Divide by two:

```c
x >>= 1
```

---

Count set bits in number (Brian Kernighan):

```c
// repeatedly "subtract" the smallest power of two
// return the count of how many times we do so
int countSetBits(int x) {
    int count = 0;
    while (x > 0) {
        x &= (x - 1);
        count++;
    }
    return count;
}
```

---

Swap two numbers:

```c
int swap(int n1, int n2) {
    n1 = n1 ^ n2;
    n2 = n1 ^ n2;
    n1 = n1 ^ n2;
}
```