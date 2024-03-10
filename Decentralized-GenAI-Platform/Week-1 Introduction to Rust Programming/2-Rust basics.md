# Rust Basics: Syntax, Control Flow, and Data Structures

Welcome to this detailed tutorial on Rust, a systems programming language that focuses on speed, memory safety, and parallelism. This guide will introduce you to the basics of Rust, including its syntax, control flow mechanisms, and common data structures.

## 1. Rust Syntax

Rust's syntax is similar to C++ but with some differences that enforce safety and readability.

### Hello, World!

Every Rust program starts with a `main` function:

```rust
fn main() {
    println!("Hello, World!");
}
```

- `fn` declares a new function.
- `main` is the entry point of the program.
- `println!` is a macro (note the `!`) that prints to the console.

### Variables and Mutability

By default, variables in Rust are immutable (cannot be changed after their initial assignment). To declare a mutable variable, use `mut`:

```rust
let x = 5; // immutable
let mut y = 5; // mutable

y = 10; // this is okay
// x = 10; // this would cause a compile-time error
```

### Data Types

Rust is statically typed, requiring all types to be known at compile-time. Common types include:

- **Integer**: `i32`, `u32`, `i64`, `u64`, etc.
- **Floating Point**: `f32`, `f64`
- **Boolean**: `bool`
- **Character**: `char`
- **String**: `String` for a dynamically-sized string

### Functions

Functions are declared with `fn`, followed by the function name, parameters, and body:

```rust
fn add(x: i32, y: i32) -> i32 {
    x + y // Rust returns the last expression without needing `return`.
}
```

## 2. Control Flow

Controlling the flow of your Rust programs can be done through various structures.

### if Expressions

Rust’s `if` expressions are used for branching:

```rust
let number = 6;

if number % 2 == 0 {
    println!("The number is even.");
} else {
    println!("The number is odd.");
}
```

### Looping with loop, while, and for

- **`loop`** repeats a block of code forever (or until you break out of it):

  ```rust
  loop {
      println!("again!"); // This will print "again!" forever unless stopped.
      break; // Exiting the loop.
  }
  ```

- **`while`** repeats a block of code as long as a condition is true:

  ```rust
  let mut number = 3;
  
  while number != 0 {
      println!("{}!", number);
      number -= 1;
  }
  println!("LIFTOFF!!!");
  ```

- **`for`** iterates over elements in a collection, such as an array:

  ```rust
  let a = [10, 20, 30, 40, 50];
  
  for element in a.iter() {
      println!("the value is: {}", element);
  }
  ```

## 3. Data Structures

Rust offers several key data structures for organizing and storing data.

### Tuples

Tuples are collections of values of different types:

```rust
let tup: (i32, f64, u8) = (500, 6.4, 1);

let (x, y, z) = tup; // Destructuring the tuple

println!("The value of y is: {}", y);
```

### Arrays

Arrays in Rust have a fixed size and elements of the same type:

```rust
let a = [1, 2, 3, 4, 5];

let first = a[0]; // Accessing array elements
let second = a[1];
```

### Vectors

Vectors are similar to arrays but are dynamic in size:

```rust
let mut vec = vec![1, 2, 3]; // vec! macro to create a new vector

vec.push(4); // Adding an element to the vector
vec.pop(); // Removing the last element
```

### Structs

Structs let you create custom data types:

```rust
struct User {
    username: String,
    email: String,
    sign_in_count: u64,
    active: bool,
}

let user1 = User {
    email: String::from("someone@example.com"),
    username: String::from("someusername123"),
    active: true,
    sign_in_count: 1,
};
```

### Enums

Enums allow you to define a type by enumerating its possible variants:

```rust
enum Direction {
    Up,
    Down,
    Left,
    Right,
}

fn move_direction(dir: Direction) {
    match dir {
        Direction::Up => println!("Moving up!"),
        Direction::Down => println!("Moving down!"),
        Direction::Left => println!("Moving left!"),
        Direction::Right => println!("Moving right!"),
    }
}
```

## Conclusion

This tutorial covered the basics of Rust, including syntax, control flow, and common data structures. Rust's safety features and powerful type system, combined with these fundamentals, provide a strong foundation for developing efficient, memory-safe applications. As you become more familiar with Rust, you'll discover more advanced features and patterns that make Rust an excellent choice for systems programming and beyond.

# Rust Vectors

Rust's vectors (`Vec<T>`) are one of the most versatile and commonly used collections. They allow you to store more than one value in a single data structure that puts all values next to each other in memory. This tutorial explores advanced concepts and operations you can perform with vectors in Rust.

## 1. Creating and Initializing Vectors

Beyond simple initialization, Rust provides flexible ways to create and initialize vectors:

### With Capacity

Pre-allocating space can improve performance when you know how many elements the vector will hold.

```rust
let mut vec: Vec<i32> = Vec::with_capacity(10);
```

### From a Range

Use a range and `collect` to quickly create vectors.

```rust
let vec: Vec<i32> = (1..=5).collect();
```

### Using `vec!` Macro with Repetitions

Initialize a vector with repeated values.

```rust
let vec = vec![0; 10]; // Ten zeros
```

## 2. Manipulating Vectors

Vectors are dynamic, allowing for runtime modification.

### Pushing and Popping

Add or remove elements at the vector's end.

```rust
let mut vec = vec![1, 2, 3];
vec.push(4);
vec.pop(); // Returns Some(4)
```

### Inserting and Removing

Insert or remove elements at any position, shifting subsequent elements.

```rust
let mut vec = vec![1, 2, 4];
vec.insert(2, 3); // vec is now [1, 2, 3, 4]
vec.remove(1); // Removes 2, vec is now [1, 3, 4]
```

## 3. Iterating Over Vectors

Rust provides several methods to iterate over vectors, allowing for flexible data manipulation.

### Immutable Iteration

Iterate over each element without modifying them.

```rust
for val in &vec {
    println!("{}", val);
}
```

### Mutable Iteration

Modify each element in place.

```rust
for val in &mut vec {
    *val += 1;
}
```

### Consuming Iteration with `into_iter`

Take ownership of the vector and its elements.

```rust
for val in vec.into_iter() {
    // Do something with val
}
```

## 4. Slicing Vectors

Slices provide a view into a portion of a vector without copying its contents.

```rust
let vec = vec![1, 2, 3, 4, 5];
let slice = &vec[1..4]; // Slice of &[2, 3, 4]
```

## 5. Using `VecDeque` for Efficient Front Operations

While vectors are efficient for back operations, consider `VecDeque` for frequent front operations.

```rust
use std::collections::VecDeque;

let mut deque: VecDeque<i32> = VecDeque::new();
deque.push_front(1);
deque.pop_front();
```

## 6. Retaining Elements

Use `retain` to keep elements that match a predicate, removing others in place.

```rust
let mut vec = vec![1, 2, 3, 4];
vec.retain(|&x| x % 2 == 0); // Keeps only even numbers
```

## 7. Drain Filter

`drain_filter` allows for filtering and removing elements simultaneously, yielding removed items.

```rust
let mut vec = vec![1, 2, 3, 4];
let evens: Vec<_> = vec.drain_filter(|x| *x % 2 == 0).collect();
```

## 8. Splitting and Joining

Split a vector into two at a specific index with `split_off`, or join two vectors with `append` or `extend`.

```rust
let mut vec = vec![1, 2, 3, 4, 5];
let higher = vec.split_off(2); // vec is now [1, 2], higher is [3, 4, 5]

vec.extend(higher);
```

## 9. Using `BinaryHeap` for Priority Queue Functionality

For use cases requiring sorting or priority queues, `BinaryHeap` offers an alternative to vectors.

```rust
use std::collections::BinaryHeap;

let mut heap = BinaryHeap::new();
heap.push(4);
heap.push(2);
```

## Conclusion

Rust vectors offer a combination of flexibility, efficiency, and convenience for handling collections of data. Beyond basic operations, understanding advanced techniques and related collections can significantly enhance your Rust development skills, allowing you to choose the most appropriate data structure for your specific needs.