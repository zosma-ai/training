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
        Direction

::Up => println!("Moving up!"),
        Direction::Down => println!("Moving down!"),
        Direction::Left => println!("Moving left!"),
        Direction::Right => println!("Moving right!"),
    }
}
```

## Conclusion

This tutorial covered the basics of Rust, including syntax, control flow, and common data structures. Rust's safety features and powerful type system, combined with these fundamentals, provide a strong foundation for developing efficient, memory-safe applications. As you become more familiar with Rust, you'll discover more advanced features and patterns that make Rust an excellent choice for systems programming and beyond.