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

# Structs
In Rust, structures, or "structs", are custom data types that let you name and package together multiple related values that make up a meaningful group. Structs are similar to tuples, which were discussed earlier, but each piece of data in a struct can be named so it's clear what the values mean. This makes structs more flexible than tuples because you don't have to rely on the order of the data to specify or access the values of an instance.

### Defining and Instantiating Structs

To define a struct, you use the `struct` keyword and give the entire definition a name, followed by curly brackets containing the definitions of the fields. Fields are the names you give to the pieces of data so they can be accessed later.

```rust
struct User {
    username: String,
    email: String,
    sign_in_count: u64,
    active: bool,
}
```

To use a struct after you've defined it, you create an instance of that struct by specifying concrete values for each of the fields. You create an instance by stating the name of the struct, and then add curly brackets with `key: value` pairs, where the keys are the names of the fields and the values are the data you want to store in those fields.

```rust
let user1 = User {
    email: String::from("someone@example.com"),
    username: String::from("someusername123"),
    active: true,
    sign_in_count: 1,
};
```

### Types of Structs

Rust has three types of structs: *classic structs*, *tuple structs*, and *unit structs*.

1. **Classic Structs**:
   
These are the most commonly used structs. They are defined with named fields. The example of the `User` struct above is a classic struct.

2. **Tuple Structs**:

Tuple structs have fields, but the fields do not have names. They are useful when you want to give the whole tuple a name and add meaning to the tuple structure, but the individual fields do not require names.

```rust
struct Color(i32, i32, i32);
struct Point(i32, i32, i32);

let black = Color(0, 0, 0);
let origin = Point(0, 0, 0);
```

Even though `Color` and `Point` are both composed of three `i32` values, they are not of the same type because they are instances of different tuple structs.

3. **Unit Structs**:

Unit structs are fields-less structs, useful for situations where you need to implement a trait on some type but don't have any data to store in the type itself.

```rust
struct AlwaysEqual;

let subject = AlwaysEqual;
```

### Methods in Structs

Structs can also have methods defined within them with the `impl` keyword. Methods are functions that are defined within the context of a struct (or an enum or a trait object), and the first parameter is always `self`, which represents the instance of the struct the method is being called on.

```rust
struct Rectangle {
    width: u32,
    height: u32,
}

impl Rectangle {
    fn area(&self) -> u32 {
        self.width * self.height
    }
}

let rect = Rectangle { width: 30, height: 50 };
println!("The area of the rectangle is {} square pixels.", rect.area());
```

This functionality of methods allows you to specify the behavior that instances of your structs have.

### Conclusion

Rust structs allow you to create custom types that are meaningful for your domain. By using classic structs, tuple structs, or unit structs, you can structure data in a way that makes your code more organized and expressive. Moreover, the ability to define methods associated with your structs enables encapsulation and reusability of your logic.

# Enums

In Rust, enums (short for enumerations) are a feature that allows you to define a type by enumerating its possible variants. Enums are a powerful way to tie data and functionality together, representing data that could be one of several different variants. Each variant can optionally have data associated with it of any type.

### Basic Enum Syntax

To define an enum, use the `enum` keyword, followed by the name of the enum (which should be in CamelCase), and a set of curly braces containing the variants:

```rust
enum Direction {
    Up,
    Down,
    Left,
    Right,
}
```

This `Direction` enum represents a direction that can be one of four variants: `Up`, `Down`, `Left`, or `Right`.

### Using Enums

Enums are used by creating instances of their variants. For example:

```rust
let direction = Direction::Up;
```

### Enums with Data

Enums can also store data within their variants. The data can be of any type, including another enum, struct, or any of the primitive data types. This feature makes enums incredibly versatile in Rust.

```rust
enum Message {
    Quit,
    Move { x: i32, y: i32 },
    Write(String),
    ChangeColor(i32, i32, i32),
}
```

In this example, the `Message` enum has four variants with different types of associated data:

- `Quit` has no data associated with it.
- `Move` includes an anonymous struct with two `i32` fields.
- `Write` includes a single `String`.
- `ChangeColor` includes three `i32` values.

### Pattern Matching with Enums

One of the most powerful features of enums in Rust is their ability to be used with `match` expressions for pattern matching. This allows you to execute different code based on which variant of an enum a value is:

```rust
fn process_message(message: Message) {
    match message {
        Message::Quit => {
            println!("The Quit variant has no data to destructure.");
        },
        Message::Move { x, y } => {
            println!("Move in the X direction {} and in the Y direction {}", x, y);
        },
        Message::Write(text) => {
            println!("Text message: {}", text);
        },
        Message::ChangeColor(r, g, b) => {
            println!("Change the color to red {}, green {}, and blue {}", r, g, b);
        },
    }
}
```

### The Option Enum

Rust does not have `null`, but it has an enum called `Option<T>` that can encode the very concept of a value being present or absent. This enum is so useful that it's included in the prelude; you don't need to bring it into scope explicitly. It's variants are `Some(T)` and `None`:

```rust
enum Option<T> {
    Some(T),
    None,
}
```

For example, an `Option` can be used to indicate whether a value is found or not:

```rust
let some_number = Some(5);
let some_string = Some("a string");

let absent_number: Option<i32> = None;
```

### Conclusion

Enums in Rust provide a way to work with data that can be one of several variants, each of which can have different types and amounts of associated data. Through pattern matching with `match`, enums can be a powerful tool in the Rust programmer's toolkit, enabling expressive, type-safe, and error-resistant code. The `Option` enum is a prime example of how Rust uses enums to handle the possibility of absence in a way that's both safe and convenient.

# The Result Enum
In Rust, error handling is a fundamental aspect of writing robust programs. Rust groups errors into two major categories: recoverable and unrecoverable errors. For recoverable errors, Rust uses the `Result<T, E>` enum, and for unrecoverable errors, it uses the `panic!` macro that stops execution. This discussion focuses on the `Result<T, E>` type, a powerful tool for handling recoverable errors.

### The `Result` Enum

The `Result` enum is defined as follows:

```rust
enum Result<T, E> {
    Ok(T),
    Err(E),
}
```

- `Ok(T)`: An `Ok` variant to indicate the operation was successful and contains the resulting value of type `T`.
- `Err(E)`: An `Err` variant to indicate the operation failed and contains the error information of type `E`.

This design allows functions that might fail to return either a success containing a value or an error containing error information, making error handling explicit and integrated into the type system.

### Using `Result` in Functions

Here's an example of a function that returns a `Result`:

```rust
fn divide(numerator: f64, denominator: f64) -> Result<f64, String> {
    if denominator == 0.0 {
        Err(String::from("Error: Division by zero"))
    } else {
        Ok(numerator / denominator)
    }
}
```

This function performs division and returns a `Result`: `Ok` with the result if the division is successful, or an `Err` with an error message if attempted to divide by zero.

### Handling `Result` Values

To handle a `Result`, you can use pattern matching, `unwrap()`, `expect()`, among other methods.

#### Pattern Matching

```rust
match divide(10.0, 2.0) {
    Ok(result) => println!("Result: {}", result),
    Err(e) => println!("Error: {}", e),
}
```

#### Using `unwrap()` and `expect()`

- `.unwrap()`: Returns the value inside `Ok` if the result is `Ok`, otherwise calls the `panic!` macro.

```rust
let result = divide(10.0, 2.0).unwrap();
println!("Result: {}", result);
```

- `.expect()`: Similar to `unwrap()`, but allows specifying an error message if the result is `Err`.

```rust
let result = divide(10.0, 0.0).expect("Attempted to divide by zero");
```

### Error Propagation with `?` Operator

The `?` operator can be used to return the error early if a function returns `Err`, otherwise continues with the `Ok` value. This simplifies error propagation in functions that return `Result`.

```rust
fn perform_calculation() -> Result<f64, String> {
    let result = divide(10.0, 0.0)?;
    Ok(result)
}
```

If `divide` returns `Err`, `perform_calculation` will return early with that error. Otherwise, it proceeds with the `Ok` value.

### Conclusion

The `Result` type in Rust provides a robust and type-safe way to handle recoverable errors. By forcing explicit handling of errors, Rust encourages writing code that anticipates and properly manages potential failure points, contributing to the overall reliability and robustness of software written in Rust.


# Closures

Closures in Rust are anonymous functions you can save in a variable or pass as arguments to other functions. They're useful for short snippets of code that you might want to run multiple times or pass around. Rust's closures are flexible, allowing for capturing values from the scope in which they're defined.

### Basic Syntax

A closure is defined by a pair of vertical pipes (`|`), inside which you specify the parameters, followed by an expression:

```rust
let add_one = |x: i32| x + 1;
println!("The result is: {}", add_one(5));
```

### Capturing Environment

One of the most powerful features of closures is their ability to capture their environment, meaning they can use variables from the scope in which they were defined:

```rust
let y = 4;
let add_to_y = |x| x + y;
println!("The result is: {}", add_to_y(5));
```

Closures can capture variables by reference (`&T`), by mutable reference (`&mut T`), or by value (`T`), depending on how they are used.

### Type Inference

Rust infers the types of parameters and return values in closures, so you often don't need to annotate them:

```rust
let multiply = |x, y| x * y;
println!("The result is: {}", multiply(6, 7));
```

### Moving Captured Variables

If you want to force a closure to take ownership of the variables it uses from the environment, you can use the `move` keyword. This is particularly useful when passing a closure to a new thread:

```rust
let name = String::from("Alice");
let greet = move || println!("Hello, {}!", name);
// `name` is now owned by `greet`, and can't be used after this point

greet(); // Outputs: Hello, Alice!
```

### As Function Parameters

Closures can be passed as parameters to functions. When doing so, you can either specify a concrete type for the closure or use a generic with trait bounds:

```rust
fn apply<F>(f: F, a: i32, b: i32) -> i32
where
    F: Fn(i32, i32) -> i32,
{
    f(a, b)
}

let result = apply(|x, y| x + y, 5, 7);
println!("The result is: {}", result);
```

### Returning Closures

Because closures can have different sizes, you can't return them directly. However, you can return a `Box<dyn Fn()>`:

```rust
fn make_greeter(name: String) -> Box<dyn Fn()> {
    Box::new(move || println!("Hello, {}!", name))
}

let greeter = make_greeter(String::from("Bob"));
greeter(); // Outputs: Hello, Bob!
```

### Exercise

Create a closure that takes an integer and returns its square. Then, write a function that applies this closure to an array of integers, returning a new array of their squares.

#### Solution

```rust
fn main() {
    let square = |x: i32| x * x;
    let numbers = [1, 2, 3, 4];
    let squared_numbers: Vec<_> = numbers.iter().map(|&x| square(x)).collect();
    println!("{:?}", squared_numbers); // Outputs: [1, 4, 9, 16]
}

// As a function
fn apply_to_array<F>(arr: &[i32], operation: F) -> Vec<i32>
where
    F: Fn(i32) -> i32,
{
    arr.iter().map(|&x| operation(x)).collect()
}

fn main() {
    let numbers = [1, 2, 3, 4];
    let squared_numbers = apply_to_array(&numbers, |x| x * x);
    println!("{:?}", squared_numbers); // Outputs: [1, 4, 9, 16]
}
```

This demonstrates how closures can be used for inline operations and passed as arguments to functions, showcasing their flexibility and power in Rust.

# Iterators

Rust iterators are a powerful feature of the Rust programming language, providing a convenient way to work with sequences of values. Iterators allow you to perform various operations on these sequences, like transforming values, filtering them, or accumulating them into a single result, in a concise and expressive manner.

### Basic Concept

An iterator in Rust is any type that implements the `Iterator` trait. This trait requires the implementation of a method named `next` that, when called, returns an `Option` type: `Some(value)` if there is another value in the sequence and `None` when the iteration is over.

### Creating Iterators

Many collections in Rust, such as arrays, vectors, and hash maps, can be turned into iterators using methods like `.iter()`, `.iter_mut()`, or `.into_iter()`.

```rust
let vec = vec![1, 2, 3, 4];
let iter = vec.iter(); // Immutable reference to each element.
```

### Consuming Iterators

Iterators are "lazy" in Rust, meaning they have no effect until you consume them. Consuming an iterator is done by calling methods that use up the iterator to produce some result.

```rust
let sum: i32 = vec.iter().sum(); // Consumes the iterator to calculate the sum.
println!("The sum is: {}", sum);
```

### Iterator Adaptors

Iterator adaptors are methods that take an iterator and transform it into another iterator with a different output. These methods are also lazy and won't have an effect until the returned iterator is consumed. Common adaptors include `.map()`, `.filter()`, and `.enumerate()`.

```rust
let squared: Vec<_> = vec.iter().map(|x| x * x).collect(); // Squares each element.
println!("Squared: {:?}", squared);
```

### Example: Using Iterators

```rust
fn main() {
    let names = vec!["Alice", "Bob", "Charlie"];

    // Use `map` to transform each name into a greeting.
    let greetings: Vec<_> = names.iter()
                                 .map(|name| format!("Hello, {}!", name))
                                 .collect();

    // Use `for` loop to print each greeting.
    for greeting in greetings.iter() {
        println!("{}", greeting);
    }

    // Use `filter` to get names that start with 'B', and `count` to count them.
    let b_names_count = names.iter()
                             .filter(|name| name.starts_with('B'))
                             .count();

    println!("There are {} names starting with 'B'.", b_names_count);
}
```

### Chaining Iterators

One of the strengths of iterators is the ability to chain multiple adaptor methods together to perform complex transformations in a readable and efficient manner.

```rust
let numbers = vec![1, 2, 3, 4, 5, 6];

let even_squares: Vec<_> = numbers.iter()
                                  .map(|x| x * x)
                                  .filter(|x| x % 2 == 0)
                                  .collect();

println!("Even squares: {:?}", even_squares);
```

### Infinite Iterators

Rust also supports creating infinite iterators, which can be particularly useful when combined with other iterator adaptors.

```rust
let ones = std::iter::repeat(1);
let first_five_ones: Vec<_> = ones.take(5).collect(); // Take the first 5 elements.
println!("{:?}", first_five_ones);
```

### Conclusion

Iterators in Rust offer a powerful and flexible way to work with sequences of data. By understanding how to create, consume, and adapt iterators, you can write more concise, expressive, and efficient Rust code.

