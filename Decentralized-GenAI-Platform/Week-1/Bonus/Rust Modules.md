Developing a Rust module involves organizing code into separate namespaces, improving project maintainability, readability, and reusability. This tutorial will guide you through creating and using modules in Rust, highlighting the use of `pub` for making items public, nested modules, and splitting modules into separate files.

### Understanding Rust Modules

Modules in Rust are a way to organize code into namespaces, allowing you to group related functionality and control the visibility (public/private) of items (functions, structs, enums, etc.).

### Step 1: Creating a Module

In Rust, you define a module with the `mod` keyword. Let's start by creating a simple module named `greetings`:

```rust
mod greetings {
    pub fn hello() {
        println!("Hello, world!");
    }
}
```

### Step 2: Making Module Items Public

Notice the `pub` keyword before the `fn` keyword. This makes the `hello` function public, meaning it can be accessed from outside the `greetings` module. Without `pub`, the function would be private to `greetings`.

### Step 3: Using the Module

To use the `hello` function from the `greetings` module, you need to qualify it with the module's name:

```rust
fn main() {
    greetings::hello();
}
```

### Step 4: Nested Modules

Modules can be nested within other modules. Let's extend the `greetings` module with a nested module named `farewell`:

```rust
mod greetings {
    pub fn hello() {
        println!("Hello, world!");
    }

    pub mod farewell {
        pub fn goodbye() {
            println!("Goodbye, world!");
        }
    }
}
```

To call the `goodbye` function, you need to refer to it through both the parent and child module names:

```rust
fn main() {
    greetings::farewell::goodbye();
}
```

### Step 5: Splitting Modules into Separate Files

As your module grows, you may want to split it into separate files for better organization. Let's split the `greetings` module:

1. **Create a file named `greetings.rs`** or a directory named `greetings` with a file named `mod.rs` inside it. Both approaches are valid, but for this tutorial, we'll create a `greetings.rs` file.

2. **Move the `greetings` module code to `greetings.rs`**, removing the `mod greetings { ... }` wrapper:

```rust
// In greetings.rs

pub fn hello() {
    println!("Hello, world!");
}

pub mod farewell {
    pub fn goodbye() {
        println!("Goodbye, world!");
    }
}
```

3. **Declare the `greetings` module in your main file** (`main.rs` or `lib.rs`) using `mod greetings;`. This tells Rust to load the module from `greetings.rs`:

```rust
// In main.rs or lib.rs

mod greetings;

fn main() {
    greetings::hello();
    greetings::farewell::goodbye();
}
```

### Step 6: Public and Private Access

Understanding public (`pub`) and private access is crucial in modules. By default, all items (functions, structs, modules, etc.) are private and only accessible within their current module or child modules. Using `pub` makes them accessible from outside the module.

### Conclusion

Rust modules are a powerful feature for organizing code into namespaces, managing visibility, and enhancing code reusability. By following this tutorial, you should now understand how to create modules, use the `pub` keyword to control access, nest modules, and split modules into separate files for better project organization. This modular approach is essential for building larger, maintainable, and scalable Rust applications.