```rust
use std::sync::{Arc, Mutex};
use std::thread;

fn main()

 {
    let counter = Arc::new(Mutex::new(0));
    let mut handles = vec![];

    for _ in 0..10 {
        let counter = Arc::clone(&counter);
        let handle = thread::spawn(move || {
            let mut num = counter.lock().unwrap();
            *num += 1;
        });
        handles.push(handle);
    }

    for handle in handles {
        handle.join().unwrap();
    }

    println!("Result: {}", *counter.lock().unwrap());
}
```
This Rust code demonstrates a basic example of multi-threading where a group of threads increment a shared counter. It utilizes `std::sync::{Arc, Mutex}` for thread-safe shared state management and `std::thread` for spawning threads. Let's break down the code piece by piece:

### Understanding the Key Components

1. **`Arc`**: The `Arc` type is a thread-safe reference-counting pointer. 'Arc' stands for 'Atomic Reference Counting'. It allows multiple threads to own a piece of data and ensures that the data outlives all its owners by keeping track of the number of active references to it.

2. **`Mutex`**: The `Mutex` (mutual exclusion) is a synchronization primitive that can be used to protect shared data from being simultaneously accessed by multiple threads. A mutex allows only one thread to access the data at any given time. To access the data, a thread must first lock the mutex. When it has finished working with the data, the thread must unlock the mutex.

3. **`thread::spawn`**: This function spawns a new thread and returns a handle to it. The new thread will execute the closure given as an argument to `spawn`.

### Code Walkthrough

- **`let counter = Arc::new(Mutex::new(0));`**: A `Mutex` is created to protect the shared counter (initialized to 0), and this mutex is wrapped inside an `Arc` to allow safe sharing across threads.

- **`let mut handles = vec![];`**: A vector is initialized to hold the handles for all spawned threads. This allows the main thread to wait for all spawned threads to complete.

- **Loop to Spawn Threads**:
    - The loop iterates 10 times, each iteration spawning a new thread.
    - **`let counter = Arc::clone(&counter);`**: For each iteration, the `Arc` containing the `Mutex` is cloned. The `Arc::clone` method increments the reference count of the `Arc` but does not clone the underlying data (the `Mutex` and the counter). This cloned `Arc` is moved into the closure of the spawned thread, giving the thread shared ownership of the `Mutex`.
    - **`thread::spawn(move || { ... })`**: A new thread is spawned. The `move` keyword moves the cloned `Arc` into the closure, ensuring the closure owns the `Arc`, and by extension, can access the `Mutex`-protected counter.
    - **`let mut num = counter.lock().unwrap();`**: Inside each thread, the mutex is locked to safely access the shared counter. The `lock` method returns a `MutexGuard`, which is unwrapped to panic in case of any errors (not typically recommended for production code). This guard provides mutable access to the underlying data.
    - **`*num += 1;`**: The dereference operator (`*`) is used to access the value inside the `MutexGuard`, and the counter is incremented.
    - **`handles.push(handle);`**: The handle to the spawned thread is stored in the vector of handles.

- **Waiting for All Threads to Complete**:
    - After spawning all threads, the main thread iterates over the vector of handles, calling `join` on each. The `join` method waits for its thread to finish, ensuring that the main thread waits for all spawned threads to complete before proceeding.

- **`println!("Result: {}", *counter.lock().unwrap());`**: Finally, the main thread locks the mutex one last time to safely access and print the final value of the counter.

### Conclusion

This example showcases how to safely share mutable state (`counter`) across multiple threads using `Arc` for shared ownership and `Mutex` for exclusive access. It demonstrates basic multi-threaded programming patterns in Rust, including spawning threads, sharing state, and synchronizing access to shared data.