# Hands-on: Building a Simple Rust Application

In this tutorial, we'll dive into building a simple Rust application from scratch. We'll create a command-line to-do list application that allows users to add tasks, list tasks, and mark tasks as done. This project will introduce you to basic Rust concepts, including ownership, the collections API, file I/O, and error handling.

## Prerequisites

Ensure you have Rust installed on your system. If you haven't set up Rust yet, refer to the tutorial on setting up a Rust development environment.

## Step 1: Setting Up Your Project

1. **Create a New Project**: Open a terminal and run the following command to create a new Rust project:

   ```sh
   cargo new rust_todo
   cd rust_todo
   ```

   This command creates a new directory named `rust_todo`, initializes a new Rust project inside it, and generates a `Cargo.toml` file for dependencies.

2. **Test the Setup**: Test that your project is set up correctly by running:

   ```sh
   cargo run
   ```

   You should see "Hello, world!" printed to the console, indicating that your project is ready.

## Step 2: Defining the Task Structure

Our to-do application needs to manage tasks. Let's define a `Task` struct in `src/main.rs`.

1. Open `src/main.rs` in your favorite editor.
2. Define a `Task` struct with a description and a done flag:

   ```rust
   struct Task {
       description: String,
       done: bool,
   }
   ```

## Step 3: Adding and Listing Tasks

### Adding Tasks

1. We'll use a `Vec<Task>` to store our tasks. Add a function `add_task` to add tasks to this vector:

   ```rust
   fn add_task(tasks: &mut Vec<Task>, description: String) {
       let task = Task {
           description,
           done: false,
       };
       tasks.push(task);
   }
   ```

### Listing Tasks

2. Implement a function `list_tasks` to print out the tasks:

   ```rust
   fn list_tasks(tasks: &[Task]) {
       for (index, task) in tasks.iter().enumerate() {
           println!("{}: {} [{}]", index + 1, task.description, if task.done { "x" } else { " " });
       }
   }
   ```

## Step 4: Handling User Input

We want to allow users to interact with our application through the command line. We'll parse user input to add tasks, list tasks, or mark a task as done.

1. **Parsing Command Line Arguments**: Use the `std::env::args` function to get user input:

   ```rust
   use std::env;

   fn main() {
       let args: Vec<String> = env::args().collect();
       let mut tasks = Vec::new();

       if args.len() > 1 {
           match args[1].as_str() {
               "add" => {
                   if args.len() < 3 {
                       println!("Usage: rust_todo add <task description>");
                   } else {
                       let description = args[2..].join(" ");
                       add_task(&mut tasks, description);
                   }
               }
               "list" => list_tasks(&tasks),
               _ => println!("Unknown command"),
           }
       } else {
           println!("Usage: rust_todo <command> [arguments]");
       }
   }
   ```

## Step 5: Saving and Loading Tasks

Let's persist our tasks to a file. We'll save our tasks in a file when the program exits and load them when it starts.

1. **Saving Tasks**: Implement a function to save tasks to a file:

   ```rust
   use std::fs::File;
   use std::io::{self, Write, BufRead, BufReader};

   fn save_tasks(tasks: &[Task]) -> io::Result<()> {
       let file = File::create("tasks.txt")?;
       let mut file = io::BufWriter::new(file);

       for task in tasks {
           writeln!(file, "{}\t{}", task.description, task.done)?;
       }

       Ok(())
   }
   ```

2. **Loading Tasks**: Implement a function to load tasks from a file:

   ```rust
   fn load_tasks() -> io::Result<Vec<Task>> {
       let file = File::open("tasks.txt")?;
       let file = BufReader::new(file);
       let mut tasks = Vec::new();

       for line in file.lines() {
           let line = line?;
           let parts: Vec<&str> = line.split('\t').collect();
           if parts.len() == 2 {
               let task = Task {
                   description: parts[0].to_string(),
                   done: parts[1] == "true",
               };
               tasks.push(task);
           }
       }

       Ok(tasks)
   }
   ```

3. Update your `main` function to call `load_tasks` at the start and `save_tasks` before exiting.

    ```rust
    fn main() {
        let args: Vec<String> = env::args().collect();
        let mut tasks = load_tasks().unwrap();

        if args.len() > 1 {
            match args[1].as_str() {
                "add" => {
                    if args.len() < 3 {
                        println!("Usage: rust_todo add <task description>");
                    } else {
                        let description = args[2..].join(" ");
                        add_task(&mut tasks, description);
                        save_tasks(&tasks);
                    }
                }
                "list" => list_tasks(&tasks),
                _ => println!("Unknown command"),
            }
        } else {
            println!("Usage: rust_todo <command> [arguments]");
        }
    }
    ```
## Step 6: Handling Marking Tasks as Done

Add functionality to mark tasks as done:

```rust
if args.len() > 2 && args[1] == "done" {
    let index: usize = args[2].parse().expect("Please enter a number") - 1;
    if index < tasks.len() {
        tasks[index].done = true;
        println!("Task {} marked as done.", index + 1);
    } else {
        println!("Invalid task number: {}", index + 1);
    }
}
```

## Conclusion

You've just built a simple yet functional to-do list application in Rust! This application introduces you to fundamental Rust concepts such as structs, vectors, handling user input, file I/O, and error handling. Rust's powerful type system and ownership model help ensure that our application manages resources safely and efficiently. Experiment with extending the application, such as adding the ability to delete tasks or improving the UI. Happy coding!