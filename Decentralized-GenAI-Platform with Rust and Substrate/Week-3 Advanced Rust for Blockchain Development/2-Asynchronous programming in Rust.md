
# Asynchronous Programming in Rust

Asynchronous programming in Rust is facilitated by the `async`, `await`, and `Future` traits, allowing you to write non-blocking code that's capable of performing multiple tasks concurrently. This tutorial covers the basics of async programming in Rust, including practical exercises with explanations.

## Prerequisites

- Rust 1.39 or newer.
- Basic understanding of Rust programming.
- The `tokio` runtime, as it's one of the most commonly used async runtimes in the Rust ecosystem.

### Getting Started

First, add `tokio` to your `Cargo.toml` to use it as the async runtime for our examples:

```toml
[dependencies]
tokio = { version = "1", features = ["full"] }
```

## Async/Await Basics

In Rust, `async` marks a block of code or function as asynchronous, turning it into a `Future`. This `Future` doesn't execute until you `.await` it.

Here's a simple async function:

```rust
async fn say_hello() {
    println!("Hello, async world!");
}
```

To run this function, you await it inside an async context:

```rust
#[tokio::main]
async fn main() {
    say_hello().await;
}
```

The #[tokio::main] attribute macro transforms the synchronous main function into an asynchronous entry point, enabling the use of async/await syntax and allowing the function to execute asynchronous code directly

### Exercise 1: Asynchronous Sleep

**Task**: Create an async function that simulates a long computation using `tokio::time::sleep`.

```rust
async fn long_computation() {
    println!("Computation started.");
    tokio::time::sleep(tokio::time::Duration::from_secs(2)).await;
    println!("Computation completed after 2 seconds.");
}
```

**Explanation**: This function first prints a message, then asynchronously waits for 2 seconds without blocking the entire program, and finally prints a completion message.

### Exercise 2: Concurrent Execution

**Task**: Run two async functions concurrently and observe which completes first.

```rust
async fn task_one() {
    tokio::time::sleep(tokio::time::Duration::from_secs(3)).await;
    println!("Task one completed.");
}

async fn task_two() {
    tokio::time::sleep(tokio::time::Duration::from_secs(1)).await;
    println!("Task two completed.");
}

#[tokio::main]
async fn main() {
    tokio::join!(task_one(), task_two());
}
```

**Explanation**: `tokio::join!` allows both tasks to run concurrently. Despite `task_one` being called first, `task_two` completes first due to its shorter sleep duration.

### Exercise 3: Fetching Data Asynchronously

**Task**: Use `reqwest` to perform an asynchronous HTTP GET request.

First, add `reqwest` to your `Cargo.toml`:

```toml
[dependencies]
reqwest = { version = "0.11", features = ["json"] }
```

Then, write the async function:

```rust
async fn fetch_url() -> Result<(), reqwest::Error> {
    let res = reqwest::get("https://www.rust-lang.org")
        .await?
        .text()
        .await?;

    println!("Fetched content: {}", &res[0..100]); // Print the first 100 chars
    Ok(())
}

#[tokio::main]
async fn main() {
    if let Err(e) = fetch_url().await {
        println!("Error fetching URL: {}", e);
    }
}
```

**Explanation**: This function asynchronously fetches the content of the Rust homepage and prints the first 100 characters. `reqwest::get().await` initiates the request and waits for the response non-blockingly, demonstrating the power of async programming in handling I/O-bound tasks.

## Conclusion

Asynchronous programming in Rust, with the help of the `tokio` runtime and the `async`/`await` syntax, provides a powerful paradigm for writing efficient and maintainable non-blocking code. Through these exercises, you've practiced writing async functions, executing tasks concurrently, and performing asynchronous network requests, which are foundational skills for modern Rust development.

## Building a Web Scraper

In this tutorial, we'll build a simple web scraper using asynchronous programming in Rust. This practical application will demonstrate how to perform concurrent HTTP requests to scrape data from multiple web pages simultaneously.

## Prerequisites

Ensure you have Rust and Cargo installed. We'll use the `tokio` runtime for asynchronous execution and the `reqwest` crate for making HTTP requests, along with `scraper` for parsing HTML.

Add the following dependencies to your `Cargo.toml`:

```toml
[dependencies]
tokio = { version = "1", features = ["full"] }
reqwest = "0.11"
scraper = "0.12"
```

## Setting Up the Asynchronous Runtime

We'll use Tokio as our asynchronous runtime. Start by creating an asynchronous entry point for our application:

```rust
#[tokio::main]
async fn main() {
    println!("Starting the web scraper...");
}
```

## Defining the Scraper

Our scraper will perform HTTP GET requests to fetch HTML content from a list of URLs and then parse specific data from the HTML. For simplicity, let's scrape the titles of Rust-related blog posts from the [Rust Blog](https://blog.rust-lang.org).

### Fetching HTML Content

First, define an asynchronous function to fetch the HTML content of a given URL:

```rust
async fn fetch_html(url: &str) -> Result<String, reqwest::Error> {
    let response_text = reqwest::get(url).await?.text().await?;
    Ok(response_text)
}
```

### Parsing HTML Content

Next, define a function to parse the HTML and extract the titles of blog posts. We'll look for `<a>` tags within elements with the class `post-title`.

```rust
fn parse_titles(html: &str) -> Vec<String> {
    let document = scraper::Html::parse_document(html);
    let selector = scraper::Selector::parse("article.post > h1 > a").unwrap();
    document.select(&selector).map(|element| element.inner_html()).collect()
}
```

## Scraping Multiple Pages Concurrently

Now, let's use these functions to scrape multiple pages concurrently. We'll define a list of URLs to scrape and then use `tokio::join!` to fetch and parse them in parallel.

```rust
#[tokio::main]
async fn main() {
    let urls = vec![
        "https://blog.rust-lang.org",
        // Add more URLs as needed
    ];

    let mut fetch_futures = vec![];

    for url in urls {
        let future = fetch_and_parse(url);
        fetch_futures.push(future);
    }

    let results = futures::future::join_all(fetch_futures).await;

    for (url, titles) in urls.iter().zip(results.iter()) {
        match titles {
            Ok(titles) => {
                println!("Titles from {}: {:?}", url, titles);
            }
            Err(e) => {
                eprintln!("Error fetching {}: {}", url, e);
            }
        }
    }
}

async fn fetch_and_parse(url: &str) -> Result<Vec<String>, reqwest::Error> {
    let html = fetch_html(url).await?;
    let titles = parse_titles(&html);
    Ok(titles)
}
```

### Explanation

- We define a vector `urls` containing the URLs we want to scrape.
- For each URL, we create an asynchronous future using `fetch_and_parse`, which fetches the HTML content and parses the titles.
- We use `futures::future::join_all` to wait for all fetching and parsing futures to complete, running them concurrently.
- Finally, we iterate over the results, printing the titles extracted from each URL or logging errors if any occur.

## Conclusion

This tutorial provided a hands-on example of asynchronous programming in Rust by building a simple web scraper. We demonstrated how to perform concurrent HTTP requests, parse HTML content, and process the results asynchronously. This practical application highlights the power and efficiency of Rust's async/await syntax and the `tokio` runtime for building concurrent applications. Asynchronous programming in Rust opens up a vast array of possibilities for developing high-performance, scalable applications that can handle I/O-bound tasks effectively.