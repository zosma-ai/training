
# Practical Asynchronous Programming in Rust: Building a Web Scraper

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