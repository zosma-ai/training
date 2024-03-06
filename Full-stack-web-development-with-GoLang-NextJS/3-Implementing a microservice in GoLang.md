Creating a microservice in Go (also known as Golang) is a great way to build scalable, high-performance backend services. This tutorial will guide you through the process of creating a simple microservice in Go that exposes a RESTful API. We'll build a service that manages a list of "tasks", allowing clients to add, retrieve, and delete tasks.

### Prerequisites

- Basic understanding of Go programming language.
- Go installed on your machine ([Download Go](https://golang.org/dl/)).
- Any text editor or IDE of your choice.

### Step 1: Setting Up Your Go Environment

First, ensure Go is properly installed by running `go version` in your terminal. You should see the Go version printed out.

### Step 2: Create Your Project

Create a new directory for your project and navigate into it:

```bash
mkdir go-microservice && cd go-microservice
```

Initialize a new Go module by running:

```bash
go mod init go-microservice
```

### Step 3: Write Your Microservice

Create a new file named `main.go`. This file will contain your service code.

```go
package main

import (
    "encoding/json"
    "log"
    "net/http"
    "github.com/gorilla/mux"
)

// Task represents a task with an ID and a Title
type Task struct {
    ID    string `json:"id"`
    Title string `json:"title"`
}

// tasks slice to seed task data.
var tasks = []Task{
    {ID: "1", Title: "Learn Go"},
    {ID: "2", Title: "Build a microservice"},
}

func getTasks(w http.ResponseWriter, r *http.Request) {
    w.Header().Set("Content-Type", "application/json")
    json.NewEncoder(w).Encode(tasks)
}

func createTask(w http.ResponseWriter, r *http.Request) {
    var task Task
    _ = json.NewDecoder(r.Body).Decode(&task)
    tasks = append(tasks, task)
    w.Header().Set("Content-Type", "application/json")
    json.NewEncoder(w).Encode(task)
}

func main() {
    router := mux.NewRouter()
    router.HandleFunc("/tasks", getTasks).Methods("GET")
    router.HandleFunc("/tasks", createTask).Methods("POST")
    log.Fatal(http.ListenAndServe(":8000", router))
}
```

### Step 4: Run Your Microservice

Before you can run your service, you need to install the `gorilla/mux` package, which makes handling routes easier:

```bash
go get -u github.com/gorilla/mux
```

Now, run your service:

```bash
go run main.go
```

Your microservice is now running on `http://localhost:8000`. You can test it using a tool like Postman or `curl`:

- To get tasks:

  ```bash
  curl http://localhost:8000/tasks
  ```

- To add a new task:

  ```bash
  curl -X POST -H "Content-Type: application/json" -d '{"id":"3","title":"New Task"}' http://localhost:8000/tasks
  ```

### Step 5: Expand Your Microservice

This basic tutorial covers setting up a simple microservice. From here, you can expand your service by:

- Adding a database to store tasks persistently.
- Implementing additional endpoints (e.g., updating and deleting tasks).
- Adding authentication to secure your API.
- Writing tests to ensure your code works as expected.

### Conclusion

You've now seen how to create a simple microservice in Go. Go's standard library provides a robust foundation for building web services, and external packages like `gorilla/mux` make routing more intuitive. As you become more comfortable with Go, you'll find it an excellent choice for building reliable, efficient microservices.