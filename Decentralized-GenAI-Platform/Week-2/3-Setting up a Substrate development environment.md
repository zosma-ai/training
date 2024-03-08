# Setting up a Substrate Development Environment

Substrate is a powerful framework for building blockchain applications. This tutorial will guide you through setting up a Substrate development environment, enabling you to start building your own blockchains. Substrate is developed by Parity Technologies and is the foundation of Polkadot and many other blockchain projects.

## Prerequisites

- Basic understanding of blockchain technology.
- Familiarity with Rust programming language.
- A computer with macOS, Linux, or Windows (WSL for Windows users).

## Step 1: Install Rust and Required Dependencies

Substrate is built with the Rust programming language, so you'll need Rust and the `cargo` package manager installed.

### On macOS and Linux:

1. Open a terminal window.
2. Install Rust using `rustup`, which is Rust's version management and installation tool. Execute the following command:

   ```sh
   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
   ```

3. Follow the on-screen instructions. Typically, the default installation options are recommended.
4. After installation, configure your current shell to load `rustc` and `cargo` by running:

   ```sh
   source $HOME/.cargo/env
   ```

5. Install the necessary dependencies:

   - **Linux**:
     ```sh
     sudo apt update && sudo apt install -y git clang curl libssl-dev llvm libudev-dev
     ```
   
   - **macOS** (using [Homebrew](https://brew.sh/)):
     ```sh
     brew update
     brew install openssl cmake llvm
     ```

### On Windows (WSL):

1. It's recommended to use Windows Subsystem for Linux (WSL) for Substrate development on Windows. [Follow Microsoft's guide to install WSL](https://docs.microsoft.com/en-us/windows/wsl/install).
2. Once WSL is set up, open the WSL terminal and follow the macOS/Linux instructions above to install Rust and the necessary dependencies.

## Step 2: Install Substrate

With Rust installed, you can now install Substrate's prerequisites.

1. Set up the default Rust toolchain to the latest stable version:

   ```sh
   rustup default stable
   rustup update
   ```

2. Add the nightly toolchain and the `wasm` target. Substrate uses WebAssembly (WASM) for smart contract and runtime development:

   ```sh
   rustup update nightly
   rustup target add wasm32-unknown-unknown --toolchain nightly
   ```

## Step 3: Create a New Substrate Project

Substrate provides a Node Template that serves as a good starting point for building a new blockchain.

1. Clone the Substrate Node Template from its repository:

   ```sh
   git clone https://github.com/substrate-developer-hub/substrate-node-template
   ```

2. Change into the new directory:

   ```sh
   cd substrate-node-template
   ```

3. Compile the project to ensure everything is set up correctly:

   ```sh
   cargo build --release
   ```

   Note: This step might take a while as it compiles all necessary dependencies.

## Step 4: Run Your Substrate Node

After compiling the node template, you can start your Substrate node.

1. Execute the following command within the `substrate-node-template` directory:

   ```sh
   ./target/release/node-template --dev
   ```

2. You should see the node starting and producing blocks. The `--dev` flag runs your node in development mode, using a pre-defined development chain specification.

Congratulations! You have successfully set up your Substrate development environment and are running your first Substrate node.

## Step 5: Exploring Further

With your environment set up and your node running, you can now explore further:

- **Learn More About Substrate**: The [official Substrate documentation](https://docs.substrate.io/) provides in-depth information about developing with Substrate.
- **Smart Contracts with Ink!**: Explore [Ink!](https://docs.substrate.io/v3/runtime/contracts/ink/), Substrate's Rust-based eDSL for writing smart contracts.
- **Substrate Front-End Template**: Clone and run the [Substrate Front-End Template](https://github.com/substrate-developer-hub/substrate-front-end-template) to interact with your node through a user interface.
- **Join the Community**: The Substrate [Technical Chat](https://substrate.dev/en/seminar) and [Stack Exchange](https://substrate.stackexchange.com/) are great places to ask questions and learn from other developers.

Happy building on Substrate!