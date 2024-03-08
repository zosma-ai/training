# Hands-on: Creating a Simple Substrate-based Blockchain

In this tutorial, we'll walk through the process of creating a simple Substrate-based blockchain from scratch. We will use the Substrate Node Template as our starting point, which provides a basic blockchain framework, allowing us to focus on customizing our chain's logic.

## Prerequisites

- Completion of the Substrate development environment setup tutorial.
- Basic understanding of Rust programming language.
- Familiarity with blockchain concepts.

## Step 1: Set Up the Substrate Node Template

1. **Clone the Substrate Node Template**: If you haven't already, clone the Substrate Node Template repository with the following command:

   ```sh
   git clone https://github.com/substrate-developer-hub/substrate-node-template
   ```

2. **Navigate to the Template Directory**:

   ```sh
   cd substrate-node-template
   ```

3. **Compile the Node Template**:

   ```sh
   cargo build --release
   ```

   This step compiles the node template and may take some time to complete.

## Step 2: Creating a Custom Pallet

Pallets are modular libraries that encapsulate functionality for the blockchain. We will create a simple pallet that allows users to store and retrieve values.

1. **Generate a New Pallet Template**:

   Within the node template directory, navigate to the pallets directory:

   ```sh
   cd pallets
   ```

   Use the Substrate Pallet Generator to create a new pallet named `simple_storage`:

   ```sh
   substrate-pallet-new simple_storage --template https://github.com/substrate-developer-hub/substrate-pallet-template
   ```

2. **Navigate to Your New Pallet**:

   ```sh
   cd simple_storage
   ```

3. **Define Your Pallet Logic**:

   Open `src/lib.rs` in your favorite editor. You will find some template code. Replace the content with the following to create a simple storage pallet:

   ```rust
   #[frame_support::pallet]
   pub mod pallet {
       use frame_support::{dispatch::DispatchResult, pallet_prelude::*};
       use frame_system::pallet_prelude::*;

       #[pallet::pallet]
       #[pallet::generate_store(pub(super) trait Store)]
       pub struct Pallet<T>(_);

       #[pallet::config]
       pub trait Config: frame_system::Config {}

       #[pallet::storage]
       #[pallet::getter(fn something)]
       pub type Something<T> = StorageValue<_, u32>;

       #[pallet::call]
       impl<T: Config> Pallet<T> {
           #[pallet::weight(10_000)]
           pub fn store_something(origin: OriginFor<T>, value: u32) -> DispatchResult {
               let _who = ensure_signed(origin)?;
               Something::<T>::put(value);
               Ok(())
           }
       }
   }
   ```

   This pallet defines a single storage value `Something` that stores a `u32` value. It also includes a function `store_something` that allows users to store a value.

## Step 3: Add Your Pallet to the Runtime

To use your new pallet, you must add it to the blockchain's runtime.

1. **Open `runtime/src/lib.rs`**:

   Add a reference to your new pallet at the top with other pallet references:

   ```rust
   pub use pallet_simple_storage;
   ```

2. **Configure the Runtime to Include Your Pallet**:

   Within the same file (`runtime/src/lib.rs`), find the `construct_runtime!` macro call. Add your pallet to the list as follows:

   ```rust
   SimpleStorage: pallet_simple_storage::{Pallet, Call, Storage},
   ```

3. **Implement the Config Trait for Your Pallet**:

   Still, in `runtime/src/lib.rs`, find the `impl` block for your pallet's `Config` trait and add:

   ```rust
   impl pallet_simple_storage::Config for Runtime {}
   ```

## Step 4: Compile and Run Your Blockchain

1. **Navigate Back to the Root Directory**:

   ```sh
   cd ../..
   ```

2. **Compile Your Blockchain**:

   ```sh
   cargo build --release
   ```

3. **Start Your Node**:

   ```sh
   ./target/release/node-template --dev --tmp
   ```

   The `--dev` flag runs your node in developer mode, and `--tmp` starts a temporary node for testing.

## Step 5: Interact with Your Blockchain

To interact with your blockchain, you can use the Polkadot JS Apps interface:

1. **Open [Polkadot JS Apps](https://polkadot.js.org/apps/#/explorer)** in your browser.
2. **Connect to Your Local Node**: Navigate to Settings

 > Development, and enter `ws://127.0.0.1:9944` as the custom endpoint.
3. **Interact with Your Pallet**: Go to Developer > Extrinsics, select your pallet's `store_something` function, enter a value, and submit the transaction.

## Conclusion

Congratulations! You've created a simple Substrate-based blockchain with a custom pallet for storing and retrieving values. This tutorial introduced you to the basics of Substrate development, including setting up a development environment, creating a custom pallet, adding it to your blockchain's runtime, and interacting with your blockchain. As you become more familiar with Substrate, you can explore more complex pallets, consensus mechanisms, and other advanced features to build sophisticated blockchain applications.