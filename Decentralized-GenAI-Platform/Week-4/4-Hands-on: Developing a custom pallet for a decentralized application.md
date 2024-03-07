Developing a custom pallet for a decentralized application using Substrate is a hands-on way to learn about blockchain development. This tutorial will guide you through the process of creating a basic pallet and integrating it into a Substrate runtime. We'll build a simple voting pallet that allows users to create and vote on proposals.

### Prerequisites

- Basic understanding of Rust programming language.
- Familiarity with blockchain concepts.
- Substrate development environment set up ([official Substrate docs](https://substrate.dev/docs/en/knowledgebase/getting-started/) provide a comprehensive guide).

### Step 1: Setting Up Your Substrate Node Template

1. Clone the Substrate node template from the official repository. This template provides a good starting point for building your blockchain.
   
   ```bash
   git clone https://github.com/substrate-developer-hub/substrate-node-template
   ```

2. Navigate into the cloned directory and compile the code to ensure everything is set up correctly.

   ```bash
   cd substrate-node-template
   cargo build --release
   ```

### Step 2: Creating the Voting Pallet

1. **Pallet Skeleton**: Use the floowing Substrate pallet template to generate your voting pallet.

https://github.com/substrate-developer-hub/substrate-node-template/tree/main/pallets/template

Refere to this guide:
https://github.com/substrate-developer-hub/substrate-node-template/tree/main?tab=readme-ov-file#pallets


2. **Add Dependencies**: Open `pallets/voting/Cargo.toml` and add necessary dependencies. You might need dependencies like `frame-support` and `frame-system`.

3. **Define Pallet Configuration Trait**: In `pallets/voting/src/lib.rs`, define the configuration trait for your pallet. This includes specifying the types for events and any other configuration needed.

   ```rust
   #[pallet::config]
   pub trait Config: frame_system::Config {
       type Event: From<Event<Self>> + IsType<<Self as frame_system::Config>::Event>;
   }
   ```

### Step 3: Implementing the Pallet Logic

1. **Define Storage Items**: Use the `pallet::storage` macro to define storage items for your voting application. For example, store proposals and votes.

   ```rust
   #[pallet::storage]
   pub type Proposals<T> = StorageMap<_, Blake2_128Concat, u32, Proposal>;
   #[pallet::storage]
   pub type Votes<T> = StorageMap<_, Blake2_128Concat, u32, Vote>;
   ```

2. **Implement Functions**: Create functions to allow users to create proposals and vote on them. Ensure to use `pallet::call` to make these functions callable from outside.

   ```rust
   #[pallet::call]
   impl<T: Config> Pallet<T> {
       #[pallet::weight(10_000)]
       pub fn create_proposal(origin: OriginFor<T>, content: Vec<u8>) -> DispatchResult {
           // Implementation here
       }

       #[pallet::weight(10_000)]
       pub fn vote(origin: OriginFor<T>, proposal_id: u32, vote: bool) -> DispatchResult {
           // Implementation here
       }
   }
   ```

3. **Handle Events**: Define events to notify users of successful operations, like proposal creation and voting.

   ```rust
   #[pallet::event]
   #[pallet::generate_deposit(pub(super) fn deposit_event)]
   pub enum Event<T: Config> {
       ProposalCreated(u32, Vec<u8>),
       Voted(u32, bool),
   }
   ```

### Step 4: Integrating the Pallet into Your Runtime

1. **Add Your Pallet to the Runtime**: Open `runtime/src/lib.rs`. Import your pallet and add it to the `construct_runtime!` macro.

   ```rust
   impl voting::Config for Runtime {
       type Event = Event;
   }

   construct_runtime!(
       pub enum Runtime where
           Block = Block,
           NodeBlock = opaque::Block,
           UncheckedExtrinsic = UncheckedExtrinsic
       {
           // Other pallets
           Voting: voting::{Pallet, Call, Storage, Event<T>},
       }
   );
   ```

2. **Update `Cargo.toml`**: Ensure that the `runtime`'s `Cargo.toml` file includes your pallet as a dependency.

### Step 5: Testing Your Pallet

1. Write unit tests in `pallets/voting/src/tests.rs` to verify the functionality of your pallet. Test creating proposals, voting, and event emission.

2. Execute tests using Cargo:

   ```bash
   cargo test -p pallet-voting
   ```

### Step 6: Running Your Blockchain

1. Compile your node again with the new pallet integrated.

   ```bash
   cargo build --release
   ```

2. Start your blockchain.

   ```bash
   ./target/release