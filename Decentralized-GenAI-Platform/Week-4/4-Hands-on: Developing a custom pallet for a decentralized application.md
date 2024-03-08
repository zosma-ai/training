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

3. **Define Pallet Configuration Trait**: Modify `pallets/voting/src/lib.rs`, 

```
#[frame_support::pallet]
pub mod pallet {
	use super::*;
	use frame_support::pallet_prelude::*;
	use frame_system::pallet_prelude::*;

	#[pallet::pallet]
	pub struct Pallet<T>(_);
   ...
}
```

The frame_support::pallet macro is part of the FRAME (Framework for Runtime Aggregation of Modularized Entities) framework, which is a set of libraries and tools for building blockchain runtimes in Rust, particularly for the Substrate blockchain framework. FRAME is designed to make it easier to develop secure, efficient, and modular blockchain systems. The pallet macro plays a crucial role in this ecosystem by providing a declarative way to define a pallet's components, including storage items, events, errors, and callable functions, among other things.

The statement use super::*; in Rust is used within a module to import all items (functions, types, constants, etc.) from the parent module into the current module's scope. This allows for convenient access to the parent module's public items without needing to prefix them with the module's name each time they are used.

Purpose of use frame_support::pallet_prelude::*;

The purpose of this statement is to bring into scope a predefined set of types, traits, and macros from the frame_support::pallet_prelude module. These are essential for defining pallets, which are modular units of logic in the Substrate runtime.

The pallet_prelude typically includes but is not limited to:

   Commonly used FRAME macros such as #[pallet::constant] and #[pallet::extra_constants] for defining constants in a pallet.
   Traits like MaybeSerializeDeserialize, Member, and FullCodec that are often used for type constraints in pallet storage.
   Utilities for handling weights and transactions, like Weight and DispatchResult.
   Other foundational types and traits necessary for pallet developm

While the exact contents can evolve over time with the development of FRAME, typically, frame_system::pallet_prelude::* might include:

   Traits like Origin, Config, and Call, which are essential for defining the pallet's configuration, origin of calls, and dispatchable functions.
   Types related to block information (BlockNumber, Hash), account data (AccountId, AccountInfo), and system events.

Define the configuration trait for your pallet. This includes specifying the types for events and any other configuration needed.

   ```rust
	#[pallet::config]
	pub trait Config: frame_system::Config {
		/// Because this pallet emits events, it depends on the runtime's definition of an event.
		type RuntimeEvent: From<Event<Self>> + IsType<<Self as frame_system::Config>::RuntimeEvent>;
		/// Type representing the weight of this pallet
		type WeightInfo: WeightInfo;
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