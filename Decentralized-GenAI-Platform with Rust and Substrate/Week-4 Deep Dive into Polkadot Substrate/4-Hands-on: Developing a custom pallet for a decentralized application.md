Developing a custom pallet for a decentralized application using Substrate is a hands-on way to learn about blockchain development. This tutorial will guide you through the process of creating a basic pallet and integrating it into a Substrate runtime. We'll build a simple voting pallet that allows users to create and vote on proposals.


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
#[pallet::config]: This attribute macro indicates that the following trait defines the configuration for a pallet. The macro processes the trait to integrate it with the pallet's functionality.

pub trait Config: frame_system::Config: This line defines a public trait named Config, which is the configuration trait for your pallet. It inherits from frame_system::Config, meaning your pallet's configuration will automatically include (and must comply with) the configuration of the frame_system pallet. The frame_system pallet provides essential types and functions for the blockchain runtime, such as account management and block processing.

RuntimeEvent: This associated type defines the event type for the pallet. Events in Substrate are used to notify external entities about occurrences within the blockchain. The constraint From<Event<Self>> ensures that the pallet's events can be converted into the runtime-wide event type. IsType<<Self as frame_system::Config>::RuntimeEvent> ensures that this type is compatible with the event type defined in the frame_system configuration. This line is crucial for pallets that emit events, as it integrates the pallet's events into the broader runtime event system.

WeightInfo: This associated type specifies a trait or struct that provides weight information for the pallet's transactions. Transaction weights in Substrate are a measure of the computational and storage resources required to execute transactions. This information is used to calculate fees and limit the impact of transactions on block processing time. Implementations of this trait typically provide functions that return the weight of each callable function in the pallet, allowing for fine-tuned control over the blockchain's resource accounting.

### Step 3: Implementing the Pallet Logic

1. **Define Storage Items**: Use the `pallet::storage` macro to define storage items for your voting application. For example, store proposals and votes.

   ```rust
   #[pallet::storage]
   pub type Proposals<T> = StorageMap<_, Blake2_128Concat, u32, Proposal>;
   #[pallet::storage]
   pub type Votes<T> = StorageMap<_, Blake2_128Concat, u32, Vote>;
   ```

   StorageMap is a crucial concept in Substrate's FRAME framework, which facilitates the development of blockchain runtimes. It's a part of the FRAME Support library and is used within pallets to define a key-value storage structure. This structure allows you to map keys of one type to values of another type, providing efficient storage and retrieval. It's analogous to a hash map or dictionary in many programming languages but is specifically designed for use in the blockchain's state storage.

   The primary use of a StorageMap is to store and manage data on-chain in a way that is persistent between blocks. It allows developers to:

    * Associate values with specific keys, enabling efficient data lookup.
    * Iterate over entries when necessary, though iteration should be used judiciously due to the linear time complexity with respect to the number of items.
    * Easily insert, update, and remove items.

   
    #[pallet::storage]: This attribute marks the following type as a storage item for the pallet. Storage items are persistent data stored on the blockchain.

    pub type Proposals<T> and pub type Votes<T>: These lines define storage maps named Proposals and Votes, respectively. These are types that are parameterized over the pallet's configuration trait T (which must satisfy the Config trait bounds).

    StorageMap<_, Blake2_128Concat, u32, Proposal> and StorageMap<_, Blake2_128Concat, u32, Vote>: These storage maps map u32 keys to Proposal and Vote values, respectively. The _ placeholder for the hasher type is filled in by Blake2_128Concat, indicating that this hasher should be used for generating storage keys. The Blake2_128Concat hasher is a good default choice for most use cases, providing a strong hash function with key concatenation capabilities.
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

   The provided Rust code snippet is an example of how to define dispatchable functions in a Substrate pallet using the FRAME framework. Dispatchable functions are the public interface of the pallet; they can be called by external entities, such as users sending transactions to the blockchain. These functions are capable of altering the state of the blockchain. Let's break down the key components and understand how they work together.

   #[pallet::call]: This attribute marks the following impl block as containing dispatchable functions for the pallet. The macro processes these functions to integrate them into the pallet's callable interface.

   create_proposal:

    origin: OriginFor<T>: The first parameter of a dispatchable function is always the transaction's origin. This is a type that represents where the transaction came from (e.g., a signed transaction from a user, or another pallet).
    content: Vec<u8>: The data content of the proposal being created. Vec<u8> indicates that this content is a dynamic array of bytes, which is flexible to represent various forms of data.
    DispatchResult: The return type for dispatchable functions, indicating whether the function succeeded or failed. It can also provide an error message in case of failure.

    vote:
    Similar to create_proposal, but it takes a proposal_id to identify which proposal is being voted on, and a vote parameter of type bool indicating the user's vote (true for yes, false for no).

    #[pallet::weight(10_000)]: This attribute specifies the weight of the dispatchable function, which represents its computational and storage cost. The weight is used to calculate transaction fees and limit the impact of transactions on block execution time. In these examples, a placeholder weight of 10_000 is used, but in a real-world scenario, weights should be carefully benchmarked and set according to the actual resources the function consumes.


   Complete implementatyion of above code fragment:

    ```rust
    #[pallet::call]
   impl<T: Config> Pallet<T> {
      #[pallet::weight(10_000)]
      pub fn create_proposal(origin: OriginFor<T>, content: Vec<u8>) -> DispatchResult {
         let who = ensure_signed(origin)?;

         let proposal_id = ProposalCount::<T>::get();

         Proposals::<T>::insert(proposal_id, content.clone());
         Votes::<T>::insert(proposal_id, (Vec::new(), Vec::new())); // Initialize voting lists
         ProposalCount::<T>::put(proposal_id + 1);

         Self::deposit_event(Event::ProposalCreated(proposal_id, content));

         Ok(())
      }

      #[pallet::weight(10_000)]
      pub fn vote(origin: OriginFor<T>, proposal_id: u32, vote: bool) -> DispatchResult {
         let who = ensure_signed(origin)?;

         // Ensure the proposal exists
         ensure!(Proposals::<T>::contains_key(proposal_id), Error::<T>::ProposalNotFound);

         Votes::<T>::mutate(proposal_id, |vote_record| {
               if vote {
                  vote_record.0.push(who.clone()); // Yes votes
               } else {
                  vote_record.1.push(who.clone()); // No votes
               }
         });

         Self::deposit_event(Event::Voted(who, proposal_id, vote));

         Ok(())
      }
   }

   ```
   ensure_signed: Checks that the call to create_proposal is made by a signed account (not by another pallet). The account ID of the caller is returned and stored in who, which you might use, for example, to track who created the proposal.

    ProposalCount::\<T>::get(): Retrieves the current number of proposals from storage, used as the ID for the new proposal.

    Proposals::\<T>::insert(proposal_id, content.clone()): Stores the new proposal content in the Proposals map, keyed by the new proposal ID.
    ProposalCount::\<T>::put(proposal_id + 1): Increments the proposal 
    count and updates the value in storage.

    Self::deposit_event(Event::ProposalCreated(proposal_id, content)): Emits an event indicating a new proposal was created. Events can be listened to by external applications or within the blockchain for triggers.

   3. **Handle Events**: Define events to notify users of successful operations, like proposal creation and voting.

      ```rust
      #[pallet::event]
      #[pallet::generate_deposit(pub(super) fn deposit_event)]
      pub enum Event<T: Config> {
         ProposalCreated(u32, Vec<u8>),
         Voted(u32, bool),
      }
   ```

ProposalCreated(u32, Vec<u8>): The first variant of the Event enum. This event is emitted when a new proposal is created within the pallet. It contains two pieces of data:

   A u32 which could represent a proposal ID or index.
   A Vec<u8>, which is a dynamic array of bytes, potentially holding a serialized form of the proposal or some metadata about it.

Voted(u32, bool): The second variant of the Event enum. This event is emitted when a vote is cast on a proposal. It also contains two pieces of data:

   A u32 representing the proposal ID or index that was voted on.
   A bool indicating the outcome of the vote (e.g., true for yes/accept, false for no/reject).
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

   pub enum Runtime: This defines the Runtime enum, which represents the entirety of the runtime logic for the blockchain.

   The construct_runtime! macro serves as a central point where the runtime is pieced together from various pallets (modules) and configurations. Each part of the runtime, such as consensus mechanism, on-chain governance, or token handling, is developed in its own pallet, and construct_runtime! brings these components together to form a cohesive runtime.
   
   This macro is where you specify which pallets are included in your runtime, how they interact, and additional details like inherent data providers, and indexes for transaction fees, among other things.

    NodeBlock and UncheckedExtrinsic further detail the block's structure and the format of transactions that haven't been verified yet, respectively. 

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