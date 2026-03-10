## co-jolt-coordinator TEE AWS Nitro port 

### 1. Key Requirements (Invariant) for Security and Performance

**Security Invariants:**

* **End-to-End Encryption (E2EE):** The Host EC2 instance must remain a "dumb proxy." You must terminate a TLS connection (using a crate like `rustls`) strictly *inside* the enclave. The network payload is routed through the host via TCP-to-vsock, but remains encrypted until it crosses the enclave boundary.
* **Cryptographic Attestation:** Before any sumcheck data is transmitted, the enclave must generate an ephemeral keypair and ask the Nitro Security Module (NSM) for an Attestation Document. The external client must verify this document (checking the AWS root of trust and the code hashes) to guarantee they are talking to your untampered code.
* **Constant-Time Execution:** TEEs encrypt memory but do not prevent CPU timing side-channels. Your `JoltField` operations and polynomial evaluations must use constant-time math to prevent the Host OS from inferring secret data by monitoring execution times.

**Performance Invariants:**

* **I/O Batching:** While `vsock` is fast, syscalls between the host and the enclave have overhead. Avoid sending single field elements. Batch your `broadcast_request` calls where the protocol allows.
* **Memory Provisioning:** Nitro Enclaves do not support swap space. Your `compressed_polys` and `r_sumcheck` vectors will consume physical RAM. If the enclave exceeds its allocated memory, it will panic and terminate. You must pre-allocate sufficient memory in the host's allocator configuration.

---

### 2. Constraints

* **No Network Access:** The enclave cannot make HTTP calls, resolve DNS, or connect to the internet. All communication is restricted to local `vsock` connections to the parent EC2 instance.
* **No Persistent Storage:** There is no disk access. All state is ephemeral and lives in RAM. If the enclave crashes or the EC2 instance reboots, all data, attestation documents, and ephemeral keys are permanently lost.
* **Limited System Calls:** Standard Linux syscalls like `fork`, `exec`, or file I/O will fail. Your Rust binary must be statically compiled (e.g., using the `musl` target).
* **The Debugging Paradox:** Running the enclave with the `--debug-mode` flag allows you to read console output (stdout/stderr) for troubleshooting, but it alters the Platform Configuration Registers (PCR hashes) and breaks production security guarantees.
* **Simulation Limitations:** The `fystack/nitro-enclaves-simulation` environment provides an *emulated* NSM device. It will generate dummy attestation documents for testing your application flow, but you cannot test true cryptographic verification against the real AWS Root of Trust until you are on a real Nitro instance.

---

### 3. Development Steps (Updated for Local Simulation)

**Step 1: Build the Enclave E2EE Transport Layer**

* Create a `VsockTlsCoordinator` that implements your `Rep3NetworkCoordinator` trait.
* Use the `vsock` crate to listen for incoming local connections from the host.
* Wrap the `vsock` stream in `rustls`. Your enclave will act as the TLS server, completing the handshake and decrypting the payload in memory.

**Step 2: Implement Cryptographic Attestation**

* Integrate the `aws-nitro-enclaves-nsm-api` crate.
* On boot, generate an ephemeral RSA or ECDSA keypair.
* Request an attestation document from the NSM, passing the public key as the `public_key` parameter. Embed this document in your initial TCP/TLS handshake. *(Note: The simulation environment will intercept this request and return a mock document so your code doesn't crash).*

**Step 3: Build the Untrusted Host Proxy**

* Write a lightweight TCP server (e.g., using `tokio`) that runs alongside the enclave.
* This server listens on a standard port, accepts external TCP connections from other MPC nodes, and blindly pipes the raw byte stream into the enclave's `vsock` CID and port.

**Step 4: Local Compilation and Simulation (`fystack`)**

* Compile your Rust code to `x86_64-unknown-linux-musl`.
* Clone `https://github.com/fystack/nitro-enclaves-simulation` and set up the QEMU environment as per their docs.
* Use the simulation tools to build your `.eif` file locally.
* Boot your `.eif` inside the QEMU VM. The simulation handles the `vsock` bridging between your host machine (Mac/Linux) and the QEMU VM.
* Run your Host Proxy locally to bridge standard TCP to the emulated `vsock` endpoint.

**Step 5: Update and Test the External Verifier/Client**

* Modify the external clients connecting to your prover.
* Test the complete flow locally: Client -> Local TCP Proxy -> Simulated vsock -> Enclave.
* *Crucial local test:* The client should temporarily accept the dummy attestation document provided by the `fystack` simulation to verify that the E2EE TLS tunnel opens properly.
