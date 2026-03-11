FYSTACK_DIR ?= ./nitro-enclaves-simulation
VSOCK_PORT  ?= 9000
SSH_PORT    ?= 2222

# Clone the simulation repo (one-time)
setup-sim:
	[ -d $(FYSTACK_DIR) ] || git clone https://github.com/fystack/nitro-enclaves-simulation.git $(FYSTACK_DIR)

# Build coordinator musl binary
build:
	docker build -f Dockerfile.builder -t co-jolt-coordinator-builder ../..
	docker create --name extract co-jolt-coordinator-builder && \
	docker cp extract:/src/target/x86_64-unknown-linux-musl/release/co-jolt-coordinator . && \
	docker rm extract

# Boot QEMU VM (wait for login prompt, then Ctrl+A X to detach)
setup-vm: setup-sim
	cd $(FYSTACK_DIR) && make setup-vm

# Copy coordinator binary into VM and start it
start-enclave: build
	scp -P $(SSH_PORT) ./co-jolt-coordinator user@localhost:~/
	ssh -p $(SSH_PORT) user@localhost 'nohup ~/co-jolt-coordinator &'

# Start LocalStack + vsock proxy
start-vsock-proxy: setup-sim
	cd $(FYSTACK_DIR) && make start-vsock-proxy

# SSH into VM for debugging
ssh-vm:
	ssh -p $(SSH_PORT) user@localhost

# View enclave logs
view-logs:
	cd $(FYSTACK_DIR) && make view-logs

clean:
	cd $(FYSTACK_DIR) && make clean 2>/dev/null; true
	rm -f co-jolt-coordinator

.PHONY: setup-sim build setup-vm start-enclave start-vsock-proxy ssh-vm view-logs clean
