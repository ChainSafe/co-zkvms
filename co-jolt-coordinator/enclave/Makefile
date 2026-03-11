VSOCK_PORT     ?= 9000
ENCLAVE_CPUS   ?= 2
ENCLAVE_MEM_MB ?= 4096
DOCKER_TAG     ?= co-jolt-coordinator-enclave
BUILDER_TAG    ?= co-jolt-coordinator-builder
EIF_PATH       ?= coordinator.eif

# Dory SRS range to embed in the EIF.
# SRS vars = 8 + log2(padded_trace_length):
#   18 → trace ~1K     (69K file)
#   22 → trace ~16K    (216K file)
#   24 → trace ~64K    (410K file)
#   26 → trace ~256K   (796K file)
#   28 → trace ~1M     (1.6M file)
SRS_MIN ?= 18
SRS_MAX ?= 28

# SRS files live at the repo root (generated on first worker/client run).
REPO_ROOT := ../..
SRS_STAGING := $(REPO_ROOT)/.srs

# Build EIF from Dockerfile.eif
build-eif: stage-srs
	docker build -f Dockerfile.eif -t $(DOCKER_TAG) $(REPO_ROOT)
	nitro-cli build-enclave --docker-uri $(DOCKER_TAG) --output-file $(EIF_PATH)
	rm -rf $(SRS_STAGING)

# Stage SRS files into .srs/ for Docker COPY
stage-srs:
	@mkdir -p $(SRS_STAGING)
	@found=0; \
	for n in $$(seq $(SRS_MIN) $(SRS_MAX)); do \
		f="$(REPO_ROOT)/dory_urs_$${n}_variables.urs"; \
		if [ -f "$$f" ]; then \
			cp "$$f" $(SRS_STAGING)/; \
			found=$$((found + 1)); \
		fi; \
	done; \
	if [ $$found -eq 0 ]; then \
		echo "ERROR: No dory_urs_*.urs files found in range [$(SRS_MIN), $(SRS_MAX)]"; \
		echo "  Run a worker/client first to generate them, or adjust SRS_MIN/SRS_MAX."; \
		rm -rf $(SRS_STAGING); \
		exit 1; \
	fi; \
	echo "Staged $$found SRS files (vars $(SRS_MIN)..$(SRS_MAX))"

# Build host_proxy binary (extracted from Docker)
build-host-proxy:
	docker build -f Dockerfile.builder -t $(BUILDER_TAG) $(REPO_ROOT)
	docker create --name extract $(BUILDER_TAG) 2>/dev/null || true
	docker cp extract:/src/target/x86_64-unknown-linux-musl/release/host_proxy .
	docker rm extract

# Run enclave (debug mode for console output)
run-enclave:
	nitro-cli run-enclave \
	  --cpu-count $(ENCLAVE_CPUS) \
	  --memory $(ENCLAVE_MEM_MB) \
	  --eif-path $(EIF_PATH) \
	  --debug-mode

# Start host_proxy (TCP:VSOCK_PORT -> enclave vsock)
start-host-proxy:
	@VSOCK_CID=$$(nitro-cli describe-enclaves | python3 -c \
	  "import sys,json; print(json.load(sys.stdin)[0]['EnclaveCID'])") && \
	echo "host_proxy: TCP 0.0.0.0:$(VSOCK_PORT) -> vsock $$VSOCK_CID:$(VSOCK_PORT)" && \
	VSOCK_CID=$$VSOCK_CID TCP_LISTEN_ADDR=0.0.0.0:$(VSOCK_PORT) VSOCK_PORT=$(VSOCK_PORT) \
	  ./host_proxy

# View enclave console (debug mode only)
console:
	nitro-cli console --enclave-id $$(nitro-cli describe-enclaves | python3 -c \
	  "import sys,json; print(json.load(sys.stdin)[0]['EnclaveID'])")

# Terminate running enclave
terminate:
	nitro-cli terminate-enclave --all 2>/dev/null || true

# Describe running enclaves
describe:
	nitro-cli describe-enclaves

clean: terminate
	rm -f coordinator.eif host_proxy
	rm -rf $(SRS_STAGING)
	docker rmi $(DOCKER_TAG) $(BUILDER_TAG) 2>/dev/null || true

.PHONY: build-eif stage-srs build-host-proxy run-enclave start-host-proxy console terminate describe clean
