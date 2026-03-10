FROM rust:1-alpine AS builder
RUN apk add --no-cache musl-dev cmake make perl linux-headers g++
WORKDIR /src
COPY . .
RUN RUSTFLAGS="-A warnings" cargo build --release \
    --target x86_64-unknown-linux-musl \
    -p co-jolt-coordinator
