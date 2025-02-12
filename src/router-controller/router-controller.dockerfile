FROM rust:latest as builder
WORKDIR /app
COPY src/router-controller/crates/llm-router-gateway-api .
RUN cargo build --release

FROM nvcr.io/nvidia/base/ubuntu:22.04_20240212
RUN apt-get update && apt-get install -y curl jq ca-certificates
COPY --from=builder /app/target/release/llm-router-gateway-api /usr/local/bin/
RUN mkdir -p /app
WORKDIR /app
COPY src/router-controller/config.yaml /app/config.yaml
ENV RUST_LOG=info

ENTRYPOINT ["llm-router-gateway-api"]
CMD ["--config-path", "/app/config.yaml"]
