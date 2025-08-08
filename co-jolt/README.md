# Co-Jolt zkVM


MPC based on Replicated Secret Sharing (RSS).

## Requirements

- Rust `nightly`
    - rustup target add riscv32im-unknown-none-elf
- Misc packages: `build-essential libssl-dev pkg-config`

## Demo

```bash
bash ./demo.sh
```

## Acknowledgements

This prototype builds up on the following works:


## Known issues

- Delegator acts as coordinator (with logarithmic communication overhead).
- No private shared witness.
