# Co-Spartan with Noir

Collaborative Spartan (based on [DFS](https://eprint.iacr.org/2025/296) scheme) with [Noir](https://github.com/noir-lang/noir) frontend.

MPC based on Replicated Secret Sharing (RSS).

## Requirements

- Rust `nightly`
    - rustup target add riscv32im-unknown-none-elf
- Noir `noirup --version nightly-2025-05-28`
- Misc packages: `openmpi-bin libopenmpi-dev build-essential libssl-dev pkg-config`

## Demo

```bash
bash ./demo.sh
```

## Acknowledgements

This prototype builds up on the following works:

- `co-spartan` and `spartan` crates: code adapted from [DFS paper prototype](https://zenodo.org/records/14677749?token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6ImI0NjE1ZWVkLWQ2MTgtNDEwNy1hMjFmLTg0MmQ0ZWE4MWE5NyIsImRhdGEiOnt9LCJyYW5kb20iOiIzM2QzYTM5ZjQ5ZWZkZjM2NTE1ZjllYjkzODA1NmU4ZiJ9.2y5WljMWenkgkxJCZVOilnGeMY1EkbeyZtph-2tu6W3Srh4LOGX7jxre8bZtooAkX8TRVScfV-HWA7THJ9ofpQ)
- `r1cs-noir` crate: https://github.com/worldfnd/ProveKit

## Known issues

- Delegator acts as coordinator (with logarithmic communication overhead).
- No private shared witness.
