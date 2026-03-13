#!/usr/bin/env bash

setup_jemalloc_preset() {
  local preset=${1:-default}
  local current=${MALLOC_CONF:-}
  local effective=$current

  if [ -z "$effective" ] && [ "$preset" != "default" ]; then
    case "$preset" in
      return_os)
        effective="background_thread:true,dirty_decay_ms:1000,muzzy_decay_ms:1000,retain:false"
        ;;
      aggressive)
        effective="background_thread:true,dirty_decay_ms:0,muzzy_decay_ms:0,retain:false"
        ;;
      narenas1)
        effective="background_thread:true,narenas:1,percpu_arena:disabled,dirty_decay_ms:1000,muzzy_decay_ms:1000,retain:false"
        ;;
      *)
        echo "Unknown JEMALLOC_PRESET=$preset (expected default|return_os|aggressive|narenas1)" >&2
        return 1
        ;;
    esac
  fi

  if [ -n "$effective" ]; then
    export MALLOC_CONF="$effective"
  fi
}
