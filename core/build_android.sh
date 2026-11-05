#!/bin/sh
# Cross-compile the core + CLI for arm64-v8a (NDK r29): core/build-android/stego_core_cli.
set -e
cd "$(dirname "$0")/.."
ANDROID_HOME="${ANDROID_HOME:-/opt/homebrew/share/android-commandlinetools}"
NDK="${NDK:-$ANDROID_HOME/ndk/29.0.14206865}"   # sdkmanager --install "ndk;29.0.14206865"
cmake -S core -B core/build-android -G Ninja -DCMAKE_TOOLCHAIN_FILE="$NDK/build/cmake/android.toolchain.cmake" \
  -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-26 -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXE_LINKER_FLAGS=-static-libstdc++
cmake --build core/build-android -j4
