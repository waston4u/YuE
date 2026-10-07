#!/bin/bash
# Build YuE2 Studio.app (Apple Silicon only, macOS 14+, ad-hoc signed).
set -euo pipefail

cd "$(dirname "$0")/../studio/YuE2Studio"

if [[ "$(uname -m)" != "arm64" ]]; then
    echo "YuE2 Studio builds on Apple Silicon only." >&2
    exit 1
fi

xcodebuild -project YuE2Studio.xcodeproj \
           -scheme YuE2Studio \
           -configuration Release \
           -destination 'platform=macOS,arch=arm64' \
           -derivedDataPath build \
           build

APP="build/Build/Products/Release/YuE2 Studio.app"
if [[ -d "$APP" ]]; then
    codesign --force --deep --sign - "$APP" 2>/dev/null || true
    echo "Built: $APP"
else
    echo "Build output not found; check xcodebuild log." >&2
    exit 1
fi
