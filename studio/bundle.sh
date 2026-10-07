#!/bin/bash
# Build a self-contained YuE2 Studio.app — engine (Python + torch) inside.
set -eo pipefail
STUDIO_DIR="$(cd "$(dirname "$0")" && pwd)"
ENGINE="$STUDIO_DIR/.build/engine"
cd "$STUDIO_DIR/YuE2Studio"

# 1. Build the Release app
xcodebuild -project YuE2Studio.xcodeproj -scheme YuE2Studio \
    -configuration Release build 2>&1 | grep -E "error:|BUILD" | tail -3

APP="$HOME/Library/Developer/Xcode/DerivedData/YuE2Studio-glbkfhmpzigcerbhqfjjdplejzqt/Build/Products/Release/YuE2 Studio.app"

# 2. Inject the bundled engine (assembled once into studio/.build/engine)
if [ ! -x "$ENGINE/bin/yue2" ]; then
    echo "Bundled engine missing — expected at $ENGINE"
    exit 1
fi
# Re-sync the app code so engine edits are never stale in the bundle
rsync -a --delete --exclude "__pycache__" "$STUDIO_DIR/../src/yue2" \
    "$ENGINE/site-packages/"
mkdir -p "$APP/Contents/Resources/engine"
rsync -a --delete --exclude "python.tar.gz" \
    "$ENGINE/" "$APP/Contents/Resources/engine/"

# 3. Ad-hoc sign everything (engine dylibs included), drop quarantine
codesign --force --deep --sign - "$APP" 2>/dev/null || true
xattr -dr com.apple.quarantine "$APP" 2>/dev/null || true

# 4. Deliver — ditto, not cp: preserves the bundle without trying to
#    replicate protected xattrs (macl/provenance), which cp chokes on.
rm -rf "$HOME/Desktop/YuE2 Studio.app"
ditto "$APP" "$HOME/Desktop/YuE2 Studio.app"

# 5. Installer package — payload → /Applications, license screen on install
PKG_STAGE="$STUDIO_DIR/.build/pkgroot"
rm -rf "$PKG_STAGE" "$STUDIO_DIR/.build/YuE2Studio-component.pkg" \
    "$STUDIO_DIR/.build/pkgres"
mkdir -p "$PKG_STAGE/Applications" "$STUDIO_DIR/.build/pkgres"
ditto "$APP" "$PKG_STAGE/Applications/YuE2 Studio.app"
# License as RTF — the installer centers plain .txt; RTF renders left-aligned.
# Both agreements: Apache 2.0 (code) + CC BY-NC 4.0 (model weights).
{ echo "YuE2 Studio — Software License Agreement"; echo;
  cat "$STUDIO_DIR/../LICENSE"; echo; echo;
  echo "YuE2 MODEL WEIGHTS — License Agreement"; echo;
  cat "$STUDIO_DIR/../MODEL_LICENSE"; } \
    | textutil -convert rtf -stdin -stdout \
    > "$STUDIO_DIR/.build/pkgres/license.rtf"

pkgbuild --root "$PKG_STAGE" \
    --identifier com.yue2.studio \
    --version 0.1 \
    --install-location / \
    "$STUDIO_DIR/.build/YuE2Studio-component.pkg"

cat > "$STUDIO_DIR/.build/distribution.xml" <<'XML'
<?xml version="1.0" encoding="utf-8"?>
<installer-gui-script minSpecVersion="1">
    <title>YuE2 Studio</title>
    <license file="license.rtf"/>
    <pkg-ref id="com.yue2.studio"/>
    <options customize="never" require-scripts="false"
             hostArchitectures="arm64"/>
    <choices-outline>
        <line choice="default">
            <line choice="com.yue2.studio"/>
        </line>
    </choices-outline>
    <choice id="default" title=""/>
    <choice id="com.yue2.studio" visible="false">
        <pkg-ref id="com.yue2.studio"/>
    </choice>
    <pkg-ref id="com.yue2.studio" version="0.1"
             onConclusion="none">YuE2Studio-component.pkg</pkg-ref>
</installer-gui-script>
XML

rm -f "$HOME/Desktop/YuE2-Studio-Installer.pkg"
productbuild --distribution "$STUDIO_DIR/.build/distribution.xml" \
    --resources "$STUDIO_DIR/.build/pkgres" \
    --package-path "$STUDIO_DIR/.build" \
    "$HOME/Desktop/YuE2-Studio-Installer.pkg"

echo "→ $HOME/Desktop/YuE2-Studio-Installer.pkg"
ls -lh "$HOME/Desktop/YuE2-Studio-Installer.pkg" | awk '{print $5}'
