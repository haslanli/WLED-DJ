#!/bin/zsh
# Build WLED DJ.app in the project folder: the menu-bar app plus the capture helper, ad-hoc signed.
# A rebuilt app has a new signature, so macOS asks for System Audio Recording permission again.
set -euo pipefail
root="${0:A:h:h}"
cd -- "$root"
swift build -c release --package-path app
app="$root/WLED DJ.app"
rm -rf "$app"
mkdir -p "$app/Contents/MacOS" "$app/Contents/Resources"
cp app/.build/release/WLEDDJ "$app/Contents/MacOS/WLED DJ"
cp app/.build/release/wled-capture "$app/Contents/MacOS/wled-capture"
cp app/Resources/Info.plist "$app/Contents/Info.plist"
/usr/libexec/PlistBuddy -c "Add :WLEDDJProjectRoot string $root" "$app/Contents/Info.plist"
codesign --force --sign - --identifier local.wled.dj.capture "$app/Contents/MacOS/wled-capture"
codesign --force --sign - --identifier local.wled.dj "$app"
echo "$app"
