#!/bin/zsh
# Creates an installable .dmg (drag to Applications) from dist/ToshLLM.app
#
# The window layout lives in a committed .DS_Store template, because CI builds
# the release DMG and driving Finder from a runner does not work. Regenerate it
# here, on a real desktop, after moving anything:
#   TOSH_DMG_LAYOUT=1 ./scripts/make-dmg.sh
set -e
cd "$(dirname "$0")/.."

VERSION="${1:-v$(<VERSION)}"
APP="dist/ToshLLM.app"

[ -d "$APP" ] || { echo "$APP not found — run ./make-app.sh first"; exit 1; }

# The no-AVX2 legacy build ships as a distinctly named DMG so it stays on its own
# update channel. Read the variant straight from the built bundle (single source).
SUFFIX=""
if /usr/libexec/PlistBuddy -c "Print :TOSHNoAVX2" "$APP/Contents/Info.plist" 2>/dev/null | grep -qi true; then
    SUFFIX="-noavx2"
fi
DMG="dist/ToshLLM-$VERSION$SUFFIX.dmg"
VOLNAME="ToshLLM"
TEMPLATE="Assets/dmg/DS_Store"
ARTWORK="Assets/dmg/obsidian-hardware.png"

STAGE=$(mktemp -d)
TMPDMG=$(mktemp -u).dmg
cleanup() { rm -rf "$STAGE" "$TMPDMG" "$BGDIR" 2>/dev/null || true }
trap cleanup EXIT

cp -R "$APP" "$STAGE/"
ln -s /Applications "$STAGE/Applications"

# Window dressing is best-effort: a plain DMG still installs, so a toolchain
# without swiftc or tiffutil must not fail the release.
BGDIR=$(mktemp -d)
if swiftc -O scripts/dmg-background.swift -o "$BGDIR/dmgbg" 2>/dev/null \
   && "$BGDIR/dmgbg" "$BGDIR/bg.png" "$ARTWORK" AppIcon.icon/Assets/glyph.png 1 \
   && "$BGDIR/dmgbg" "$BGDIR/bg@2x.png" "$ARTWORK" AppIcon.icon/Assets/glyph.png 2; then
    mkdir -p "$STAGE/.background"
    if tiffutil -cathidpicheck "$BGDIR/bg.png" "$BGDIR/bg@2x.png" \
                -out "$STAGE/.background/background.tiff" > /dev/null 2>&1; then
        :
    else
        cp "$BGDIR/bg.png" "$STAGE/.background/background.tiff"
    fi
    [ -f "$APP/Contents/Resources/AppIcon.icns" ] \
        && cp "$APP/Contents/Resources/AppIcon.icns" "$STAGE/.VolumeIcon.icns"
    [ -f "$TEMPLATE" ] && [ -z "${TOSH_DMG_LAYOUT:-}" ] && cp "$TEMPLATE" "$STAGE/.DS_Store"
else
    echo "warning: could not render the background, shipping a plain DMG" >&2
fi

rm -f "$DMG"
hdiutil create -volname "$VOLNAME" -srcfolder "$STAGE" -ov -format UDRW "$TMPDMG" > /dev/null

# Finder addresses disks by name under /Volumes, so the layout pass cannot use a
# private mountpoint; everything else stays out of the user's way.
if [ -n "${TOSH_DMG_LAYOUT:-}" ]; then
    hdiutil detach "/Volumes/$VOLNAME" > /dev/null 2>&1 || true
    hdiutil attach "$TMPDMG" -noautoopen > /dev/null
    MOUNT="/Volumes/$VOLNAME"
else
    MOUNT=$(mktemp -d)
    hdiutil attach "$TMPDMG" -nobrowse -noautoopen -mountpoint "$MOUNT" > /dev/null
fi

# custom volume icon: the file alone does nothing without the folder's icon bit
if [ -f "$MOUNT/.VolumeIcon.icns" ]; then
    SetFile -a C "$MOUNT" 2>/dev/null || true
fi

if [ -n "${TOSH_DMG_LAYOUT:-}" ]; then
    echo "arranging the window with Finder…"
    osascript - "$VOLNAME" <<'APPLESCRIPT' || echo "warning: Finder would not arrange the window" >&2
on run argv
    set volName to item 1 of argv
    tell application "Finder"
        tell disk volName
            open
            set current view of container window to icon view
            set toolbar visible of container window to false
            set statusbar visible of container window to false
            -- the sidebar eats into the content area, and the background is sized to it
            set sidebar width of container window to 0
            set the bounds of container window to {200, 120, 840, 548}
            set opts to the icon view options of container window
            set arrangement of opts to not arranged
            set icon size of opts to 128
            set text size of opts to 13
            set background picture of opts to file ".background:background.tiff"
            set position of item "ToshLLM.app" of container window to {165, 194}
            set position of item "Applications" of container window to {475, 194}
            update without registering applications
            delay 1
            set the bounds of container window to {200, 120, 840, 548}
            delay 2
            close
        end tell
    end tell
end run
APPLESCRIPT
    sync
    mkdir -p "$(dirname "$TEMPLATE")"
    cp "$MOUNT/.DS_Store" "$TEMPLATE" && echo "saved the layout to $TEMPLATE"
fi

hdiutil detach "$MOUNT" > /dev/null
[ -n "${TOSH_DMG_LAYOUT:-}" ] || rmdir "$MOUNT" 2>/dev/null || true
hdiutil convert "$TMPDMG" -format UDZO -imagekey zlib-level=9 -o "$DMG" > /dev/null

echo "Done: $DMG"
