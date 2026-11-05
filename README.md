# Stegologo

A logo that carries a link. Its outlines are nudged in and out by about 15 micrometres, far below what the eye (or a
printer's own wobble) would notice, and a phone camera reads them back. No QR code needed.

This repository holds the StegoCam Android scanner, the shared C++ decoder core it runs on, and a printable sheet of
demo cards (`demo_cards.pdf`).

## Try it on Android

You need an Android phone running Android 8.0 or newer with a 64-bit ARM processor (nearly every phone since about
2017), and a printer.

1. **Print the demo cards.** Print `demo_cards.pdf` on plain paper with any laser printer, at **100 % / "actual size"**
   (not "fit to page").
2. **Download the app.** On the phone, open the
   [latest release](https://github.com/stedra-demoscene/stegologo/releases/latest) and download `stegocam-1.0.apk`.
3. **Allow installing it.** The app isn't from the Play Store, so Android asks first. When you open the downloaded file,
   Android says something like *"For your security, your phone is not allowed to install unknown apps from this
   source"*. Tap **Settings**, switch on **Allow from this source** for your browser (or for your Files app, if you open
   the APK from there), then go back.
   The wording depends on the phone; the setting usually lives under *Settings → Apps → Special app access → Install
   unknown apps*.
4. **Install.** Tap **Install**. If Google Play Protect warns about an unknown developer, choose **More details →
   Install anyway**.
5. **Scan.** Open StegoCam, allow camera access, and hold the phone over a card so the card fills most of the frame.
   When it has read the card, a pill appears: **Open link**. Tap it.

You can switch the "unknown apps" permission off again after installing.

## FAQ

**What exactly is hidden, and where?**
The link is stored in the shape of the logo's outline. Along the edges, every 0.6 mm stretch (a "chip") is shifted a
tiny bit outwards or inwards, by about ±15 µm. The pattern of shifts carries the data. The ink, colours and design stay
the same.

**How can a phone see 15 µm?**
It can't, not for any single chip: one dimple is a fraction of a camera pixel and lost in the blur. The trick is that
the decoder knows the logo's ideal outline. It registers the photo to that reference, measures how far each stretch of
edge sits from where it should be, and then adds up many weak measurements:

- Each bit is spread over about 22 chips (spread spectrum), so random errors average out.
- A known pilot pattern on part of the chips tells the decoder that a code is present and how strong it is.
- Error correction (Golay(24,12)) repairs the bits that still come out wrong, and a CRC-16 checks the result.

Golay(24,12) is the same code that carried Voyager's pictures home from Jupiter and Saturn.

**How much data fits in a logo?**
The demo carries 42 bits: a 7-character short-link token. Capacity grows with the length of the outline, so a busier
logo, or a paragraph of text, holds more. A full cryptographic signature (e.g. 512 bits for Ed25519) needs roughly a
page of body text's worth of outline.

**Why a short link?**
42 bits is small, so the logo stores a 7-character token and the app turns it into a `bit.ly` link. Larger codes could
carry a full URL or a signature directly.

**Does it need special printing?**
No. The demo cards were printed on a cheap home laser printer. No special inks, no special press: just the printing
that's already there.

**Can I use my own logo?**
Not with this release. The app knows the reference geometry of the demo logo and lettering, and the encoder that dimples
new artwork isn't part of this repository.

**Why does the app also read QR codes?**
So the demo cards and ordinary QR codes give the same experience: whatever the app reads, you get one **Open link**
pill.

**Does the app send anything anywhere?**
No. It asks only for the camera (and vibration), not for internet access. Decoding runs on the phone. Tapping
**Open link** hands the link to your browser.

**Is there an iPhone version?**
No, Android only.

**Is this like a digital watermark?**
It's a cousin. Digital watermarks hide bits in an image's pixels and survive compression and resizing. This hides bits in
the geometry of printed shapes, and survives the round trip to paper and back through a phone camera.

**Who made this?**
Stefan Dragnev and Claude (team "Nothing To See Here").

## Building

- Android app: `cd android && ./gradlew assembleDebug` (or `assembleRelease` with your own signing config). Needs JDK 17,
  the Android SDK with NDK `29.0.14206865` and CMake `3.22.1`. arm64-v8a only. The decoder core is built in through
  CMake/JNI; QR codes are read with ML Kit's bundled model, so there's no Play-services download.
- Decoder core: `core/README.md`.

## License

GNU Affero General Public License v3.0 (AGPL-3.0). See [`LICENSE`](LICENSE).
