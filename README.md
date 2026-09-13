# Lyra

(I do want heavily emphasise that this is still a work in progress and there will be bugs and issues, but just dm them to me or put it in the issue tab and I will have a look at it. User: plutobypluto for dc. I also do a devlog here https://www.youtube.com/@ads_will1)

Lyra is an experimental digital radio mode for the amateur HF bands. It uses two GMSK rails which are 80 Hz apart and each message starts with a chirp. This makes it look and sound different from normal FT8.

There are two modes. Lyra F is the faster one (~2.3 s) and splits the message between the two rails. The chirp goes high to low. Lyra L is longer (~4.3 s) and sends the same data on both rails which can help if one rail is affected by fading. The chirp goes low to high. The chirp direction is what marks F vs L, not the channel.

This is still experimental and I have mostly tested it using virtual audio cables, not a real radio.

## running

You need Python 3. On macOS or Linux run:

```shell
./run_lyra
```

For the long v2 demo:

```shell
./run_lyra_long_v2_demo
```

On Windows use `run_lyra.bat` or `run_lyra_long_v2_demo.bat`.

The first start can take a while because it installs the needed packages.

## using it

Select your audio input, enter your call and grid, then select F or L and a channel. Call, grid, and the USB dial are remembered.

The frequency box on the top bar is the USB dial. Pick 20m, 30m, or 15m, or type a custom dial. That is the only place to set frequency.

Monitor listens to the sound card. Decode turns the decoder on. Leave both on for normal use. max ch is how many channels the decoder will look at (up to 10).

Auto calls CQ and keeps making QSOs until you press stop. After a CQ it listens for about 4 s before calling again.

Answer only answers other stations calling CQ. It replies in the same mode as the CQ it heard, and it tries to key inside that 4 s window.

Manual makes one QSO and then stops.

The activity list is everything decoded. Double click a CQ, or select it and press work selected, to answer that station even if Auto would pick someone else.

The power slider changes the audio output level. Start it low when using a radio so it does not clip.

File → Open WAV decodes a recording. File → Preferences is ranking, already-worked, F/L answer, the dx column, and row colours.

## preferences

Priority is who Auto answers first: quickest, largest or smallest distance, highest or lowest dB, or manual pick only.

Already worked defaults to skip for this session. Answer again will call the same station more than once. Manual pick still works either way.

Answer can be F and L, F only, or L only. Auto will not start a QSO in a mode you turned off.

Show from grid can hide the dx column or show country, continent, or both.

## how it looks

Lyra F

![Lyra F](images/lyra_f.png)

Lyra L

![Lyra L](images/lyra_l.png)

## channels

USB dial presets are 20m 14.1064 MHz, 30m 10.1440 MHz, and 15m 21.1100 MHz. You can also type a custom USB dial in the frequency dropdown. F and L can use any of the 10 channels. All 10 channels sit inside a normal USB voice filter (about 380–2350 Hz audio). Channel 6 is around 1.5 kHz, not 3 kHz. The selected channel is only your transmit channel because the decoder listens to all 10. Auto answers use channels 1–5 so they stay in the middle of a typical SSB filter.

Transmit is one channel (two rails 80 Hz apart). The full 10-channel listen window is about 2 kHz above the dial, so on 20m that is about 14.1068 to 14.1088 MHz. These frequencies are not reserved for Lyra. Check that the channel is free and follow your local band rules before transmitting.

The 20m preset sits in the IARU all-mode digital area above the 14.099–14.101 beacons, not on FT8 14.074 or FT4 14.080. It is next to the 14.105 Olivia/packet watering hole, so listen first.

The 15m preset is 21.1100 MHz so the 10-channel window stays in the IARU Region 1 21.110–21.120 all-mode digital slice, below the 21.149–21.151 beacons and away from FT8 21.074 and FT4 21.140.

30m is a narrow secondary band. IARU Region 1 allows only 500 Hz-wide emissions in 10.130–10.150, and many places do not allow SSB there. Lyra transmit is one channel, but 30m is crowded: FT8 10.136, FT4/WSPR 10.140, JS8 10.130, RTTY/Olivia around 10.142, packet 10.147, APRS near 10.149. The 10.1440 preset puts the 10-channel window around 10.1444–10.1464, above those FT8/FT4 dials and below packet/APRS.

## radio

Open radio. Test is for virtual cables with no CAT. Hamlib talks to a radio over serial using the bundled `rigctld`. Hamlib network talks to an already running `rigctld`.

Lyra includes `rigctld` under `vendor/hamlib` for Windows, macOS (Apple Silicon and Intel), and Linux (x86_64 and arm64), so you do not need a separate Hamlib install. Pick Hamlib, choose your radio and serial port, then connect.

Linux needs a recent glibc (Ubuntu 24.04 or similar). You can still point the hamlib field at another folder, or use a copy from PATH.

Leave packet USB on if your radio has a data / USB-D / packet mode. That keeps the TX filter in the voice passband instead of flipping to wide SSB. Lyra sets the radio to the selected USB dial (or packet USB) and asks for a 6 kHz filter. PTT is controlled through Hamlib.

## testing

To test without a radio, open Lyra twice and connect both windows with a virtual audio cable. BlackHole can be used on macOS and VB-Cable can be used on Windows. Keep the rig option on Test.

The example qso wavs folder has some example QSOs you can open in SDR++. For the long v2 one, run `./run_lyra_long_v2_demo` and open the wav from the file menu.

## license

GPLv3.
