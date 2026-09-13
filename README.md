# Lyra

(I do want heavily emphasise that this is still a work in progress and there will be bugs and issues, but just dm them to me or put it in the issue tab and I will have a look at it. User: plutobypluto for dc. I also do a devlog here https://www.youtube.com/@ads_will1)

Lyra is an experimental digital radio mode for the amateur HF bands. It uses two GMSK rails which are 80 Hz apart and each message starts with a chirp. This makes it look and sound different from normal FT8.

There are two modes. Lyra F is the faster one (~2.3 s) and splits the message between the two rails. The chirp goes high to low. Lyra L is longer (~4.3 s) and sends the same data on both rails which can help if one rail is affected by fading. The chirp goes low to high. The chirp direction is what marks F vs L, not the channel.

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

Auto calls CQ and keeps making QSOs until you press stop. After a CQ it waits 5 s, then CQs again unless a reply starts a QSO.

Answer takes the first usable CQ right away, including ones already in the activity list, so you do not have to wait for the next CQ. Set channel to a free hole first (the channels list shows busy vs ok). That is your reply channel, not the CQ's channel.

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
