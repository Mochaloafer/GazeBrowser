"""Microphone settings shared by the speech module."""

MIC_DEVICE = None

# A small minimum-volume filter, not automatic gain calibration.
MIN_FRAME_RMS = 0.002

# WebRTC VAD: 0 is least aggressive; 3 is most aggressive.
VAD_AGGRESSIVENESS = 3

# Print microphone diagnostics while troubleshooting.
DEBUG_AUDIO = True