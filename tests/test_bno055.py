from odm_maix import bno055


def test_axis_remap_matches_adafruit():
    # adafruit_bno055 defaults: X->X, Y->Y, Z->Z, all positive => 0x24, 0x00
    assert bno055.axis_remap_bytes(0, 1, 2, 0, 0, 0) == (0x24, 0x00)
    # upright remap used by the Pi runtime: swap X/Y, negate X
    assert bno055.axis_remap_bytes(*bno055.REMAP_UPRIGHT) == (0x21, 0x04)
    assert bno055.axis_remap_bytes(*bno055.REMAP_UPSIDE_DOWN) == (0x21, 0x07)


def test_scales_match_adafruit():
    assert abs(bno055.GYRO_SCALE - 0.001090830782496456) < 1e-12
    assert bno055.ACC_SCALE == 0.01
