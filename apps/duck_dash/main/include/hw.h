/* Hardware drivers in plain C: Feetech STS bus, BNO08x over i2c-dev, GPIO lines. */
#ifndef DUCK_DASH_HW_H
#define DUCK_DASH_HW_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ---- Feetech STS3215 (same protocol and unit conversions as odm_maix/feetech.py) ---- */
#define FT_INST_PING 0x01
#define FT_INST_READ 0x02
#define FT_REG_PRESENT_POSITION 56 /* pos(2) vel(2) load(2) volt(1) temp(1) */
#define FT_REG_PRESENT_VOLTAGE 62

typedef struct {
    int fd;
    int timeout_ms;
    unsigned errors;
} ft_bus;

int ft_build_packet(uint8_t *out, int id, int inst, const uint8_t *params, int nparams);
/* Finds one status packet in buf. Returns bytes consumed (>0), 0 if incomplete, -1 if bad checksum. */
int ft_parse_status(const uint8_t *buf, int len, int *id, int *error, const uint8_t **params, int *nparams);
float ft_raw_to_rad_pos(unsigned raw);
float ft_raw_to_rad_vel(unsigned raw);
float ft_raw_to_load_pct(unsigned raw);

int ft_open(ft_bus *b, const char *port, int baud);
void ft_close(ft_bus *b);
int ft_ping(ft_bus *b, int id);
/* Reads n bytes from addr. Returns the status error byte (>=0) or -1. */
int ft_read(ft_bus *b, int id, int addr, int n, uint8_t *out);

/* ---- BNO08x (BNO085/BNO086): SH-2 over SHTP, same protocol as odm_maix/bno08x.py ---- */
#define BNO_ADDR_A 0x4A
#define BNO_ADDR_B 0x4B
#define BNO_MAX_PACKET 512

typedef struct {
    int fd;
    int bus, addr;
    uint8_t seq[6];
    float acc[3], gyro[3]; /* sensor frame, m/s^2 and rad/s */
    int have_acc, have_gyro;
    int accuracy[3];       /* accel gyro mag, 0..3, -1 unknown */
    int product_ok, reset_cause, sw_major, sw_minor, sw_patch;
    int resets;            /* unexpected chip resets seen */
    int interval_us;       /* reports enabled at this interval, 0 = none */
} bno_dev;

/* Opens /dev/i2c-bus and asks for the Product ID (soft reset only if it does
 * not answer). 0 = an SH-2 chip answered. */
int bno_open(bno_dev *d, int bus, int addr);
int bno_scan(bno_dev *d); /* buses 0..7, 0x4A/0x4B */
/* Stream calibrated accelerometer + gyroscope reports every interval_us. */
int bno_start(bno_dev *d, int interval_us);
/* Read every queued packet. Returns the number of sensor reports parsed, -1 on an I2C error. */
int bno_poll(bno_dev *d);
/* Parse one packet's payload (pure, for tests). Returns sensor reports parsed. */
int bno_parse(bno_dev *d, int channel, const uint8_t *p, int n);
void bno_close(bno_dev *d);
/* odm_maix/bno08x.py REMAP_UPRIGHT / REMAP_UPSIDE_DOWN, or a custom
 * [x_axis, y_axis, z_axis, x_neg, y_neg, z_neg] (remap[0] < 0 = use upside_down). */
void bno_remap(const int remap[6], int upside_down, const float in[3], float out[3]);

/* ---- GPIO character device (v1 line handles, as maix.gpio uses) ---- */
/* "GPIOB21" or "B21" -> chip 1, line 21 */
int gpio_parse(const char *name, int *chip, int *line);
/* returns a line fd (>=0) or -errno */
int gpio_request_input(int chip, int line, const char *label);
int gpio_get(int fd); /* 0/1 or -1 */

#ifdef __cplusplus
}
#endif
#endif
