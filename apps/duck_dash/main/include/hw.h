/* Hardware drivers in plain C: Feetech STS bus, BNO055 over i2c-dev, GPIO lines. */
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

/* ---- BNO055 ---- */
#define BNO_CHIP_ID 0xA0

typedef struct {
    int fd;
    int bus, addr;
} bno_dev;

int bno_open(bno_dev *d, int bus, int addr);       /* 0 = chip id ok */
int bno_scan(bno_dev *d);                          /* buses 0..7, 0x28/0x29 */
int bno_read(bno_dev *d, int reg, uint8_t *buf, int n);
int bno_write(bno_dev *d, int reg, uint8_t val);
/* Configure like odm_maix/imu.py (axis remap, offsets, NDOF) if the chip is
 * still in CONFIG mode. Leaves a running chip alone. */
int bno_setup(bno_dev *d, int upside_down, const char *offsets_json_path);
void bno_close(bno_dev *d);
void bno_remap_bytes(int upside_down, uint8_t *config, uint8_t *sign);

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
