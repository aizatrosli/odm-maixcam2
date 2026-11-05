#define _GNU_SOURCE
#include "hw.h"

#include <errno.h>
#include <fcntl.h>
#include <linux/gpio.h>
#include <linux/i2c-dev.h>
#include <linux/i2c.h>
#include <math.h>
#include <poll.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/file.h>
#include <sys/ioctl.h>
#include <termios.h>
#include <time.h>
#include <unistd.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static double mono_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e3 + ts.tv_nsec / 1e6;
}

/* ===================================================================== Feetech */

int ft_build_packet(uint8_t *out, int id, int inst, const uint8_t *params, int nparams)
{
    int sum = id + nparams + 2 + inst;
    out[0] = 0xFF;
    out[1] = 0xFF;
    out[2] = (uint8_t)id;
    out[3] = (uint8_t)(nparams + 2);
    out[4] = (uint8_t)inst;
    for (int i = 0; i < nparams; i++) {
        out[5 + i] = params[i];
        sum += params[i];
    }
    out[5 + nparams] = (uint8_t)(~sum & 0xFF);
    return 6 + nparams;
}

int ft_parse_status(const uint8_t *buf, int len, int *id, int *error, const uint8_t **params, int *nparams)
{
    int i = 0;
    while (i + 1 < len && !(buf[i] == 0xFF && buf[i + 1] == 0xFF)) i++;
    /* a run of 0xFF: the header is the last two before the id */
    while (i + 2 < len && buf[i + 2] == 0xFF) i++;
    if (i + 5 > len) return 0;
    int length = buf[i + 3];
    int end = i + 3 + length; /* checksum index */
    if (length < 2 || end >= len) return 0;
    int sum = 0;
    for (int k = i + 2; k < end; k++) sum += buf[k];
    if ((uint8_t)(~sum & 0xFF) != buf[end]) return -1;
    *id = buf[i + 2];
    *error = buf[i + 4];
    *params = buf + i + 5;
    *nparams = length - 2;
    return end + 1;
}

static int sign_mag(unsigned raw, int bit)
{
    if (raw & (1u << bit)) return -(int)(raw & ((1u << bit) - 1));
    return (int)raw;
}

float ft_raw_to_rad_pos(unsigned raw)
{
    return (float)(2.0 * M_PI * sign_mag(raw, 15) / 4096.0 - M_PI);
}

float ft_raw_to_rad_vel(unsigned raw)
{
    return (float)(2.0 * M_PI * sign_mag(raw, 15) / 4095.0);
}

float ft_raw_to_load_pct(unsigned raw)
{
    return sign_mag(raw & 0x7FF, 10) / 10.0f;
}

static speed_t baud_const(int baud)
{
    switch (baud) {
    case 9600: return B9600;
    case 57600: return B57600;
    case 115200: return B115200;
    case 500000: return B500000;
    case 1000000: return B1000000;
    default: return B1000000;
    }
}

int ft_open(ft_bus *b, const char *port, int baud)
{
    memset(b, 0, sizeof *b);
    b->timeout_ms = 20;
    b->fd = open(port, O_RDWR | O_NOCTTY | O_NONBLOCK | O_CLOEXEC);
    if (b->fd < 0) return -errno;
    if (flock(b->fd, LOCK_EX | LOCK_NB) < 0) {
        int e = errno;
        close(b->fd);
        b->fd = -1;
        return -e;
    }
    struct termios t;
    if (tcgetattr(b->fd, &t) == 0) {
        cfmakeraw(&t);
        t.c_cflag |= CLOCAL | CREAD;
        t.c_cflag &= ~(CSTOPB | CRTSCTS);
        cfsetispeed(&t, baud_const(baud));
        cfsetospeed(&t, baud_const(baud));
        t.c_cc[VMIN] = 0;
        t.c_cc[VTIME] = 0;
        tcsetattr(b->fd, TCSANOW, &t);
    }
    return 0;
}

void ft_close(ft_bus *b)
{
    if (b->fd >= 0) close(b->fd);
    b->fd = -1;
}

/* Sends a packet and reads until a whole status packet for `id` arrived. */
static int txrx(ft_bus *b, const uint8_t *pkt, int n, int id, uint8_t *resp, int cap, int *rlen)
{
    tcflush(b->fd, TCIFLUSH);
    if (write(b->fd, pkt, n) != n) return -1;
    tcdrain(b->fd);
    int got = 0;
    double deadline = mono_ms() + b->timeout_ms;
    for (;;) {
        int rid, err, np;
        const uint8_t *par;
        if (got) {
            int used = ft_parse_status(resp, got, &rid, &err, &par, &np);
            if (used < 0) return -1;
            if (used > 0) {
                if (rid == id) {
                    *rlen = got;
                    return 0;
                }
                memmove(resp, resp + used, got - used); /* someone else's reply */
                got -= used;
                continue;
            }
        }
        double left = deadline - mono_ms();
        if (left <= 0 || got >= cap) return -1;
        struct pollfd p = {b->fd, POLLIN, 0};
        if (poll(&p, 1, (int)left + 1) <= 0) return -1;
        ssize_t r = read(b->fd, resp + got, cap - got);
        if (r > 0) got += r;
        else if (r < 0 && errno != EAGAIN && errno != EINTR) return -1;
    }
}

int ft_ping(ft_bus *b, int id)
{
    uint8_t pkt[8], resp[32];
    int n = ft_build_packet(pkt, id, FT_INST_PING, NULL, 0), rl;
    return txrx(b, pkt, n, id, resp, sizeof resp, &rl) == 0;
}

int ft_read(ft_bus *b, int id, int addr, int len, uint8_t *out)
{
    uint8_t pkt[8], resp[64], par[2] = {(uint8_t)addr, (uint8_t)len};
    int n = ft_build_packet(pkt, id, FT_INST_READ, par, 2), rl;
    if (txrx(b, pkt, n, id, resp, sizeof resp, &rl) < 0) {
        b->errors++;
        return -1;
    }
    int rid, err, np;
    const uint8_t *p;
    if (ft_parse_status(resp, rl, &rid, &err, &p, &np) <= 0 || np < len) {
        b->errors++;
        return -1;
    }
    memcpy(out, p, len);
    return err;
}

/* ===================================================================== BNO055 */

int bno_read(bno_dev *d, int reg, uint8_t *buf, int n)
{
    uint8_t r = (uint8_t)reg;
    struct i2c_msg msgs[2] = {
        {(uint16_t)d->addr, 0, 1, &r},
        {(uint16_t)d->addr, I2C_M_RD, (uint16_t)n, buf},
    };
    struct i2c_rdwr_ioctl_data x = {msgs, 2};
    if (ioctl(d->fd, I2C_RDWR, &x) == 2) return 0;
    /* some controllers dislike repeated starts: plain write then read */
    if (write(d->fd, &r, 1) != 1) return -1;
    return read(d->fd, buf, n) == n ? 0 : -1;
}

int bno_write(bno_dev *d, int reg, uint8_t val)
{
    uint8_t b[2] = {(uint8_t)reg, val};
    return write(d->fd, b, 2) == 2 ? 0 : -1;
}

int bno_open(bno_dev *d, int bus, int addr)
{
    char path[32];
    snprintf(path, sizeof path, "/dev/i2c-%d", bus);
    d->fd = open(path, O_RDWR | O_CLOEXEC);
    d->bus = bus;
    d->addr = addr;
    if (d->fd < 0) return -errno;
    if (ioctl(d->fd, I2C_SLAVE, addr) < 0) {
        bno_close(d);
        return -errno;
    }
    uint8_t id = 0;
    bno_write(d, 0x07, 0); /* page 0 */
    if (bno_read(d, 0x00, &id, 1) < 0 || id != BNO_CHIP_ID) {
        bno_close(d);
        return -ENODEV;
    }
    return 0;
}

int bno_scan(bno_dev *d)
{
    for (int bus = 0; bus < 8; bus++)
        for (int a = 0x28; a <= 0x29; a++)
            if (bno_open(d, bus, a) == 0) return 0;
    return -ENODEV;
}

void bno_close(bno_dev *d)
{
    if (d->fd >= 0) close(d->fd);
    d->fd = -1;
}

void bno_remap_bytes(int upside_down, uint8_t *config, uint8_t *sign)
{
    /* odm_maix/bno055.py: REMAP_UPRIGHT = (Y, X, Z, -, +, +), UPSIDE_DOWN = (Y, X, Z, -, -, -) */
    *config = (2 << 4) | (0 << 2) | 1;
    *sign = upside_down ? 0x07 : 0x04;
}

static void sleep_ms(int ms)
{
    struct timespec ts = {ms / 1000, (ms % 1000) * 1000000L};
    nanosleep(&ts, NULL);
}

static int load_offsets(const char *path, int16_t out[11])
{
    FILE *f = path ? fopen(path, "r") : NULL;
    if (!f) return -1;
    char buf[1024];
    size_t n = fread(buf, 1, sizeof buf - 1, f);
    fclose(f);
    buf[n] = 0;
    char *p = strstr(buf, "bno055_offsets");
    if (!p || !(p = strchr(p, '['))) return -1;
    p++;
    for (int i = 0; i < 11; i++) {
        char *e;
        long v = strtol(p, &e, 10);
        if (e == p) return -1;
        out[i] = (int16_t)v;
        p = e;
        while (*p == ',' || *p == ' ' || *p == '\n' || *p == '\r' || *p == '\t') p++;
    }
    return 0;
}

int bno_setup(bno_dev *d, int upside_down, const char *offsets_path)
{
    uint8_t mode = 0;
    if (bno_read(d, 0x3D, &mode, 1) < 0) return -1;
    if ((mode & 0x0F) != 0) return 1; /* someone already runs it (walk.py, diagnose.py) */
    uint8_t cfg, sign;
    bno_remap_bytes(upside_down, &cfg, &sign);
    bno_write(d, 0x3E, 0x00); /* normal power */
    bno_write(d, 0x3F, 0x00); /* internal oscillator */
    bno_write(d, 0x3B, 0x00); /* m/s^2, dps, deg, C */
    bno_write(d, 0x41, cfg);
    bno_write(d, 0x42, sign);
    int16_t off[11];
    if (load_offsets(offsets_path, off) == 0) {
        const uint8_t *raw = (const uint8_t *)off; /* little endian, like struct.pack("<11h") */
        for (int i = 0; i < 22; i++) bno_write(d, 0x55 + i, raw[i]);
    }
    bno_write(d, 0x3D, 0x0C); /* NDOF */
    sleep_ms(20);
    return 0;
}

/* ===================================================================== GPIO */

int gpio_parse(const char *name, int *chip, int *line)
{
    if (strncmp(name, "GPIO", 4) == 0) name += 4;
    if (name[0] < 'A' || name[0] > 'Z') return -1;
    char *e;
    long v = strtol(name + 1, &e, 10);
    if (e == name + 1 || *e) return -1;
    *chip = name[0] - 'A';
    *line = (int)v;
    return 0;
}

int gpio_request_input(int chip, int line, const char *label)
{
    char path[32];
    snprintf(path, sizeof path, "/dev/gpiochip%d", chip);
    int fd = open(path, O_RDONLY | O_CLOEXEC);
    if (fd < 0) return -errno;
    struct gpiohandle_request req;
    memset(&req, 0, sizeof req);
    req.lineoffsets[0] = line;
    req.lines = 1;
    req.flags = GPIOHANDLE_REQUEST_INPUT;
    snprintf(req.consumer_label, sizeof req.consumer_label, "%s", label);
    int r = ioctl(fd, GPIO_GET_LINEHANDLE_IOCTL, &req);
    int e = errno;
    close(fd);
    if (r < 0) return -e;
    return req.fd;
}

int gpio_get(int fd)
{
    struct gpiohandle_data data;
    memset(&data, 0, sizeof data);
    if (ioctl(fd, GPIOHANDLE_GET_LINE_VALUES_IOCTL, &data) < 0) return -1;
    return data.values[0] ? 1 : 0;
}
