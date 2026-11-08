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

/* ===================================================================== BNO08x */

static void sleep_ms(int ms)
{
    struct timespec ts = {ms / 1000, (ms % 1000) * 1000000L};
    nanosleep(&ts, NULL);
}

static int bno_write_packet(bno_dev *d, int channel, const uint8_t *p, int n)
{
    uint8_t b[64];
    if (n + 4 > (int)sizeof b) return -1;
    b[0] = (uint8_t)((n + 4) & 0xFF);
    b[1] = (uint8_t)((n + 4) >> 8);
    b[2] = (uint8_t)channel;
    b[3] = d->seq[channel]++;
    memcpy(b + 4, p, n);
    return write(d->fd, b, n + 4) == n + 4 ? 0 : -1;
}

/* One SHTP packet into buf. Returns payload length (>0), 0 = nothing queued, -1 = I2C error. */
static int bno_read_packet(bno_dev *d, int *channel, uint8_t *buf, int cap)
{
    uint8_t h[4];
    if (read(d->fd, h, 4) != 4) return -1;
    int len = (h[0] | h[1] << 8) & 0x7FFF;
    if (len == 0 || len == 0x7FFF) return 0;
    if (len < 4 || h[2] > 5) return -1;
    *channel = h[2];
    int remaining = len - 4, got = 0;
    uint8_t chunk[256];
    while (remaining > 0) {
        int n = remaining + 4 < (int)sizeof chunk ? remaining + 4 : (int)sizeof chunk;
        if (read(d->fd, chunk, n) != n) return -1;
        int take = n - 4;
        if (got + take > cap) take = cap - got > 0 ? cap - got : 0; /* drop what does not fit */
        memcpy(buf + got, chunk + 4, take);
        got += take;
        remaining -= n - 4;
    }
    return got;
}

static int report_len(uint8_t id)
{
    switch (id) {
    case 0xF1: case 0xF8: return 16;
    case 0xFA: case 0xFB: return 5;
    case 0xFC: return 17;
    case 0x01: case 0x02: case 0x03: case 0x04: case 0x06: return 10;
    case 0x05: case 0x09: return 14;
    case 0x08: return 12;
    case 0x07: case 0x0F: return 16;
    default: return 0;
    }
}

static int bno_enable(bno_dev *d, int report_id, int interval_us)
{
    uint8_t r[17] = {0xFD, (uint8_t)report_id};
    for (int i = 0; i < 4; i++) r[5 + i] = (uint8_t)((unsigned)interval_us >> (8 * i));
    return bno_write_packet(d, 2, r, 17);
}

int bno_start(bno_dev *d, int interval_us)
{
    if (bno_enable(d, 0x01, interval_us) < 0 || bno_enable(d, 0x02, interval_us) < 0) return -1;
    d->interval_us = interval_us;
    return 0;
}

int bno_parse(bno_dev *d, int channel, const uint8_t *p, int n)
{
    if (channel == 1) {
        if (n >= 1 && p[0] == 1) {
            d->resets++;
            if (d->interval_us && d->fd >= 0) bno_start(d, d->interval_us); /* it forgot its reports */
        }
        return 0;
    }
    if (channel < 2 || channel > 4) return 0;
    int reports = 0;
    for (int i = 0; i < n;) {
        int len = report_len(p[i]);
        if (!len || i + len > n) break;
        const uint8_t *r = p + i;
        if (r[0] == 0xF8 && !d->product_ok) {
            d->product_ok = 1;
            d->reset_cause = r[1];
            d->sw_major = r[2];
            d->sw_minor = r[3];
            d->sw_patch = r[12] | r[13] << 8;
        } else if (r[0] >= 0x01 && r[0] <= 0x03) {
            float v[3];
            for (int k = 0; k < 3; k++) v[k] = (float)(int16_t)(r[4 + 2 * k] | r[5 + 2 * k] << 8);
            float scale = r[0] == 0x01 ? 1.0f / 256 : r[0] == 0x02 ? 1.0f / 512 : 1.0f / 16;
            float *dst = r[0] == 0x01 ? d->acc : r[0] == 0x02 ? d->gyro : NULL;
            if (dst)
                for (int k = 0; k < 3; k++) dst[k] = v[k] * scale;
            if (r[0] == 0x01) d->have_acc = 1;
            if (r[0] == 0x02) d->have_gyro = 1;
            d->accuracy[r[0] - 1] = r[2] & 3;
            reports++;
        }
        i += len;
    }
    return reports;
}

int bno_poll(bno_dev *d)
{
    static uint8_t buf[BNO_MAX_PACKET];
    int reports = 0;
    for (int k = 0; k < 32; k++) {
        int ch = 0;
        int n = bno_read_packet(d, &ch, buf, sizeof buf);
        if (n < 0) return -1;
        if (n == 0) break;
        reports += bno_parse(d, ch, buf, n);
    }
    return reports;
}

static int ask_product_id(bno_dev *d, int timeout_ms)
{
    uint8_t req[2] = {0xF9, 0};
    if (bno_write_packet(d, 2, req, 2) < 0) return -1;
    for (int t = 0; t < timeout_ms && !d->product_ok; t += 5) {
        if (bno_poll(d) < 0) return -1;
        if (!d->product_ok) sleep_ms(5);
    }
    return d->product_ok ? 0 : -1;
}

int bno_open(bno_dev *d, int bus, int addr)
{
    char path[32];
    memset(d, 0, sizeof *d);
    for (int i = 0; i < 3; i++) d->accuracy[i] = -1;
    snprintf(path, sizeof path, "/dev/i2c-%d", bus);
    d->fd = open(path, O_RDWR | O_CLOEXEC);
    d->bus = bus;
    d->addr = addr;
    if (d->fd < 0) return -errno;
    if (ioctl(d->fd, I2C_SLAVE, addr) < 0) {
        bno_close(d);
        return -errno;
    }
    uint8_t h[4];
    if (read(d->fd, h, 4) != 4) { /* nothing answers at this address */
        bno_close(d);
        return -ENODEV;
    }
    /* No reset first: diagnose.py may be reading it. Reset only if it stays quiet. */
    if (ask_product_id(d, 300) == 0) return 0;
    uint8_t rst = 1;
    bno_write_packet(d, 1, &rst, 1);
    memset(d->seq, 0, sizeof d->seq);
    d->seq[1] = 1;
    sleep_ms(300);
    if (ask_product_id(d, 1000) == 0) return 0;
    bno_close(d);
    return -ENODEV;
}

int bno_scan(bno_dev *d)
{
    static const int addrs[2] = {BNO_ADDR_B, BNO_ADDR_A};
    for (int bus = 0; bus < 8; bus++)
        for (int a = 0; a < 2; a++)
            if (bno_open(d, bus, addrs[a]) == 0) return 0;
    return -ENODEV;
}

void bno_close(bno_dev *d)
{
    if (d->fd >= 0) close(d->fd);
    d->fd = -1;
}

void bno_remap(const int remap[6], int upside_down, const float in[3], float out[3])
{
    /* odm_maix/bno08x.py: REMAP_UPRIGHT = (Y, X, Z, -, +, +), UPSIDE_DOWN = (Y, X, Z, -, -, -) */
    static const int up[6] = {1, 0, 2, 1, 0, 0}, down[6] = {1, 0, 2, 1, 1, 1};
    const int *m = remap && remap[0] >= 0 ? remap : upside_down ? down : up;
    for (int i = 0; i < 3; i++) out[i] = m[3 + i] ? -in[m[i]] : in[m[i]];
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
