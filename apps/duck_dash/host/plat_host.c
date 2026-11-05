/* Host / plain-Linux platform: draws to /dev/fb0 when there is one (--fb),
 * otherwise runs headless (use --snapshot and --dump). No Sipeed SDK, so no
 * camera, NPU load test or pin functions: run the MaixCDK build on the board. */
#include <fcntl.h>
#include <linux/fb.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>

#include "dash.h"

typedef struct {
    int fd;
    uint8_t *mem;
    size_t len;
    struct fb_var_screeninfo var;
    struct fb_fix_screeninfo fix;
    const char *dev;
} fb_ctx;

static int fb_open(void *p, int *w, int *h)
{
    fb_ctx *c = p;
    if (!c->dev) return -1;
    c->fd = open(c->dev, O_RDWR | O_CLOEXEC);
    if (c->fd < 0) return -1;
    if (ioctl(c->fd, FBIOGET_VSCREENINFO, &c->var) < 0 || ioctl(c->fd, FBIOGET_FSCREENINFO, &c->fix) < 0) {
        close(c->fd);
        return -1;
    }
    c->len = (size_t)c->fix.line_length * c->var.yres_virtual;
    c->mem = mmap(NULL, c->len, PROT_READ | PROT_WRITE, MAP_SHARED, c->fd, 0);
    if (c->mem == MAP_FAILED) {
        close(c->fd);
        return -1;
    }
    *w = c->var.xres;
    *h = c->var.yres;
    return 0;
}

static void fb_show(void *p, const uint8_t *rgb, int w, int h)
{
    fb_ctx *c = p;
    int bpp = c->var.bits_per_pixel / 8;
    int W = (int)c->var.xres < w ? (int)c->var.xres : w;
    int H = (int)c->var.yres < h ? (int)c->var.yres : h;
    for (int y = 0; y < H; y++) {
        uint8_t *dst = c->mem + (size_t)(y + c->var.yoffset) * c->fix.line_length;
        const uint8_t *s = rgb + (size_t)y * w * 3;
        for (int x = 0; x < W; x++, s += 3) {
            uint32_t r = s[0], g = s[1], b = s[2];
            uint32_t v = ((r >> (8 - c->var.red.length)) << c->var.red.offset) |
                         ((g >> (8 - c->var.green.length)) << c->var.green.offset) |
                         ((b >> (8 - c->var.blue.length)) << c->var.blue.offset);
            if (c->var.transp.length) v |= ((1u << c->var.transp.length) - 1) << c->var.transp.offset;
            memcpy(dst + x * bpp, &v, bpp); /* little endian */
        }
    }
}

int main(int argc, char **argv)
{
    fb_ctx ctx = {.fd = -1};
    /* --fb [DEV] is ours; everything else goes to dash_main */
    char **args = calloc(argc + 1, sizeof *args);
    int n = 0;
    for (int i = 0; i < argc; i++) {
        if (!strcmp(argv[i], "--fb")) {
            ctx.dev = (i + 1 < argc && argv[i + 1][0] == '/') ? argv[++i] : "/dev/fb0";
            continue;
        }
        args[n++] = argv[i];
    }
    dash_platform plat = {0};
    plat.ctx = &ctx;
    plat.name = "linux";
    if (ctx.dev) {
        plat.display_open = fb_open;
        plat.display_show = fb_show;
    }
    int r = dash_main(n, args, &plat);
    if (ctx.mem && ctx.mem != MAP_FAILED) munmap(ctx.mem, ctx.len);
    if (ctx.fd >= 0) close(ctx.fd);
    free(args);
    return r;
}
