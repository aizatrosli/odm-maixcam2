#include "gfx.h"

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static inline void put(gfx_canvas *c, int x, int y, gfx_color col)
{
    if ((unsigned)x >= (unsigned)c->w || (unsigned)y >= (unsigned)c->h) return;
    uint8_t *p = c->px + (y * c->w + x) * 3;
    p[0] = col >> 16;
    p[1] = col >> 8;
    p[2] = col;
}

static inline void blend(gfx_canvas *c, int x, int y, gfx_color col, int a)
{
    if (a <= 0 || (unsigned)x >= (unsigned)c->w || (unsigned)y >= (unsigned)c->h) return;
    uint8_t *p = c->px + (y * c->w + x) * 3;
    if (a >= 255) {
        p[0] = col >> 16;
        p[1] = col >> 8;
        p[2] = col;
        return;
    }
    p[0] = (uint8_t)((p[0] * (255 - a) + ((col >> 16) & 255) * a) / 255);
    p[1] = (uint8_t)((p[1] * (255 - a) + ((col >> 8) & 255) * a) / 255);
    p[2] = (uint8_t)((p[2] * (255 - a) + (col & 255) * a) / 255);
}

gfx_color gfx_mix(gfx_color a, gfx_color b, int t)
{
    if (t < 0) t = 0;
    if (t > 255) t = 255;
    unsigned r = (((a >> 16) & 255) * (255 - t) + ((b >> 16) & 255) * t) / 255;
    unsigned g = (((a >> 8) & 255) * (255 - t) + ((b >> 8) & 255) * t) / 255;
    unsigned bl = ((a & 255) * (255 - t) + (b & 255) * t) / 255;
    return (r << 16) | (g << 8) | bl;
}

void gfx_clear(gfx_canvas *c, gfx_color col)
{
    gfx_fill_rect(c, 0, 0, c->w, c->h, col);
}

void gfx_fill_rect(gfx_canvas *c, int x, int y, int w, int h, gfx_color col)
{
    if (x < 0) { w += x; x = 0; }
    if (y < 0) { h += y; y = 0; }
    if (x + w > c->w) w = c->w - x;
    if (y + h > c->h) h = c->h - y;
    if (w <= 0 || h <= 0) return;
    uint8_t r = col >> 16, g = col >> 8, b = col;
    for (int j = 0; j < h; j++) {
        uint8_t *p = c->px + ((y + j) * c->w + x) * 3;
        for (int i = 0; i < w; i++) {
            *p++ = r;
            *p++ = g;
            *p++ = b;
        }
    }
}

void gfx_rect(gfx_canvas *c, int x, int y, int w, int h, gfx_color col)
{
    gfx_fill_rect(c, x, y, w, 1, col);
    gfx_fill_rect(c, x, y + h - 1, w, 1, col);
    gfx_fill_rect(c, x, y, 1, h, col);
    gfx_fill_rect(c, x + w - 1, y, 1, h, col);
}

void gfx_round_rect(gfx_canvas *c, int x, int y, int w, int h, int r, gfx_color col)
{
    if (r * 2 > h) r = h / 2;
    if (r * 2 > w) r = w / 2;
    gfx_fill_rect(c, x, y + r, w, h - 2 * r, col);
    for (int j = 0; j < r; j++) {
        /* row j of a radius-r corner: skip the pixels outside the arc */
        int dy = r - j, dx = 0;
        while (dx < r && dx * dx + dy * dy <= r * r) dx++;
        int k = r - dx;
        gfx_fill_rect(c, x + k, y + j, w - 2 * k, 1, col);
        gfx_fill_rect(c, x + k, y + h - 1 - j, w - 2 * k, 1, col);
    }
}

void gfx_line(gfx_canvas *c, int x0, int y0, int x1, int y1, gfx_color col)
{
    int dx = abs(x1 - x0), sx = x0 < x1 ? 1 : -1;
    int dy = -abs(y1 - y0), sy = y0 < y1 ? 1 : -1;
    int err = dx + dy;
    for (;;) {
        put(c, x0, y0, col);
        if (x0 == x1 && y0 == y1) break;
        int e2 = 2 * err;
        if (e2 >= dy) { err += dy; x0 += sx; }
        if (e2 <= dx) { err += dx; y0 += sy; }
    }
}

void gfx_thick_line(gfx_canvas *c, int x0, int y0, int x1, int y1, int t, gfx_color col)
{
    int dx = abs(x1 - x0), sx = x0 < x1 ? 1 : -1;
    int dy = -abs(y1 - y0), sy = y0 < y1 ? 1 : -1;
    int err = dx + dy, r = t / 2;
    for (;;) {
        gfx_fill_rect(c, x0 - r, y0 - r, t, t, col);
        if (x0 == x1 && y0 == y1) break;
        int e2 = 2 * err;
        if (e2 >= dy) { err += dy; x0 += sx; }
        if (e2 <= dx) { err += dx; y0 += sy; }
    }
}

void gfx_fill_circle(gfx_canvas *c, int cx, int cy, int r, gfx_color col)
{
    /* 4x4 supersampled edge for a smooth rim */
    for (int y = -r - 1; y <= r + 1; y++) {
        for (int x = -r - 1; x <= r + 1; x++) {
            int cov = 0;
            for (int sy = 0; sy < 4; sy++)
                for (int sx = 0; sx < 4; sx++) {
                    float fx = x + (sx + 0.5f) / 4 - 0.5f, fy = y + (sy + 0.5f) / 4 - 0.5f;
                    if (fx * fx + fy * fy <= (float)r * r) cov++;
                }
            if (cov) blend(c, cx + x, cy + y, col, cov * 255 / 16);
        }
    }
}

void gfx_circle(gfx_canvas *c, int cx, int cy, int r, gfx_color col)
{
    int x = r, y = 0, err = 1 - r;
    while (x >= y) {
        put(c, cx + x, cy + y, col); put(c, cx + y, cy + x, col);
        put(c, cx - y, cy + x, col); put(c, cx - x, cy + y, col);
        put(c, cx - x, cy - y, col); put(c, cx - y, cy - x, col);
        put(c, cx + y, cy - x, col); put(c, cx + x, cy - y, col);
        y++;
        if (err < 0) err += 2 * y + 1;
        else { x--; err += 2 * (y - x) + 1; }
    }
}

int gfx_text_width(const gfx_font *f, const char *s)
{
    return (int)strlen(s) * f->adv;
}

int gfx_text(gfx_canvas *c, const gfx_font *f, int x, int y, gfx_color col, int align, const char *s)
{
    int w = gfx_text_width(f, s);
    if (align == GFX_ALIGN_RIGHT) x -= w;
    else if (align == GFX_ALIGN_CENTER) x -= w / 2;
    int gsize = f->adv * f->h;
    for (; *s; s++, x += f->adv) {
        int ch = (unsigned char)*s;
        if (ch < f->first || ch > f->last) ch = '?';
        if (ch == ' ') continue;
        const unsigned char *g = f->bits + (ch - f->first) * gsize;
        for (int j = 0; j < f->h; j++)
            for (int i = 0; i < f->adv; i++) blend(c, x + i, y + j, col, g[j * f->adv + i]);
    }
    return w;
}

int gfx_textf(gfx_canvas *c, const gfx_font *f, int x, int y, gfx_color col, int align, const char *fmt, ...)
{
    char buf[256];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf, sizeof buf, fmt, ap);
    va_end(ap);
    return gfx_text(c, f, x, y, col, align, buf);
}

void gfx_blit_scaled(gfx_canvas *c, int x, int y, int w, int h, const uint8_t *src, int sw, int sh)
{
    if (!src || sw <= 0 || sh <= 0) return;
    for (int j = 0; j < h; j++) {
        int yy = y + j;
        if ((unsigned)yy >= (unsigned)c->h) continue;
        const uint8_t *row = src + (j * sh / h) * sw * 3;
        uint8_t *p = c->px + (yy * c->w) * 3;
        for (int i = 0; i < w; i++) {
            int xx = x + i;
            if ((unsigned)xx >= (unsigned)c->w) continue;
            const uint8_t *s = row + (i * sw / w) * 3;
            p[xx * 3] = s[0];
            p[xx * 3 + 1] = s[1];
            p[xx * 3 + 2] = s[2];
        }
    }
}

int gfx_save_ppm(const gfx_canvas *c, const char *path)
{
    FILE *f = fopen(path, "wb");
    if (!f) return -1;
    fprintf(f, "P6\n%d %d\n255\n", c->w, c->h);
    size_t n = fwrite(c->px, 3, (size_t)c->w * c->h, f);
    fclose(f);
    return n == (size_t)c->w * c->h ? 0 : -1;
}
