/* Tiny RGB888 canvas: rectangles, lines, circles and anti-aliased mono text. */
#ifndef DUCK_DASH_GFX_H
#define DUCK_DASH_GFX_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int adv, h, ascent, first, last;
    const unsigned char *bits; /* (last-first+1) glyphs of adv*h coverage bytes */
} gfx_font;

extern const gfx_font FONT_SMALL, FONT_MEDIUM, FONT_LARGE;

typedef struct {
    uint8_t *px; /* RGB888, row-major, stride = w*3 */
    int w, h;
} gfx_canvas;

typedef uint32_t gfx_color; /* 0xRRGGBB */

#define GFX_ALIGN_LEFT 0
#define GFX_ALIGN_RIGHT 1
#define GFX_ALIGN_CENTER 2

void gfx_clear(gfx_canvas *c, gfx_color col);
void gfx_fill_rect(gfx_canvas *c, int x, int y, int w, int h, gfx_color col);
void gfx_rect(gfx_canvas *c, int x, int y, int w, int h, gfx_color col);
void gfx_round_rect(gfx_canvas *c, int x, int y, int w, int h, int r, gfx_color col);
void gfx_line(gfx_canvas *c, int x0, int y0, int x1, int y1, gfx_color col);
void gfx_thick_line(gfx_canvas *c, int x0, int y0, int x1, int y1, int t, gfx_color col);
void gfx_fill_circle(gfx_canvas *c, int cx, int cy, int r, gfx_color col);
void gfx_circle(gfx_canvas *c, int cx, int cy, int r, gfx_color col);
/* Draws text with its top-left (or right/center anchor) at x,y; returns the width in pixels. */
int gfx_text(gfx_canvas *c, const gfx_font *f, int x, int y, gfx_color col, int align, const char *s);
int gfx_textf(gfx_canvas *c, const gfx_font *f, int x, int y, gfx_color col, int align, const char *fmt, ...)
    __attribute__((format(printf, 7, 8)));
int gfx_text_width(const gfx_font *f, const char *s);
/* Nearest-neighbour blit of an RGB888 image into the box x,y,w,h. */
void gfx_blit_scaled(gfx_canvas *c, int x, int y, int w, int h, const uint8_t *src, int sw, int sh);
gfx_color gfx_mix(gfx_color a, gfx_color b, int t255);
/* Writes a binary PPM (P6). Returns 0 on success. */
int gfx_save_ppm(const gfx_canvas *c, const char *path);

#ifdef __cplusplus
}
#endif
#endif
