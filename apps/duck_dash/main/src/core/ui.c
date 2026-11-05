/* Screen layout, drawn for 640x480 (the MaixCAM2 panel). Page 0 is the
 * overview, page 1 the servo table. Tap the screen to switch. */
#include <math.h>
#include <stdio.h>
#include <string.h>

#include "dash.h"
#include "gfx.h"

#define BG 0x0D1015
#define PANEL 0x171C24
#define EDGE 0x262E3A
#define TEXT 0xE8ECF2
#define DIM 0x8B95A5
#define FAINT 0x4A5361
#define C_OK 0x3DD68C
#define C_WARN 0xF5B83D
#define C_FAIL 0xF0524F
#define C_OFF 0x5A6372
#define ACCENT 0x4EA8FF
#define SKY 0x2B5C8A
#define GROUND 0x6B4A2B

static gfx_color st_color(dash_status st)
{
    switch (st) {
    case ST_OK: return C_OK;
    case ST_WARN: return C_WARN;
    case ST_FAIL: return C_FAIL;
    case ST_OFF: return C_OFF;
    default: return FAINT;
    }
}

static const char *st_word(dash_status st)
{
    switch (st) {
    case ST_OK: return "OK";
    case ST_WARN: return "CHECK";
    case ST_FAIL: return "MISSING";
    case ST_OFF: return "OFF";
    default: return "...";
    }
}

static void panel(gfx_canvas *c, int x, int y, int w, int h, const char *title)
{
    gfx_round_rect(c, x, y, w, h, 6, EDGE);
    gfx_round_rect(c, x + 1, y + 1, w - 2, h - 2, 5, PANEL);
    if (title) gfx_text(c, &FONT_SMALL, x + 8, y + 4, DIM, GFX_ALIGN_LEFT, title);
}

static void dot(gfx_canvas *c, int x, int y, dash_status st)
{
    gfx_fill_circle(c, x, y, 5, st_color(st));
}

/* text clipped to max_chars, with an ellipsis */
static void text_clip(gfx_canvas *c, const gfx_font *f, int x, int y, gfx_color col, int max_chars, const char *s)
{
    char buf[128];
    int n = (int)strlen(s);
    if (n <= max_chars || max_chars < 3) {
        gfx_text(c, f, x, y, col, GFX_ALIGN_LEFT, s);
        return;
    }
    snprintf(buf, sizeof buf, "%.*s..", max_chars - 2, s);
    gfx_text(c, f, x, y, col, GFX_ALIGN_LEFT, buf);
}

/* one line of the status column: dot, label, value on the right, message below */
static void status_row(gfx_canvas *c, int x, int y, int w, dash_status st, const char *label, const char *value,
                       const char *msg)
{
    dot(c, x + 8, y + 9, st);
    gfx_text(c, &FONT_SMALL, x + 20, y + 1, TEXT, GFX_ALIGN_LEFT, label);
    gfx_text(c, &FONT_SMALL, x + w - 4, y + 1, st == ST_OK ? TEXT : st_color(st), GFX_ALIGN_RIGHT, value);
    text_clip(c, &FONT_SMALL, x + 20, y + 16, DIM, (w - 24) / FONT_SMALL.adv, msg);
}

static void bar(gfx_canvas *c, int x, int y, int w, int h, float frac, gfx_color col)
{
    if (frac < 0) frac = 0;
    if (frac > 1) frac = 1;
    gfx_fill_rect(c, x, y, w, h, EDGE);
    gfx_fill_rect(c, x, y, (int)(w * frac + 0.5f), h, col);
}

static const char *src_name(const dash_snapshot *s)
{
    if (s->source == SRC_SIM) return s->walk_live ? "SIM + WALK" : "SIMULATED";
    if (s->walk_live) return "WALK LOOP";
    return "LIVE PROBE";
}

static void header(gfx_canvas *c, const dash_snapshot *s, int page)
{
    gfx_fill_rect(c, 0, 0, c->w, 30, 0x12161D);
    gfx_fill_rect(c, 0, 30, c->w, 1, EDGE);
    gfx_text(c, &FONT_MEDIUM, 10, 4, TEXT, GFX_ALIGN_LEFT, "DUCK");
    gfx_text(c, &FONT_MEDIUM, 54, 4, ACCENT, GFX_ALIGN_LEFT, "TELEMETRY");
    /* source pill */
    const char *src = src_name(s);
    int pw = gfx_text_width(&FONT_SMALL, src) + 16, px = 160;
    gfx_color pc = s->source == SRC_SIM ? C_WARN : s->walk_live ? C_OK : ACCENT;
    gfx_round_rect(c, px, 6, pw, 19, 9, gfx_mix(PANEL, pc, 70));
    gfx_text(c, &FONT_SMALL, px + 8, 7, pc, GFX_ALIGN_LEFT, src);

    /* overall verdict */
    dash_status parts[] = {s->bus_st, s->batt_st, s->imu_st, s->feet_st, s->npu_st, s->cam_st == ST_FAIL ? ST_FAIL : ST_OK};
    int fail = 0, warn = 0;
    for (unsigned i = 0; i < sizeof parts / sizeof parts[0]; i++) {
        fail += parts[i] == ST_FAIL;
        warn += parts[i] == ST_WARN || parts[i] == ST_UNKNOWN;
    }
    char v[48];
    gfx_color vc;
    if (fail) {
        snprintf(v, sizeof v, "%d MISSING", fail);
        vc = C_FAIL;
    } else if (warn) {
        snprintf(v, sizeof v, "%d TO CHECK", warn);
        vc = C_WARN;
    } else {
        snprintf(v, sizeof v, "ALL CONNECTED");
        vc = C_OK;
    }
    int vx = px + pw + 10;
    gfx_fill_circle(c, vx + 6, 15, 6, vc);
    gfx_text(c, &FONT_SMALL, vx + 16, 7, vc, GFX_ALIGN_LEFT, v);

    /* page dots + ip */
    for (int i = 0; i < dash_ui_pages(); i++) gfx_fill_circle(c, c->w - 14 - (dash_ui_pages() - 1 - i) * 12, 15, 3, i == page ? TEXT : FAINT);
    gfx_text(c, &FONT_SMALL, c->w - 44, 7, DIM, GFX_ALIGN_RIGHT, s->ip[0] ? s->ip : "no network");
}

static float deg(float rad) { return rad * 57.2958f; }

/* ------------------------------------------------------------- page 0 */

static void status_column(gfx_canvas *c, const dash_snapshot *s, int x, int y, int w)
{
    char val[32], msg[DASH_STR + 16];
    int rh = 34;

    snprintf(val, sizeof val, "%d/%d", s->servos_found, DASH_NJ);
    status_row(c, x, y, w, s->bus_st, "Servos", val, s->bus_msg);
    y += rh;

    if (s->batt_v > 0) snprintf(val, sizeof val, "%.2fV %d%%", s->batt_v, s->batt_pct);
    else snprintf(val, sizeof val, "--");
    snprintf(msg, sizeof msg, "%s", s->batt_v > 0 ? "2S pack, via servos" : "read from the servos");
    status_row(c, x, y, w, s->batt_st, "Battery", val, msg);
    y += rh;

    status_row(c, x, y, w, s->imu_st, "IMU", s->imu_st == ST_OK ? "BNO055" : st_word(s->imu_st), s->imu_msg);
    y += rh;

    snprintf(val, sizeof val, "L:%s R:%s", s->foot[0] < 0 ? "?" : s->foot[0] ? "gnd" : "air",
             s->foot[1] < 0 ? "?" : s->foot[1] ? "gnd" : "air");
    status_row(c, x, y, w, s->feet_st, "Feet", s->feet_st == ST_OK ? val : st_word(s->feet_st), s->feet_msg);
    y += rh;

    if (s->cam_st == ST_OK) snprintf(val, sizeof val, "%.0f fps", s->cam_fps);
    else dash_strlcpy(val, st_word(s->cam_st), sizeof val);
    status_row(c, x, y, w, s->cam_st, "Camera", val, s->cam_msg);
    y += rh;

    snprintf(val, sizeof val, "%s", s->backend[0] ? s->backend : "?");
    status_row(c, x, y, w, s->npu_st, "Policy", val, s->npu_msg);
    y += rh;

    status_row(c, x, y, w, s->vis_st, "Detector", st_word(s->vis_st), s->vis_msg);
    y += rh;

    status_row(c, x, y, w, s->pad_st, "Gamepad", s->pad_st == ST_OK ? "paired" : "none",
               s->pad_st == ST_OK ? s->pad_name : "BLE pad not connected");
    y += rh;

    if (s->wifi_dbm) snprintf(val, sizeof val, "%d dBm", s->wifi_dbm);
    else dash_strlcpy(val, s->wifi_st == ST_OK ? "up" : "down", sizeof val);
    status_row(c, x, y, w, s->wifi_st, "WiFi", val, s->ip[0] ? s->ip : "no IP address");
    y += rh;

    snprintf(val, sizeof val, "%d card%s", s->audio_cards, s->audio_cards == 1 ? "" : "s");
    status_row(c, x, y, w, s->audio_st, "Audio", val, "onboard speaker + mics");
    y += rh;

    status_row(c, x, y, w, s->config_loaded ? ST_OK : ST_WARN, "Config", s->config_loaded ? "loaded" : "defaults",
               s->config_path);
}

static void board_panel(gfx_canvas *c, const dash_snapshot *s, int x, int y, int w, int h)
{
    panel(c, x, y, w, h, "BOARD");
    int yy = y + 22;
    gfx_textf(c, &FONT_SMALL, x + 8, yy, TEXT, GFX_ALIGN_LEFT, "CPU %3.0f%%", s->cpu_pct[0]);
    bar(c, x + 84, yy + 5, w - 92, 7, s->cpu_pct[0] / 100, s->cpu_pct[0] > 85 ? C_WARN : ACCENT);
    yy += 18;
    gfx_textf(c, &FONT_SMALL, x + 8, yy, s->soc_temp > 70 ? C_WARN : TEXT, GFX_ALIGN_LEFT, "SoC %.0fC", s->soc_temp);
    gfx_textf(c, &FONT_SMALL, x + w - 8, yy, TEXT, GFX_ALIGN_RIGHT, "RAM %dM free", s->mem_avail_mb);
}

static void servo_panel(gfx_canvas *c, const dash_snapshot *s, int x, int y, int w, int h)
{
    panel(c, x, y, w, h, "SERVOS  deg");
    gfx_text(c, &FONT_SMALL, x + w - 8, y + 4, DIM, GFX_ALIGN_RIGHT, "C");
    int ry = y + 24, rh = 25;
    for (int i = 0; i < DASH_NJ; i++) {
        const dash_servo *sv = &s->servo[i];
        if (i == 5 || i == 9) ry += 4; /* left leg | head | right leg */
        dash_status st = !sv->present ? (s->bus_st == ST_OFF ? ST_OFF : ST_FAIL)
                         : (sv->err || sv->temp >= 65) ? ST_FAIL
                         : sv->temp >= 55 ? ST_WARN : ST_OK;
        gfx_fill_circle(c, x + 10, ry + 8, 4, st_color(st));
        gfx_textf(c, &FONT_SMALL, x + 18, ry, sv->present ? TEXT : FAINT, GFX_ALIGN_LEFT, "%-2d", DASH_JOINT_IDS[i]);
        text_clip(c, &FONT_SMALL, x + 38, ry, sv->present ? DIM : FAINT, 10, DASH_JOINT_SHORT[i]);
        if (!sv->present) {
            gfx_text(c, &FONT_SMALL, x + w - 8, ry, s->bus_st == ST_OFF ? FAINT : C_FAIL, GFX_ALIGN_RIGHT, "no answer");
        } else {
            gfx_textf(c, &FONT_SMALL, x + w - 34, ry, TEXT, GFX_ALIGN_RIGHT, "%+.0f", deg(sv->pos));
            if (sv->temp) gfx_textf(c, &FONT_SMALL, x + w - 8, ry, st_color(st), GFX_ALIGN_RIGHT, "%d", sv->temp);
            else gfx_text(c, &FONT_SMALL, x + w - 8, ry, FAINT, GFX_ALIGN_RIGHT, "-");
            /* position within its range (calibrated soft limits, else +-90 deg) as a thin track under the row */
            float lo = sv->hi > sv->lo ? sv->lo : -1.5708f, hi = sv->hi > sv->lo ? sv->hi : 1.5708f;
            int bx = x + 38, bw = w - 46;
            gfx_fill_rect(c, bx, ry + 18, bw, 2, EDGE);
            float f = (sv->pos - lo) / (hi - lo);
            gfx_color mc = (f < 0.03f || f > 0.97f) ? C_WARN : ACCENT;
            if (f < 0) f = 0;
            if (f > 1) f = 1;
            gfx_fill_rect(c, bx + (int)(f * (bw - 10)), ry + 17, 10, 4, mc);
        }
        ry += rh;
    }
    ry += 2;
    gfx_fill_rect(c, x + 8, ry, w - 16, 1, EDGE);
    ry += 5;
    if (s->walk_live && s->source != SRC_SIM)
        gfx_textf(c, &FONT_SMALL, x + 8, ry, DIM, GFX_ALIGN_LEFT, "hottest %dC  err %u", s->servo_temp_max, s->bus_errors);
    else
        gfx_textf(c, &FONT_SMALL, x + 8, ry, DIM, GFX_ALIGN_LEFT, "sweep %.1fms err %u  %dC", s->bus_sweep_ms, s->bus_errors,
                  s->servo_temp_max);
}

static void attitude(gfx_canvas *c, const dash_snapshot *s, int x, int y, int w, int h)
{
    panel(c, x, y, w, h, "IMU");
    int r = 52, cx = x + 8 + r, cy = y + 24 + r;
    /* artificial horizon: sky/ground split by a line tilted by roll, shifted by pitch */
    float ro = s->roll / 57.2958f, pi = s->pitch;
    float shift = pi / 45.0f * r;
    float sn = sinf(ro), cs = cosf(ro);
    for (int j = -r; j <= r; j++)
        for (int i = -r; i <= r; i++) {
            if (i * i + j * j > r * r) continue;
            /* signed distance from the horizon line through (0, shift) rotated by roll */
            float d = -i * sn + j * cs - shift;
            gfx_color col = s->imu_st == ST_OK || s->imu_st == ST_WARN ? (d < 0 ? SKY : GROUND) : EDGE;
            gfx_fill_rect(c, cx + i, cy + j, 1, 1, col);
        }
    gfx_circle(c, cx, cy, r, EDGE);
    gfx_thick_line(c, cx - 22, cy, cx - 8, cy, 3, C_WARN);
    gfx_thick_line(c, cx + 8, cy, cx + 22, cy, 3, C_WARN);
    gfx_fill_circle(c, cx, cy, 3, C_WARN);

    int tx = cx + r + 10, ty = y + 24;
    if (s->imu_st == ST_OK || s->imu_st == ST_WARN) {
        gfx_textf(c, &FONT_SMALL, tx, ty, TEXT, GFX_ALIGN_LEFT, "roll %+5.1f", s->roll);
        gfx_textf(c, &FONT_SMALL, tx, ty + 17, TEXT, GFX_ALIGN_LEFT, "pitch%+5.1f", s->pitch);
        gfx_text(c, &FONT_SMALL, tx, ty + 38, DIM, GFX_ALIGN_LEFT, "gyro deg/s");
        for (int k = 0; k < 3; k++)
            gfx_textf(c, &FONT_SMALL, tx, ty + 55 + 16 * k, TEXT, GFX_ALIGN_LEFT, "%c %+6.1f", "xyz"[k], deg(s->gyro[k]));
    } else {
        gfx_text(c, &FONT_SMALL, tx, ty + 30, C_FAIL, GFX_ALIGN_LEFT, "no IMU");
    }
    int by = y + h - 20;
    float g = sqrtf(s->acc[0] * s->acc[0] + s->acc[1] * s->acc[1] + s->acc[2] * s->acc[2]);
    if (s->calib[0] >= 0 && s->source != SRC_FEED)
        gfx_textf(c, &FONT_SMALL, x + 8, by, DIM, GFX_ALIGN_LEFT, "|g| %.2f  cal S%d G%d A%d M%d", g, s->calib[0],
                  s->calib[1], s->calib[2], s->calib[3]);
    else
        gfx_textf(c, &FONT_SMALL, x + 8, by, DIM, GFX_ALIGN_LEFT, "|g| %.2f m/s2", g);
}

static void feet(gfx_canvas *c, const dash_snapshot *s, int x, int y, int w, int h)
{
    panel(c, x, y, w, h, "FEET");
    const char *lab[2] = {"LEFT", "RIGHT"};
    for (int i = 0; i < 2; i++) {
        int fx = x + 60 + i * 72, fy = y + 6;
        int v = s->foot[i];
        gfx_color col = v < 0 ? EDGE : v ? C_OK : gfx_mix(PANEL, C_OK, 50);
        gfx_round_rect(c, fx, fy, 62, h - 12, 6, col);
        gfx_text(c, &FONT_SMALL, fx + 31, fy + 2, v == 1 ? BG : v == 0 ? TEXT : FAINT, GFX_ALIGN_CENTER, lab[i]);
        gfx_text(c, &FONT_SMALL, fx + 31, fy + 18, v == 1 ? BG : v == 0 ? DIM : FAINT, GFX_ALIGN_CENTER,
                 v < 0 ? "?" : v ? "ground" : "air");
    }
}

static void camera_or_walk(gfx_canvas *c, const dash_snapshot *s, int x, int y, int w, int h, const uint8_t *cam,
                           int cw, int ch)
{
    if (s->walk_live) {
        panel(c, x, y, w, h, "WALK LOOP");
        int yy = y + 24;
        gfx_textf(c, &FONT_MEDIUM, x + 8, yy, s->paused ? C_WARN : C_OK, GFX_ALIGN_LEFT, "%s%s", s->walk_mode,
                  s->paused ? " (paused)" : "");
        yy += 24;
        text_clip(c, &FONT_SMALL, x + 8, yy, DIM, (w - 16) / 8, s->walk_state[0] ? s->walk_state : "-");
        yy += 20;
        gfx_color lc = s->loop_p99_ms > 20 ? C_FAIL : s->loop_p99_ms > 15 ? C_WARN : TEXT;
        gfx_textf(c, &FONT_SMALL, x + 8, yy, lc, GFX_ALIGN_LEFT, "loop %.1f ms  p99 %.1f", s->loop_mean_ms, s->loop_p99_ms);
        yy += 17;
        bar(c, x + 8, yy + 2, w - 16, 6, s->loop_mean_ms / 20.0f, lc == TEXT ? ACCENT : lc);
        yy += 12;
        gfx_textf(c, &FONT_SMALL, x + 8, yy, TEXT, GFX_ALIGN_LEFT, "policy %.2f ms", s->policy_ms);
        gfx_textf(c, &FONT_SMALL, x + w - 8, yy, s->overruns ? C_WARN : DIM, GFX_ALIGN_RIGHT, "overrun %d", s->overruns);
        yy += 19;
        gfx_textf(c, &FONT_SMALL, x + 8, yy, TEXT, GFX_ALIGN_LEFT, "vx%+.2f vy%+.2f wz%+.2f", s->cmd[0], s->cmd[1], s->cmd[2]);
        yy += 17;
        if (s->det_fps > 0) gfx_textf(c, &FONT_SMALL, x + 8, yy, TEXT, GFX_ALIGN_LEFT, "detector %.1f fps", s->det_fps);
        return;
    }
    panel(c, x, y, w, h, "CAMERA");
    int iw = w - 12, ih = iw * 3 / 4;
    if (ih > h - 26) {
        ih = h - 26;
        iw = ih * 4 / 3;
    }
    int ix = x + (w - iw) / 2, iy = y + 22;
    if (cam && s->cam_st == ST_OK) {
        gfx_blit_scaled(c, ix, iy, iw, ih, cam, cw, ch);
        gfx_textf(c, &FONT_SMALL, x + w - 8, y + 4, DIM, GFX_ALIGN_RIGHT, "%.0f fps", s->cam_fps);
    } else {
        gfx_fill_rect(c, ix, iy, iw, ih, 0x10141A);
        gfx_rect(c, ix, iy, iw, ih, EDGE);
        gfx_text(c, &FONT_SMALL, ix + iw / 2, iy + ih / 2 - 16, st_color(s->cam_st), GFX_ALIGN_CENTER,
                 s->cam_st == ST_OK ? "camera ok" : st_word(s->cam_st));
        text_clip(c, &FONT_SMALL, ix + 8, iy + ih / 2 + 4, DIM, (iw - 16) / 8, s->cam_msg);
    }
}

static void page_overview(gfx_canvas *c, const dash_snapshot *s, const uint8_t *cam, int cw, int ch)
{
    panel(c, 6, 36, 214, 380, NULL);
    status_column(c, s, 8, 42, 210);
    board_panel(c, s, 6, 420, 214, 54);
    servo_panel(c, s, 226, 36, 196, 438);
    attitude(c, s, 428, 36, 206, 152);
    feet(c, s, 428, 192, 206, 46);
    camera_or_walk(c, s, 428, 242, 206, 232, cam, cw, ch);
}

/* ------------------------------------------------------------- page 1 */

static void page_servos(gfx_canvas *c, const dash_snapshot *s)
{
    panel(c, 6, 36, c->w - 12, c->h - 42, NULL);
    static const char *hdr[] = {"ID", "joint", "pos deg", "vel", "load%", "volt", "temp", "status"};
    static const int col[] = {16, 52, 210, 290, 360, 430, 492, 548};
    int y = 44;
    for (int i = 0; i < 8; i++) gfx_text(c, &FONT_SMALL, col[i], y, DIM, GFX_ALIGN_LEFT, hdr[i]);
    y += 22;
    for (int i = 0; i < DASH_NJ; i++) {
        const dash_servo *sv = &s->servo[i];
        if (i % 2) gfx_fill_rect(c, 8, y - 2, c->w - 16, 26, 0x1B212B);
        gfx_textf(c, &FONT_SMALL, col[0], y + 3, TEXT, GFX_ALIGN_LEFT, "%d", DASH_JOINT_IDS[i]);
        gfx_text(c, &FONT_SMALL, col[1], y + 3, sv->present ? TEXT : FAINT, GFX_ALIGN_LEFT, DASH_JOINT_NAMES[i]);
        if (!sv->present) {
            gfx_text(c, &FONT_SMALL, col[7], y + 3, s->bus_st == ST_OFF ? FAINT : C_FAIL, GFX_ALIGN_LEFT, "no answer");
        } else {
            gfx_textf(c, &FONT_SMALL, col[2], y + 3, TEXT, GFX_ALIGN_LEFT, "%+7.1f", deg(sv->pos));
            gfx_textf(c, &FONT_SMALL, col[3], y + 3, TEXT, GFX_ALIGN_LEFT, "%+5.2f", sv->vel);
            if (s->walk_live && s->source == SRC_FEED) {
                gfx_text(c, &FONT_SMALL, col[4], y + 3, FAINT, GFX_ALIGN_LEFT, "  -");
                gfx_text(c, &FONT_SMALL, col[5], y + 3, FAINT, GFX_ALIGN_LEFT, "  -");
                gfx_text(c, &FONT_SMALL, col[6], y + 3, FAINT, GFX_ALIGN_LEFT, " -");
            } else {
                gfx_textf(c, &FONT_SMALL, col[4], y + 3, fabsf(sv->load) > 80 ? C_WARN : TEXT, GFX_ALIGN_LEFT, "%+5.0f", sv->load);
                gfx_textf(c, &FONT_SMALL, col[5], y + 3, dash_batt_status(sv->volt) == ST_OK ? TEXT : C_WARN, GFX_ALIGN_LEFT,
                          "%.1f", sv->volt);
                gfx_textf(c, &FONT_SMALL, col[6], y + 3, sv->temp >= 65 ? C_FAIL : sv->temp >= 55 ? C_WARN : TEXT,
                          GFX_ALIGN_LEFT, "%d", sv->temp);
            }
            char e[40] = "ok";
            if (sv->err) {
                e[0] = 0;
                if (sv->err & 0x01) strcat(e, "volt ");
                if (sv->err & 0x04) strcat(e, "heat ");
                if (sv->err & 0x08) strcat(e, "amps ");
                if (sv->err & 0x20) strcat(e, "load ");
                if (sv->err & 0x02) strcat(e, "sensor ");
            }
            gfx_text(c, &FONT_SMALL, col[7], y + 3, sv->err ? C_FAIL : C_OK, GFX_ALIGN_LEFT, e);
        }
        y += 26;
    }
    y += 6;
    gfx_textf(c, &FONT_SMALL, 16, y, DIM, GFX_ALIGN_LEFT, "port %s   sweep %.1f ms   read errors %u   volt %.1f-%.1f V",
              s->servo_port, s->bus_sweep_ms, s->bus_errors, s->servo_volt_min, s->servo_volt_max);
}

int dash_ui_pages(void) { return 2; }

void dash_ui_draw(gfx_canvas *c, const dash_snapshot *s, int page, const uint8_t *cam, int cw, int ch)
{
    gfx_clear(c, BG);
    header(c, s, page);
    if (page == 1) page_servos(c, s);
    else page_overview(c, s, cam, cw, ch);
}
