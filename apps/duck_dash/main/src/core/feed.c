/* walk.py's telemetry feed (odm_maix/telemetry.py): a small text file in
 * /dev/shm rewritten 10 times a second while the walk loop runs. While it is
 * fresh, the walk loop owns the servo bus, the IMU and the feet, so the
 * dashboard shows the loop's own readings instead of opening the hardware. */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "dash.h"

/* Up to n numbers. A NaN, an infinity or a value past +-1e6 ends the list, so
 * a broken writer can never put a non-finite number on the screen. */
static int floats(const char *p, float *out, int n)
{
    int i = 0;
    char *e;
    for (; i < n; i++) {
        float v = strtof(p, &e);
        if (e == p || !isfinite(v) || fabsf(v) > 1e6f) break;
        out[i] = v;
        p = e;
    }
    return i;
}

/* one number, 0 when it is missing or not finite */
static float num(const char *p)
{
    float v;
    return floats(p, &v, 1) == 1 ? v : 0;
}

/* one integer clamped to [lo, hi] */
static int inum(const char *p, int lo, int hi)
{
    float v = num(p);
    return v < lo ? lo : v > hi ? hi : (int)v;
}

/* exactly n numbers, each in [lo, hi], or nothing */
static int floats_in(const char *p, float *out, int n, float lo, float hi)
{
    float tmp[DASH_NJ];
    if (n > DASH_NJ || floats(p, tmp, n) != n) return 0;
    for (int i = 0; i < n; i++)
        if (tmp[i] < lo || tmp[i] > hi) return 0;
    memcpy(out, tmp, sizeof(float) * n);
    return n;
}

static void word(const char *p, char *out, int n)
{
    while (*p == ' ') p++;
    dash_strlcpy(out, p, n);
    out[strcspn(out, "\r\n")] = 0;
}

int dash_feed_read(const char *path, double max_age_s, dash_snapshot *s, const dash_config *cfg)
{
    FILE *f = fopen(path, "r");
    if (!f) return 0;
    char line[512];
    double stamp = 0;
    int ok = 0;
    float pos[DASH_NJ], vel[DASH_NJ], v[8];
    float present[DASH_NJ], load[DASH_NJ], temps[DASH_NJ], svolt[DASH_NJ], errs[DASH_NJ];
    int npos = 0, nvel = 0, npresent = 0, nload = 0, ntemp = 0, nsvolt = 0, nerr = 0;
    float volt = 0, temp = 0;
    int bus_err = -1;
    dash_snapshot n = *s;
    n.walk_state[0] = n.walk_mode[0] = 0;
    n.screen_busy = 0;
    n.det_fps = 0;
    n.policy_steps = -1;
    /* what the feed does not say is unknown, not the last probe's reading */
    memset(n.acc, 0, sizeof n.acc);
    memset(n.gyro, 0, sizeof n.gyro);
    n.foot[0] = n.foot[1] = -1;
    int have_feet = 0;
    while (fgets(line, sizeof line, f)) {
        size_t len = strlen(line);
        if (len == sizeof line - 1 && line[len - 1] != '\n') { /* an over-long line: skip the rest of it */
            int ch;
            while ((ch = fgetc(f)) != EOF && ch != '\n') {}
            continue;
        }
        char *val = strchr(line, ' ');
        if (!val) continue;
        *val++ = 0;
        if (!strcmp(line, "duck_telemetry")) ok = 1;
        else if (!strcmp(line, "time")) stamp = atof(val);
        else if (!strcmp(line, "mode")) word(val, n.walk_mode, sizeof n.walk_mode);
        else if (!strcmp(line, "state")) word(val, n.walk_state, sizeof n.walk_state);
        else if (!strcmp(line, "paused")) n.paused = inum(val, 0, 1);
        else if (!strcmp(line, "backend")) word(val, n.backend, sizeof n.backend);
        else if (!strcmp(line, "loop_ms") && floats_in(val, v, 2, 0, 1e4f) == 2) {
            n.loop_mean_ms = v[0];
            n.loop_p99_ms = v[1];
        } else if (!strcmp(line, "policy_ms") && floats_in(val, v, 1, 0, 1e4f) == 1) n.policy_ms = v[0];
        else if (!strcmp(line, "overruns")) n.overruns = inum(val, 0, 99999999);
        else if (!strcmp(line, "steps")) n.policy_steps = inum(val, 0, 2000000000);
        else if (!strcmp(line, "cmd")) floats_in(val, n.cmd, 7, -100, 100);
        else if (!strcmp(line, "pos")) npos = floats_in(val, pos, DASH_NJ, -10, 10);
        else if (!strcmp(line, "vel")) nvel = floats_in(val, vel, DASH_NJ, -1000, 1000);
        else if (!strcmp(line, "present")) npresent = floats_in(val, present, DASH_NJ, 0, 1);
        else if (!strcmp(line, "load")) nload = floats_in(val, load, DASH_NJ, -1000, 1000);
        else if (!strcmp(line, "temp")) ntemp = floats_in(val, temps, DASH_NJ, -60, 255);
        else if (!strcmp(line, "svolt")) nsvolt = floats_in(val, svolt, DASH_NJ, 0, 25.5f);
        else if (!strcmp(line, "err")) nerr = floats_in(val, errs, DASH_NJ, 0, 255);
        else if (!strcmp(line, "gyro")) floats_in(val, n.gyro, 3, -1000, 1000);
        else if (!strcmp(line, "acc")) floats_in(val, n.acc, 3, -1000, 1000);
        else if (!strcmp(line, "feet") && floats_in(val, v, 2, -1e6f, 1e6f) == 2) {
            n.foot[0] = v[0] > 0.5f;
            n.foot[1] = v[1] > 0.5f;
            have_feet = 1;
        } else if (!strcmp(line, "volt")) volt = num(val);
        else if (!strcmp(line, "temp_max")) temp = num(val);
        else if (!strcmp(line, "bus_errors")) bus_err = inum(val, -1, 2000000000);
        else if (!strcmp(line, "det_fps")) n.det_fps = num(val);
        else if (!strcmp(line, "screen")) n.screen_busy = inum(val, 0, 1);
    }
    fclose(f);
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    double age = ts.tv_sec + ts.tv_nsec / 1e9 - stamp;
    /* walk.py announces itself before it turns the servos on (~5 s), so a
     * "starting" feed stays valid longer: the dashboard lets go of the feet
     * GPIOs and the camera before walk.py asks for them */
    if (!strcmp(n.walk_state, "starting")) max_age_s = 15;
    if (!ok || age > max_age_s || age < -5) return 0;

    n.walk_live = 1;
    n.source = SRC_FEED;
    n.bus_st = ST_OK;
    snprintf(n.bus_msg, sizeof n.bus_msg, "owned by walk loop");
    if (npos == DASH_NJ) {
        /* load / temp / svolt / err / present are optional (walk.py does not
         * publish them; tools/duck_stress.py does): 0 = not known */
        char missing[DASH_STR] = "";
        int found = 0, errbits = 0;
        float vmin = 99, vmax = 0;
        for (int i = 0; i < DASH_NJ; i++) {
            dash_servo *sv = &n.servo[i];
            sv->present = npresent == DASH_NJ ? present[i] > 0.5f : 1;
            sv->pos = pos[i];
            sv->vel = nvel == DASH_NJ ? vel[i] : 0;
            sv->load = nload == DASH_NJ ? load[i] : 0;
            sv->volt = nsvolt == DASH_NJ ? svolt[i] : 0;
            sv->temp = ntemp == DASH_NJ ? (int)temps[i] : 0;
            sv->err = nerr == DASH_NJ ? (int)errs[i] : 0;
            sv->lo = cfg->joint_lo[i];
            sv->hi = cfg->joint_hi[i];
            if (sv->present) {
                found++;
                errbits |= sv->err;
                if (sv->temp > temp) temp = (float)sv->temp;
                if (sv->volt > 0 && sv->volt < vmin) vmin = sv->volt;
                if (sv->volt > vmax) vmax = sv->volt;
            } else {
                char id[8];
                snprintf(id, sizeof id, "%s%d", missing[0] ? " " : "", DASH_JOINT_IDS[i]);
                if (strlen(missing) + strlen(id) < sizeof missing - 4) strcat(missing, id);
            }
        }
        n.servos_found = found;
        n.servo_volt_min = vmax > 0 ? vmin : 0;
        n.servo_volt_max = vmax;
        n.bus_sweep_ms = 0;
        if (found == 0) {
            n.bus_st = ST_FAIL;
            snprintf(n.bus_msg, sizeof n.bus_msg, "walk loop: no servo answers");
        } else if (found < DASH_NJ) {
            n.bus_st = ST_FAIL;
            snprintf(n.bus_msg, sizeof n.bus_msg, "walk loop: missing ID %.41s", missing);
        } else if (errbits || temp >= 55) {
            n.bus_st = ST_WARN;
            snprintf(n.bus_msg, sizeof n.bus_msg, "%s", temp >= 55 ? "a servo runs hot" : "servo status error");
        }
    } else {
        n.servos_found = 0;
        n.bus_st = ST_FAIL;
        snprintf(n.bus_msg, sizeof n.bus_msg, "walk loop has no servo readings");
    }
    if (bus_err > 0) {
        n.bus_errors = bus_err;
        if (n.bus_st == ST_OK) n.bus_st = ST_WARN;
    }
    if (volt < 0 || volt > 30) volt = 0;
    n.batt_v = volt;
    n.batt_pct = volt > 0 ? (int)dash_batt_pct(volt) : -1;
    n.batt_st = dash_batt_status(volt);
    n.servo_temp_max = temp < -60 ? -60 : temp > 255 ? 255 : (int)temp;
    if (n.det_fps < 0 || n.det_fps > 1000) n.det_fps = 0;

    float g = sqrtf(n.acc[0] * n.acc[0] + n.acc[1] * n.acc[1] + n.acc[2] * n.acc[2]);
    if (g > 1) {
        n.imu_st = ST_OK;
        n.roll = atan2f(n.acc[1], n.acc[2]) * 57.2958f;
        n.pitch = atan2f(-n.acc[0], sqrtf(n.acc[1] * n.acc[1] + n.acc[2] * n.acc[2])) * 57.2958f;
        snprintf(n.imu_msg, sizeof n.imu_msg, "from walk loop");
    } else {
        n.imu_st = ST_WARN;
        snprintf(n.imu_msg, sizeof n.imu_msg, "walk loop has no IMU data yet");
    }
    n.feet_st = have_feet ? ST_OK : ST_WARN;
    snprintf(n.feet_msg, sizeof n.feet_msg, "%s", have_feet ? "from walk loop" : "walk loop has no foot readings");
    n.npu_st = n.policy_ms > 0 ? (n.policy_ms < 10 ? ST_OK : ST_WARN) : ST_UNKNOWN;
    snprintf(n.npu_msg, sizeof n.npu_msg, "%s %.2f ms/step", n.backend, n.policy_ms);
    *s = n;
    return 1;
}
