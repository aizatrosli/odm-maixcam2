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

static int floats(const char *p, float *out, int n)
{
    int i = 0;
    char *e;
    for (; i < n; i++) {
        float v = strtof(p, &e);
        if (e == p) break;
        out[i] = v;
        p = e;
    }
    return i;
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
    int npos = 0, nvel = 0;
    float volt = 0, temp = 0;
    int bus_err = -1;
    dash_snapshot n = *s;
    n.walk_state[0] = n.walk_mode[0] = 0;
    n.screen_busy = 0;
    n.det_fps = 0;
    while (fgets(line, sizeof line, f)) {
        char *val = strchr(line, ' ');
        if (!val) continue;
        *val++ = 0;
        if (!strcmp(line, "duck_telemetry")) ok = 1;
        else if (!strcmp(line, "time")) stamp = atof(val);
        else if (!strcmp(line, "mode")) word(val, n.walk_mode, sizeof n.walk_mode);
        else if (!strcmp(line, "state")) word(val, n.walk_state, sizeof n.walk_state);
        else if (!strcmp(line, "paused")) n.paused = atoi(val);
        else if (!strcmp(line, "backend")) word(val, n.backend, sizeof n.backend);
        else if (!strcmp(line, "loop_ms") && floats(val, v, 2) == 2) {
            n.loop_mean_ms = v[0];
            n.loop_p99_ms = v[1];
        } else if (!strcmp(line, "policy_ms") && floats(val, v, 1) == 1) n.policy_ms = v[0];
        else if (!strcmp(line, "overruns")) n.overruns = atoi(val);
        else if (!strcmp(line, "cmd")) floats(val, n.cmd, 7);
        else if (!strcmp(line, "pos")) npos = floats(val, pos, DASH_NJ);
        else if (!strcmp(line, "vel")) nvel = floats(val, vel, DASH_NJ);
        else if (!strcmp(line, "gyro")) floats(val, n.gyro, 3);
        else if (!strcmp(line, "acc")) floats(val, n.acc, 3);
        else if (!strcmp(line, "feet") && floats(val, v, 2) == 2) {
            n.foot[0] = v[0] > 0.5f;
            n.foot[1] = v[1] > 0.5f;
        } else if (!strcmp(line, "volt")) volt = (float)atof(val);
        else if (!strcmp(line, "temp_max")) temp = (float)atof(val);
        else if (!strcmp(line, "bus_errors")) bus_err = atoi(val);
        else if (!strcmp(line, "det_fps")) n.det_fps = (float)atof(val);
        else if (!strcmp(line, "screen")) n.screen_busy = atoi(val);
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
        n.servos_found = DASH_NJ;
        for (int i = 0; i < DASH_NJ; i++) {
            n.servo[i].present = 1;
            n.servo[i].pos = pos[i];
            n.servo[i].vel = nvel == DASH_NJ ? vel[i] : 0;
            n.servo[i].load = 0; /* not in the feed: don't show stale probe values */
            n.servo[i].volt = 0;
            n.servo[i].temp = 0;
            n.servo[i].err = 0;
            n.servo[i].lo = cfg->joint_lo[i];
            n.servo[i].hi = cfg->joint_hi[i];
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
    n.batt_v = volt;
    n.batt_pct = volt > 0 ? (int)dash_batt_pct(volt) : -1;
    n.batt_st = dash_batt_status(volt);
    n.servo_temp_max = (int)temp;

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
    n.feet_st = ST_OK;
    snprintf(n.feet_msg, sizeof n.feet_msg, "from walk loop");
    n.npu_st = n.policy_ms > 0 ? (n.policy_ms < 10 ? ST_OK : ST_WARN) : ST_UNKNOWN;
    snprintf(n.npu_msg, sizeof n.npu_msg, "%s %.2f ms/step", n.backend, n.policy_ms);
    *s = n;
    return 1;
}
