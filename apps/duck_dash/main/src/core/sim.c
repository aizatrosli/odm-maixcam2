/* A simulated duck for running the dashboard on a PC (--sim). Faults use the
 * same spellings as odm_maix/diag/simhw.py where they overlap:
 *   missing:23  lowvolt:6.4  hot:24  overload:13  no_imu  no_feet  no_bus
 *   no_camera  no_npu  no_pad  walk (pretend walk.py is running) */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "dash.h"

static const float INIT_POS[DASH_NJ] = {0.002f, 0.053f, -0.63f, 1.368f, -0.784f, 0, 0, 0, 0,
                                        -0.003f, -0.065f, 0.635f, 1.379f, -0.796f};

/* value of fault `name` ("missing:23" -> 23), or -1 / `dflt` */
static int has_fault(const char *faults, const char *name, double *arg)
{
    if (!faults) return 0;
    size_t nl = strlen(name);
    for (const char *p = faults; p && *p;) {
        while (*p == ',' || *p == ' ') p++;
        if (!strncmp(p, name, nl) && (p[nl] == 0 || p[nl] == ',' || p[nl] == ':')) {
            if (arg && p[nl] == ':') *arg = atof(p + nl + 1);
            return 1;
        }
        p = strchr(p, ',');
    }
    return 0;
}

/* every "name:ID" occurrence matching id */
static int fault_for_id(const char *faults, const char *name, int id)
{
    if (!faults) return 0;
    size_t nl = strlen(name);
    for (const char *p = faults; p && *p;) {
        while (*p == ',' || *p == ' ') p++;
        if (!strncmp(p, name, nl) && p[nl] == ':' && atoi(p + nl + 1) == id) return 1;
        p = strchr(p, ',');
    }
    return 0;
}

void dash_sim_fill(dash_snapshot *s, const char *faults, double t)
{
    double a;
    s->source = SRC_SIM;
    s->config_loaded = 1;
    dash_strlcpy(s->config_path, "(simulated duck)", sizeof s->config_path);

    /* servos */
    dash_strlcpy(s->servo_port, "sim:/dev/ttyACM0", sizeof s->servo_port);
    float volt = 7.9f - 0.05f * (float)sin(t * 0.2);
    if (has_fault(faults, "lowvolt", &a)) volt = (float)a;
    int found = 0, hot = 0;
    float vmin = 99, vmax = 0;
    int nobus = has_fault(faults, "no_bus", NULL);
    for (int i = 0; i < DASH_NJ; i++) {
        dash_servo *sv = &s->servo[i];
        int id = DASH_JOINT_IDS[i];
        sv->present = !nobus && !fault_for_id(faults, "missing", id);
        if (!sv->present) continue;
        found++;
        float sway = (float)(0.25 * sin(t * 1.3 + i * 0.7));
        sv->pos = INIT_POS[i] + sway;
        sv->vel = (float)(0.25 * 1.3 * cos(t * 1.3 + i * 0.7));
        sv->load = (float)(12 * sin(t * 0.9 + i));
        sv->volt = volt + (i % 3) * 0.05f;
        sv->temp = 33 + (i * 7) % 9;
        sv->err = 0;
        if (fault_for_id(faults, "hot", id)) sv->temp = 72;
        if (fault_for_id(faults, "overload", id)) sv->err = 0x20;
        sv->lo = INIT_POS[i] - 0.9f;
        sv->hi = INIT_POS[i] + 0.9f;
        if (sv->temp > hot) hot = sv->temp;
        if (sv->volt < vmin) vmin = sv->volt;
        if (sv->volt > vmax) vmax = sv->volt;
    }
    s->servos_found = found;
    s->servo_temp_max = hot;
    s->servo_volt_min = found ? vmin : 0;
    s->servo_volt_max = vmax;
    s->bus_sweep_ms = found ? 0.42f * found : 0;
    if (nobus) {
        s->bus_st = ST_FAIL;
        snprintf(s->bus_msg, sizeof s->bus_msg, "/dev/ttyACM0 not found");
    } else if (found < DASH_NJ) {
        s->bus_st = ST_FAIL;
        snprintf(s->bus_msg, sizeof s->bus_msg, "%d of %d servos answer", found, DASH_NJ);
    } else if (hot >= 55 || has_fault(faults, "overload", NULL)) {
        s->bus_st = ST_WARN;
        snprintf(s->bus_msg, sizeof s->bus_msg, "%s", hot >= 55 ? "a servo runs hot" : "servo status error");
    } else {
        s->bus_st = ST_OK;
        snprintf(s->bus_msg, sizeof s->bus_msg, "all %d servos answer", DASH_NJ);
    }
    s->batt_v = found ? volt : 0;
    s->batt_pct = found ? (int)dash_batt_pct(volt) : -1;
    s->batt_st = dash_batt_status(s->batt_v);

    /* IMU */
    if (has_fault(faults, "no_imu", NULL)) {
        s->imu_st = ST_FAIL;
        snprintf(s->imu_msg, sizeof s->imu_msg, "no BNO055 on i2c-0..7");
        memset(s->acc, 0, sizeof s->acc);
    } else {
        float roll = (float)(6 * sin(t * 0.7)), pitch = (float)(4 * sin(t * 0.5 + 1));
        float r = roll / 57.2958f, p = pitch / 57.2958f;
        s->acc[0] = -9.81f * sinf(p);
        s->acc[1] = 9.81f * sinf(r) * cosf(p);
        s->acc[2] = 9.81f * cosf(r) * cosf(p);
        s->gyro[0] = (float)(0.07 * cos(t * 0.7));
        s->gyro[1] = (float)(0.03 * cos(t * 0.5 + 1));
        s->gyro[2] = 0.002f;
        s->roll = roll;
        s->pitch = pitch;
        s->imu_temp = 29;
        s->calib[0] = 3; s->calib[1] = 3; s->calib[2] = 3; s->calib[3] = 2;
        s->imu_st = ST_OK;
        snprintf(s->imu_msg, sizeof s->imu_msg, "i2c-7 @0x28, NDOF");
    }

    /* feet */
    if (has_fault(faults, "no_feet", NULL)) {
        s->feet_st = ST_FAIL;
        s->foot[0] = s->foot[1] = -1;
        snprintf(s->feet_msg, sizeof s->feet_msg, "GPIOB21/B20 not available");
    } else {
        double ph = fmod(t * 0.8, 1.0);
        s->foot[0] = ph < 0.6;
        s->foot[1] = ph > 0.4;
        s->feet_st = ST_OK;
        snprintf(s->feet_msg, sizeof s->feet_msg, "B21 / B20");
    }

    /* camera, NPU, vision */
    if (has_fault(faults, "no_camera", NULL)) {
        s->cam_st = ST_FAIL;
        snprintf(s->cam_msg, sizeof s->cam_msg, "camera open failed");
        s->cam_fps = 0;
    } else {
        s->cam_st = ST_OK;
        s->cam_fps = 29.8f;
        snprintf(s->cam_msg, sizeof s->cam_msg, "320x240");
    }
    dash_strlcpy(s->backend, "npu", sizeof s->backend);
    if (has_fault(faults, "no_npu", NULL)) {
        s->npu_st = ST_FAIL;
        snprintf(s->npu_msg, sizeof s->npu_msg, "walk_policy.mud not found");
    } else {
        s->npu_st = ST_OK;
        s->npu_load_ms = 41;
        s->npu_infer_ms = 0.31f;
        snprintf(s->npu_msg, sizeof s->npu_msg, "walk_policy.mud %.2f ms", s->npu_infer_ms);
    }
    s->vis_st = ST_OK;
    snprintf(s->vis_msg, sizeof s->vis_msg, "yolo11n.mud");

    /* walk loop */
    s->walk_live = has_fault(faults, "walk", NULL);
    if (s->walk_live) {
        s->source = SRC_SIM;
        dash_strlcpy(s->walk_mode, "interact", sizeof s->walk_mode);
        dash_strlcpy(s->walk_state, "tracking/follow/happy", sizeof s->walk_state);
        s->loop_mean_ms = 6.8f + 0.3f * (float)sin(t);
        s->loop_p99_ms = 9.6f;
        s->policy_ms = 0.31f;
        s->overruns = 0;
        s->cmd[0] = 0.12f; s->cmd[2] = (float)(0.3 * sin(t * 0.4));
        s->det_fps = 24.5f;
    }

    /* board */
    s->cpu_pct[1] = (float)(38 + 10 * sin(t * 0.9));
    s->cpu_pct[2] = (float)(22 + 8 * sin(t * 1.3));
    s->cpu_pct[0] = (s->cpu_pct[1] + s->cpu_pct[2]) / 2;
    s->soc_temp = 52.5f;
    s->mem_total_mb = 1006;
    s->mem_avail_mb = 612;
    s->cmm_used_mb = 312;
    s->cmm_total_mb = 1536;
    dash_strlcpy(s->governor, "performance", sizeof s->governor);
    s->uptime_s = 3600 + t;
    s->wifi_st = ST_OK;
    dash_strlcpy(s->ip, "192.168.1.42", sizeof s->ip);
    s->wifi_dbm = -48;
    if (has_fault(faults, "no_pad", NULL)) {
        s->pad_st = ST_OFF;
        s->pad_name[0] = 0;
    } else {
        s->pad_st = ST_OK;
        dash_strlcpy(s->pad_name, "Xbox Wireless Controller", sizeof s->pad_name);
    }
    s->audio_st = ST_OK;
    s->audio_cards = 1;
}
