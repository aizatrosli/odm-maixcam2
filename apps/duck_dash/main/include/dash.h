/* duck_dash: live telemetry for Open Duck Mini on the MaixCAM2 screen.
 *
 * The core (everything under src/core) is plain C99 + POSIX. It talks to the
 * hardware through Linux device files: the Feetech bus through the tty,
 * the BNO086 (BNO08x) through /dev/i2c-N, the foot switches through /dev/gpiochipN.
 * The few things that need Sipeed's SDK (screen, touch, camera, NPU, pin
 * functions) go through dash_platform, which the MaixCDK shim fills in.
 */
#ifndef DUCK_DASH_H
#define DUCK_DASH_H

#include <stdint.h>

#include "gfx.h"

#ifdef __cplusplus
extern "C" {
#endif

#define DASH_NJ 14
#define DASH_STR 64

typedef enum { ST_UNKNOWN = 0, ST_OK, ST_WARN, ST_FAIL, ST_OFF } dash_status;
typedef enum { SRC_PROBE = 0, SRC_FEED, SRC_SIM } dash_source;

extern const char *const DASH_JOINT_NAMES[DASH_NJ];
extern const char *const DASH_JOINT_SHORT[DASH_NJ];
extern const int DASH_JOINT_IDS[DASH_NJ];

typedef struct {
    int present;  /* answered on the last sweep */
    float pos;    /* rad, policy frame (joint offset removed) */
    float vel;    /* rad/s */
    float load;   /* percent of max torque, signed */
    float volt;   /* V, 0 = not read yet */
    int temp;     /* C, 0 = not read yet */
    int err;      /* status error bits (overload, overheat, ...) */
    float lo, hi; /* soft limits from calibrate.py ranges, lo>=hi = none */
} dash_servo;

typedef struct {
    dash_source source;
    double t; /* monotonic seconds of this snapshot */
    unsigned seq;

    /* config */
    int config_loaded;
    char config_path[160];

    /* servo bus */
    dash_status bus_st;
    char bus_msg[DASH_STR];
    char servo_port[DASH_STR];
    int servos_found;
    dash_servo servo[DASH_NJ];
    float bus_sweep_ms; /* time to read every servo once */
    int servo_temp_max; /* C, 0 = unknown */
    float servo_volt_min, servo_volt_max;
    unsigned bus_errors;

    /* battery (from the servos' supply voltage: the 2S pack) */
    dash_status batt_st;
    float batt_v;
    int batt_pct;

    /* IMU */
    dash_status imu_st;
    char imu_msg[DASH_STR];
    float gyro[3], acc[3]; /* rad/s, m/s^2, body frame after the axis remap */
    float roll, pitch;     /* deg, from gravity */
    int calib[3]; /* report accuracy: accel gyro mag, 0..3, -1 unknown */

    /* foot switches: -1 unknown, 0 in the air, 1 contact */
    dash_status feet_st;
    char feet_msg[DASH_STR];
    int foot[2];

    /* camera */
    dash_status cam_st;
    char cam_msg[DASH_STR];
    float cam_fps;

    /* walking policy on the NPU / CPU */
    dash_status npu_st;
    char npu_msg[DASH_STR];
    char backend[16];
    float npu_load_ms, npu_infer_ms;

    /* person detector model */
    dash_status vis_st;
    char vis_msg[DASH_STR];

    /* walk loop (only when walk.py publishes its telemetry feed) */
    int walk_live;
    char walk_mode[24];
    char walk_state[48];
    int paused;
    float loop_mean_ms, loop_p99_ms, policy_ms;
    int overruns;
    float cmd[7];
    float det_fps;
    int screen_busy; /* the vision process draws on the screen */

    /* board */
    float cpu_pct[3]; /* total, core0, core1 */
    float soc_temp;
    int mem_avail_mb, mem_total_mb, cmm_used_mb, cmm_total_mb;
    char governor[24];
    double uptime_s;

    /* links */
    dash_status wifi_st;
    char ip[24];
    int wifi_dbm;
    dash_status pad_st;
    char pad_name[48];
    dash_status audio_st;
    int audio_cards;
} dash_snapshot;

/* What the board SDK provides. Any callback may be NULL. */
typedef struct dash_platform {
    void *ctx;
    const char *name;
    int (*display_open)(void *ctx, int *w, int *h);
    void (*display_show)(void *ctx, const uint8_t *rgb, int w, int h);
    /* 1 if a touch-down happened since the last call */
    int (*touch_tap)(void *ctx, int *x, int *y);
    int (*need_exit)(void *ctx);
    int (*camera_open)(void *ctx, int w, int h);
    int (*camera_read)(void *ctx, uint8_t *rgb, int w, int h); /* 0 = frame */
    void (*camera_close)(void *ctx);
    /* load a .mud, run one zero input of in_size floats; 0 = ok */
    int (*npu_probe)(void *ctx, const char *mud, int in_size, float *load_ms, float *infer_ms, char *msg, int msglen);
    int (*pinmap)(void *ctx, const char *pin, const char *func);
} dash_platform;

/* config.c ---------------------------------------------------------------- */
typedef struct {
    char path[160];
    int loaded;
    char servo_port[DASH_STR];
    int servo_baud;
    int imu_bus; /* -1 = scan */
    int imu_addr;
    char imu_offsets_path[160];
    int imu_upside_down;
    int imu_remap[6]; /* maixcam2.imu_axis_remap, imu_remap[0] = -1 when unset */
    char left_foot[24], right_foot[24];
    char pinmap_pin[8][8], pinmap_func[8][24];
    int pinmap_n;
    char policy_backend[16];
    char policy_path[160];
    char vision_model[160];
    int screen_eyes, vision_show;
    float joint_offset[DASH_NJ];
    float joint_lo[DASH_NJ], joint_hi[DASH_NJ];
} dash_config;

void dash_config_defaults(dash_config *c);
/* 0 = loaded, 1 = file missing (defaults used), -1 = parse error */
int dash_config_load(dash_config *c, const char *path);

/* sampler.c --------------------------------------------------------------- */
typedef struct {
    const char *feed_path;
    const char *sim_faults; /* non-NULL = simulated duck, no hardware */
    int camera_mode;        /* 0 off, 1 live preview */
    int no_bus, no_imu, no_feet;
} dash_options;

typedef struct dash_sampler dash_sampler;
dash_sampler *dash_sampler_start(const dash_config *cfg, const dash_options *opt, const dash_platform *plat);
void dash_sampler_get(dash_sampler *s, dash_snapshot *out);
/* camera results, reported by the render thread */
void dash_sampler_set_camera(dash_sampler *s, dash_status st, const char *msg, float fps);
void dash_sampler_stop(dash_sampler *s);
/* run one sampling pass synchronously (tests) */
void dash_sampler_step(dash_sampler *s);
dash_sampler *dash_sampler_new(const dash_config *cfg, const dash_options *opt, const dash_platform *plat);

/* sim.c ------------------------------------------------------------------- */
void dash_sim_fill(dash_snapshot *s, const char *faults, double t);

/* feed.c ------------------------------------------------------------------ */
/* Reads walk.py's telemetry feed. 1 = fresh feed applied, 0 = none or stale. */
int dash_feed_read(const char *path, double max_age_s, dash_snapshot *s, const dash_config *cfg);

/* sysinfo.c --------------------------------------------------------------- */
typedef struct {
    unsigned long long prev[3][2]; /* busy, total for total/core0/core1 */
} dash_cpu_state;
void dash_sysinfo(dash_snapshot *s, dash_cpu_state *cpu);
/* pid of another process that has `dev` open, 0 if none */
int dash_dev_in_use(const char *dev, char *who, int wholen);

/* ui.c -------------------------------------------------------------------- */
void dash_ui_draw(gfx_canvas *canvas, const dash_snapshot *s, int page, const uint8_t *cam, int cam_w, int cam_h);
int dash_ui_pages(void);

/* util -------------------------------------------------------------------- */
double dash_now(void);
float dash_batt_pct(float v);
dash_status dash_batt_status(float v);
void dash_strlcpy(char *dst, const char *src, int n);

/* app.c: shared main loop used by every platform */
int dash_main(int argc, char **argv, const dash_platform *plat);

#ifdef __cplusplus
}
#endif
#endif
