/* The sampler thread: reads every device at ~20 Hz into a snapshot that the
 * render loop copies. Nothing here turns servo torque on or writes servo
 * registers; the only writes are the BNO086 report requests (SH-2 Set
 * Feature, harmless if diagnose.py already asked for them) and pin functions
 * through the platform. */
#define _GNU_SOURCE
#include <errno.h>
#include <math.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#include "dash.h"
#include "hw.h"

#define HZ 20
#define RETRY_S 2.0
#define FEED_MAX_AGE_S 1.0

struct dash_sampler {
    dash_config cfg;
    dash_options opt;
    const dash_platform *plat;
    char feed_path[160];
    char faults[256];

    pthread_t th;
    pthread_mutex_t mu;
    int running, threaded;
    dash_snapshot shared;
    dash_snapshot w; /* owned by the sampler thread */
    double t0;

    ft_bus bus;
    int bus_open;
    double bus_next_try;
    double ping_next[DASH_NJ];
    int fails[DASH_NJ];

    bno_dev imu;
    int imu_open, imu_fails;
    double imu_next_try, imu_last_report;

    int foot_fd[2];
    double feet_next_try;

    dash_cpu_state cpu;
    double sys_next;
    int pinmap_done, npu_done;

    dash_status cam_st;
    char cam_msg[DASH_STR];
    float cam_fps;
};

static int exists(const char *p)
{
    struct stat st;
    return p && *p && stat(p, &st) == 0;
}

static int ends_with(const char *s, const char *suf)
{
    size_t a = strlen(s), b = strlen(suf);
    return a >= b && !strcmp(s + a - b, suf);
}

/* A .mud file names its .axmodel files; check the NPU one is next to it. */
static int mud_model_ok(const char *mud, char *missing, int n)
{
    FILE *f = fopen(mud, "r");
    if (!f) return 0;
    char line[256], dir[200];
    dash_strlcpy(dir, mud, sizeof dir);
    char *slash = strrchr(dir, '/');
    if (slash) slash[1] = 0;
    else dir[0] = 0;
    int ok = 1;
    while (fgets(line, sizeof line, f)) {
        char key[64], val[160];
        if (sscanf(line, " %63[^= ] = %159s", key, val) != 2) continue;
        if (strcmp(key, "model_npu") && strcmp(key, "model")) continue;
        char p[400];
        snprintf(p, sizeof p, "%s%s", val[0] == '/' ? "" : dir, val);
        if (!exists(p)) {
            ok = 0;
            dash_strlcpy(missing, val, n);
        }
    }
    fclose(f);
    return ok;
}

static const char *base(const char *p)
{
    const char *s = strrchr(p, '/');
    return s ? s + 1 : p;
}

/* ------------------------------------------------------------------ servos */

static void bus_close(dash_sampler *s)
{
    if (s->bus_open) ft_close(&s->bus);
    s->bus_open = 0;
}

static void sample_bus(dash_sampler *s, double now)
{
    dash_snapshot *w = &s->w;
    dash_strlcpy(w->servo_port, s->cfg.servo_port, sizeof w->servo_port);
    if (s->opt.no_bus) {
        w->bus_st = ST_OFF;
        snprintf(w->bus_msg, sizeof w->bus_msg, "disabled (--no-bus)");
        return;
    }
    if (!s->bus_open) {
        if (now < s->bus_next_try) return;
        s->bus_next_try = now + RETRY_S;
        for (int i = 0; i < DASH_NJ; i++) w->servo[i].present = 0;
        w->servos_found = 0;
        w->batt_v = 0;
        w->batt_pct = -1;
        w->batt_st = ST_UNKNOWN;
        if (!exists(s->cfg.servo_port)) {
            w->bus_st = ST_FAIL;
            snprintf(w->bus_msg, sizeof w->bus_msg, "%.24s not found (USB host mode?)", base(s->cfg.servo_port));
            return;
        }
        char who[48];
        if (dash_dev_in_use(s->cfg.servo_port, who, sizeof who)) {
            w->bus_st = ST_WARN;
            snprintf(w->bus_msg, sizeof w->bus_msg, "busy: %s", who);
            return;
        }
        int r = ft_open(&s->bus, s->cfg.servo_port, s->cfg.servo_baud);
        if (r < 0) {
            w->bus_st = ST_FAIL;
            snprintf(w->bus_msg, sizeof w->bus_msg, "open failed: %s", strerror(-r));
            return;
        }
        s->bus_open = 1;
        memset(s->ping_next, 0, sizeof s->ping_next);
    }

    double t0 = dash_now();
    int found = 0, hot = 0, errbits = 0, any_read = 0;
    float vmin = 99, vmax = 0, vsum = 0;
    char missing[DASH_STR] = "";
    for (int i = 0; i < DASH_NJ; i++) {
        dash_servo *sv = &s->w.servo[i];
        sv->lo = s->cfg.joint_lo[i];
        sv->hi = s->cfg.joint_hi[i];
        if (!sv->present && now < s->ping_next[i]) goto missing;
        uint8_t d[8];
        int err = ft_read(&s->bus, DASH_JOINT_IDS[i], FT_REG_PRESENT_POSITION, 8, d);
        if (err < 0) {
            /* a servo is dropped after two misses in a row, then re-tried every RETRY_S */
            if (++s->fails[i] >= 2 || !sv->present) {
                sv->present = 0;
                s->ping_next[i] = now + RETRY_S;
            }
            if (!sv->present) goto missing;
            found++;
            continue;
        }
        any_read = 1;
        s->fails[i] = 0;
        sv->present = 1;
        sv->pos = ft_raw_to_rad_pos(d[0] | d[1] << 8) - s->cfg.joint_offset[i];
        sv->vel = ft_raw_to_rad_vel(d[2] | d[3] << 8);
        sv->load = ft_raw_to_load_pct(d[4] | d[5] << 8);
        sv->volt = d[6] / 10.0f;
        sv->temp = d[7];
        sv->err = err;
        found++;
        if (sv->temp > hot) hot = sv->temp;
        if (sv->volt < vmin) vmin = sv->volt;
        if (sv->volt > vmax) vmax = sv->volt;
        vsum += sv->volt;
        errbits |= err;
        continue;
    missing: {
        char id[8];
        snprintf(id, sizeof id, "%s%d", missing[0] ? " " : "", DASH_JOINT_IDS[i]);
        if (strlen(missing) + strlen(id) < sizeof missing - 4) strcat(missing, id);
        else if (!strstr(missing, "...")) strcat(missing, " ...");
    }
    }
    w->bus_sweep_ms = (float)((dash_now() - t0) * 1000);
    w->bus_errors = s->bus.errors;
    w->servos_found = found;
    w->servo_temp_max = hot;
    w->servo_volt_min = any_read ? vmin : 0;
    w->servo_volt_max = vmax;

    if (!exists(s->cfg.servo_port)) { /* adapter unplugged */
        bus_close(s);
        w->bus_st = ST_FAIL;
        snprintf(w->bus_msg, sizeof w->bus_msg, "%.24s disappeared", base(s->cfg.servo_port));
        return;
    }
    if (found == 0) {
        w->bus_st = ST_FAIL;
        snprintf(w->bus_msg, sizeof w->bus_msg, "no servo answers (7.4V on driver?)");
    } else if (found < DASH_NJ) {
        w->bus_st = ST_FAIL;
        snprintf(w->bus_msg, sizeof w->bus_msg, "missing ID %s", missing);
    } else if (errbits || hot >= 55) {
        w->bus_st = ST_WARN;
        snprintf(w->bus_msg, sizeof w->bus_msg, "%s", hot >= 55 ? "a servo runs hot" : "servo status error");
    } else {
        w->bus_st = ST_OK;
        snprintf(w->bus_msg, sizeof w->bus_msg, "all %d servos answer", DASH_NJ);
    }
    if (any_read) {
        int n = 0;
        for (int i = 0; i < DASH_NJ; i++) n += w->servo[i].present && w->servo[i].volt > 0;
        float v = n ? vsum / n : 0;
        w->batt_v = v;
        w->batt_pct = (int)dash_batt_pct(v);
        w->batt_st = dash_batt_status(v);
    }
}

/* --------------------------------------------------------------------- IMU */

static void sample_imu(dash_sampler *s, double now)
{
    dash_snapshot *w = &s->w;
    if (s->opt.no_imu) {
        w->imu_st = ST_OFF;
        snprintf(w->imu_msg, sizeof w->imu_msg, "disabled (--no-imu)");
        return;
    }
    if (!s->imu_open) {
        if (now < s->imu_next_try) return;
        s->imu_next_try = now + RETRY_S;
        int r = s->cfg.imu_bus >= 0 ? bno_open(&s->imu, s->cfg.imu_bus, s->cfg.imu_addr) : bno_scan(&s->imu);
        if (r < 0) {
            w->imu_st = ST_FAIL;
            if (s->cfg.imu_bus >= 0)
                snprintf(w->imu_msg, sizeof w->imu_msg, "no BNO086 on i2c-%d @0x%02x", s->cfg.imu_bus, s->cfg.imu_addr);
            else
                snprintf(w->imu_msg, sizeof w->imu_msg, "no BNO086 on i2c-0..7");
            return;
        }
        bno_start(&s->imu, 20000); /* 50 Hz is plenty for the screen */
        s->imu_open = 1;
        s->imu_fails = 0;
        s->imu_last_report = now;
    }
    int n = bno_poll(&s->imu);
    if (n > 0) s->imu_last_report = now;
    if (n < 0 || now - s->imu_last_report > 2.0) {
        if (++s->imu_fails > 5 || n >= 0) {
            bno_close(&s->imu);
            s->imu_open = 0;
            w->imu_st = ST_FAIL;
            snprintf(w->imu_msg, sizeof w->imu_msg, n < 0 ? "BNO086 stopped answering" : "BNO086 stopped reporting");
        }
        return;
    }
    s->imu_fails = 0;
    if (!s->imu.have_acc || !s->imu.have_gyro) {
        w->imu_st = ST_WARN;
        snprintf(w->imu_msg, sizeof w->imu_msg, "i2c-%d @0x%02x, waiting for reports", s->imu.bus, s->imu.addr);
        return;
    }
    bno_remap(s->cfg.imu_remap, s->cfg.imu_upside_down, s->imu.acc, w->acc);
    bno_remap(s->cfg.imu_remap, s->cfg.imu_upside_down, s->imu.gyro, w->gyro);
    for (int i = 0; i < 3; i++) w->calib[i] = s->imu.accuracy[i];
    float g = sqrtf(w->acc[0] * w->acc[0] + w->acc[1] * w->acc[1] + w->acc[2] * w->acc[2]);
    w->roll = atan2f(w->acc[1], w->acc[2]) * 57.2958f;
    w->pitch = atan2f(-w->acc[0], sqrtf(w->acc[1] * w->acc[1] + w->acc[2] * w->acc[2])) * 57.2958f;
    if (s->imu.reset_cause == 3) {
        w->imu_st = ST_WARN;
        snprintf(w->imu_msg, sizeof w->imu_msg, "i2c-%d: last reset by watchdog", s->imu.bus);
    } else if (g < 8.8f || g > 10.8f) {
        w->imu_st = ST_WARN;
        snprintf(w->imu_msg, sizeof w->imu_msg, "i2c-%d: |g| = %.1f m/s2", s->imu.bus, g);
    } else {
        w->imu_st = ST_OK;
        snprintf(w->imu_msg, sizeof w->imu_msg, "i2c-%d @0x%02x fw %d.%d.%d%s", s->imu.bus, s->imu.addr, s->imu.sw_major,
                 s->imu.sw_minor, s->imu.sw_patch, exists(s->cfg.imu_offsets_path) ? "" : ", not calibrated");
    }
}

/* ------------------------------------------------------------------- feet */

static void sample_feet(dash_sampler *s, double now)
{
    dash_snapshot *w = &s->w;
    if (s->opt.no_feet) {
        w->feet_st = ST_OFF;
        w->foot[0] = w->foot[1] = -1;
        snprintf(w->feet_msg, sizeof w->feet_msg, "disabled (--no-feet)");
        return;
    }
    const char *names[2] = {s->cfg.left_foot, s->cfg.right_foot};
    if ((s->foot_fd[0] < 0 || s->foot_fd[1] < 0) && now >= s->feet_next_try) {
        s->feet_next_try = now + RETRY_S;
        for (int i = 0; i < 2; i++) {
            if (s->foot_fd[i] >= 0) continue;
            int chip, line;
            if (gpio_parse(names[i], &chip, &line) < 0) {
                w->feet_st = ST_FAIL;
                snprintf(w->feet_msg, sizeof w->feet_msg, "bad pin name %s", names[i]);
                continue;
            }
            int fd = gpio_request_input(chip, line, i ? "duck_dash_rfoot" : "duck_dash_lfoot");
            if (fd < 0) {
                w->feet_st = fd == -EBUSY ? ST_WARN : ST_FAIL;
                if (fd == -EBUSY) snprintf(w->feet_msg, sizeof w->feet_msg, "%s in use", names[i]);
                else if (fd == -ENOENT) snprintf(w->feet_msg, sizeof w->feet_msg, "no /dev/gpiochip%d", chip);
                else snprintf(w->feet_msg, sizeof w->feet_msg, "%s: %s", names[i], strerror(-fd));
            }
            s->foot_fd[i] = fd < 0 ? -1 : fd;
        }
    }
    int ok = 1;
    for (int i = 0; i < 2; i++) {
        int v = s->foot_fd[i] >= 0 ? gpio_get(s->foot_fd[i]) : -1;
        w->foot[i] = v < 0 ? -1 : !v; /* switch closed pulls the pin low = contact */
        ok &= v >= 0;
    }
    if (ok) {
        w->feet_st = ST_OK;
        snprintf(w->feet_msg, sizeof w->feet_msg, "%s / %s", names[0] + (strncmp(names[0], "GPIO", 4) ? 0 : 4),
                 names[1] + (strncmp(names[1], "GPIO", 4) ? 0 : 4));
    }
}

/* ------------------------------------------------------------- models */

static void check_models(dash_sampler *s)
{
    dash_snapshot *w = &s->w;
    const dash_config *c = &s->cfg;
    char miss[96] = "";
    dash_strlcpy(w->backend, c->policy_backend, sizeof w->backend);
    if (!exists(c->policy_path)) {
        w->npu_st = ST_FAIL;
        snprintf(w->npu_msg, sizeof w->npu_msg, "%s not found", base(c->policy_path));
    } else if (ends_with(c->policy_path, ".mud")) {
        if (!mud_model_ok(c->policy_path, miss, sizeof miss)) {
            w->npu_st = ST_FAIL;
            snprintf(w->npu_msg, sizeof w->npu_msg, "%s missing", miss);
        } else if (s->plat && s->plat->npu_probe) {
            char msg[DASH_STR] = "";
            float lm = 0, im = 0;
            int r = s->plat->npu_probe(s->plat->ctx, c->policy_path, 101, &lm, &im, msg, sizeof msg);
            w->npu_load_ms = lm;
            w->npu_infer_ms = im;
            w->npu_st = r == 0 ? ST_OK : ST_FAIL;
            if (r == 0) snprintf(w->npu_msg, sizeof w->npu_msg, "%s %.2f ms on NPU", base(c->policy_path), im);
            else snprintf(w->npu_msg, sizeof w->npu_msg, "%s", msg[0] ? msg : "NPU load failed");
        } else {
            w->npu_st = ST_OK;
            snprintf(w->npu_msg, sizeof w->npu_msg, "%s found (not loaded)", base(c->policy_path));
        }
    } else {
        w->npu_st = strcmp(c->policy_backend, "npu") ? ST_OK : ST_WARN;
        snprintf(w->npu_msg, sizeof w->npu_msg, "%s on CPU", base(c->policy_path));
    }

    if (!exists(c->vision_model)) {
        w->vis_st = ST_WARN; /* only needed for --follow / --interact */
        snprintf(w->vis_msg, sizeof w->vis_msg, "%s not found", base(c->vision_model));
    } else if (ends_with(c->vision_model, ".mud") && !mud_model_ok(c->vision_model, miss, sizeof miss)) {
        w->vis_st = ST_FAIL;
        snprintf(w->vis_msg, sizeof w->vis_msg, "%s missing", miss);
    } else {
        w->vis_st = ST_OK;
        snprintf(w->vis_msg, sizeof w->vis_msg, "%s", base(c->vision_model));
    }
}

/* ---------------------------------------------------------------- the loop */

static void release_hw(dash_sampler *s)
{
    bus_close(s);
    if (s->imu_open) bno_close(&s->imu);
    s->imu_open = 0;
    for (int i = 0; i < 2; i++) {
        if (s->foot_fd[i] >= 0) close(s->foot_fd[i]);
        s->foot_fd[i] = -1;
    }
}

void dash_sampler_step(dash_sampler *s)
{
    double now = dash_now();
    dash_snapshot *w = &s->w;
    w->t = now;
    w->seq++;

    if (s->opt.sim_faults) {
        dash_sim_fill(w, s->faults, now - s->t0);
    } else {
        int was_live = w->walk_live;
        if (dash_feed_read(s->feed_path, FEED_MAX_AGE_S, w, &s->cfg)) {
            release_hw(s); /* the walk loop owns the hardware now */
        } else {
            if (was_live) { /* walk.py stopped: probe again right away */
                w->walk_live = 0;
                w->screen_busy = 0;
                s->bus_next_try = s->imu_next_try = s->feet_next_try = 0;
                s->npu_done = 0; /* the feed overwrote the policy line */
            }
            w->source = SRC_PROBE;
            if (!s->pinmap_done && s->plat && s->plat->pinmap) {
                s->pinmap_done = 1;
                for (int i = 0; i < s->cfg.pinmap_n; i++)
                    s->plat->pinmap(s->plat->ctx, s->cfg.pinmap_pin[i], s->cfg.pinmap_func[i]);
                const char *feet[2] = {s->cfg.left_foot, s->cfg.right_foot};
                for (int i = 0; i < 2; i++)
                    if (!strncmp(feet[i], "GPIO", 4)) s->plat->pinmap(s->plat->ctx, feet[i] + 4, feet[i]);
            }
            sample_bus(s, now);
            sample_imu(s, now);
            sample_feet(s, now);
        }
        if (!s->npu_done) {
            s->npu_done = 1;
            check_models(s);
        }
        if (now >= s->sys_next) {
            s->sys_next = now + 1.0;
            dash_sysinfo(w, &s->cpu);
        }
        w->config_loaded = s->cfg.loaded;
        dash_strlcpy(w->config_path, s->cfg.path, sizeof w->config_path);
    }

    pthread_mutex_lock(&s->mu);
    if (!s->opt.sim_faults) { /* the render thread owns the camera */
        w->cam_st = s->cam_st;
        dash_strlcpy(w->cam_msg, s->cam_msg, sizeof w->cam_msg);
        w->cam_fps = s->cam_fps;
    }
    s->shared = *w;
    pthread_mutex_unlock(&s->mu);
}

static void *thread_main(void *arg)
{
    dash_sampler *s = arg;
    const long period_ns = 1000000000L / HZ;
    struct timespec next;
    clock_gettime(CLOCK_MONOTONIC, &next);
    while (__atomic_load_n(&s->running, __ATOMIC_ACQUIRE)) {
        dash_sampler_step(s);
        next.tv_nsec += period_ns;
        while (next.tv_nsec >= 1000000000L) {
            next.tv_nsec -= 1000000000L;
            next.tv_sec++;
        }
        struct timespec now;
        clock_gettime(CLOCK_MONOTONIC, &now);
        if (now.tv_sec > next.tv_sec || (now.tv_sec == next.tv_sec && now.tv_nsec > next.tv_nsec))
            next = now; /* overran: don't try to catch up */
        else
            clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &next, NULL);
    }
    return NULL;
}

dash_sampler *dash_sampler_new(const dash_config *cfg, const dash_options *opt, const dash_platform *plat)
{
    dash_sampler *s = calloc(1, sizeof *s);
    if (!s) return NULL;
    s->cfg = *cfg;
    s->opt = *opt;
    s->plat = plat;
    dash_strlcpy(s->feed_path, opt->feed_path ? opt->feed_path : "/dev/shm/duck_telemetry", sizeof s->feed_path);
    if (opt->sim_faults) {
        dash_strlcpy(s->faults, opt->sim_faults, sizeof s->faults);
        s->opt.sim_faults = s->faults;
    }
    s->foot_fd[0] = s->foot_fd[1] = -1;
    s->bus.fd = -1;
    s->imu.fd = -1;
    s->t0 = dash_now();
    s->w.batt_pct = -1;
    s->w.foot[0] = s->w.foot[1] = -1;
    for (int i = 0; i < 3; i++) s->w.calib[i] = -1;
    s->cam_st = opt->camera_mode ? ST_UNKNOWN : ST_OFF;
    snprintf(s->cam_msg, sizeof s->cam_msg, "%s", opt->camera_mode ? "starting" : "preview off");
    pthread_mutex_init(&s->mu, NULL);
    return s;
}

dash_sampler *dash_sampler_start(const dash_config *cfg, const dash_options *opt, const dash_platform *plat)
{
    dash_sampler *s = dash_sampler_new(cfg, opt, plat);
    if (!s) return NULL;
    s->running = 1;
    dash_sampler_step(s); /* first snapshot before the screen comes up */
    if (pthread_create(&s->th, NULL, thread_main, s) == 0) s->threaded = 1;
    return s;
}

void dash_sampler_get(dash_sampler *s, dash_snapshot *out)
{
    pthread_mutex_lock(&s->mu);
    *out = s->shared;
    pthread_mutex_unlock(&s->mu);
}

void dash_sampler_set_camera(dash_sampler *s, dash_status st, const char *msg, float fps)
{
    pthread_mutex_lock(&s->mu);
    s->cam_st = st;
    dash_strlcpy(s->cam_msg, msg, sizeof s->cam_msg);
    s->cam_fps = fps;
    if (!s->opt.sim_faults) {
        s->shared.cam_st = st;
        dash_strlcpy(s->shared.cam_msg, msg, sizeof s->shared.cam_msg);
        s->shared.cam_fps = fps;
    }
    pthread_mutex_unlock(&s->mu);
}

void dash_sampler_stop(dash_sampler *s)
{
    if (!s) return;
    __atomic_store_n(&s->running, 0, __ATOMIC_RELEASE);
    if (s->threaded) pthread_join(s->th, NULL);
    release_hw(s);
    pthread_mutex_destroy(&s->mu);
    free(s);
}
