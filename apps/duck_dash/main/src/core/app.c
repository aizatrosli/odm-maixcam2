/* Main loop shared by the MaixCDK app and the host build. */
#define _GNU_SOURCE
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include "dash.h"
#include "gfx.h"

#define CANVAS_W 640
#define CANVAS_H 480
#define CAM_W 320
#define CAM_H 240
#define PID_FILE "/dev/shm/duck_dash.pid"

static volatile sig_atomic_t g_stop;
static void on_signal(int sig)
{
    (void)sig;
    g_stop = 1;
}

static void usage(void)
{
    puts("duck_dash: live telemetry for Open Duck Mini on the MaixCAM2 screen\n"
         "\n"
         "  --config PATH       duck_config.json (default ~/duck_config.json)\n"
         "  --servo-port PATH   override maixcam2.servo_port\n"
         "  --imu-bus N         override maixcam2.imu_i2c_bus\n"
         "  --feed PATH         walk.py telemetry feed (default /dev/shm/duck_telemetry)\n"
         "  --camera live|off   camera preview (default live; released while walk.py runs)\n"
         "  --no-bus --no-imu --no-feet   leave that hardware alone\n"
         "  --fps N             screen refresh rate, the most camera frames shown (default 30)\n"
         "  --ui-hz N           how often the readings are redrawn (default 10); in between\n"
         "                      only new camera frames are drawn\n"
         "  --cam-fps N         camera sensor frame rate (default 30, 0 = SDK default)\n"
         "  --cam-max-exp MS    longest exposure the dashboard's auto-exposure picks\n"
         "                      (default 33, which keeps 30 fps; 0 = the SDK's own AE,\n"
         "                      which drops to ~10 fps in dim light)\n"
         "  --page N            start page (0 overview, 1 servo table)\n"
         "  --sim[=FAULTS]      simulated duck, e.g. --sim=missing:23,lowvolt:6.6,no_imu,walk\n"
         "  --frames N          stop after N frames\n"
         "  --snapshot OUT.ppm  save the last frame (with --frames)\n"
         "  --dump              print the last snapshot as text (with --frames)\n"
         "  --stats[=S]         every S seconds (default 10) print frame times, sampler rate,\n"
         "                      source switches and memory to stderr; a summary at exit\n"
         "\n"
         "Tap the screen to switch pages, tap the title (top left) to exit. The user key\n"
         "or Ctrl-C exits too.");
}

static const char *stw(dash_status s)
{
    static const char *n[] = {"unknown", "ok", "warn", "fail", "off"};
    return n[s];
}

static void dump(const dash_snapshot *s)
{
    printf("source=%s\n", s->source == SRC_SIM ? "sim" : s->source == SRC_FEED ? "feed" : "probe");
    printf("bus=%s servos_found=%d bus_msg=%s\n", stw(s->bus_st), s->servos_found, s->bus_msg);
    printf("servo_ids_present=");
    for (int i = 0; i < DASH_NJ; i++)
        if (s->servo[i].present) printf("%d,", DASH_JOINT_IDS[i]);
    printf("\n");
    for (int i = 0; i < DASH_NJ; i++) {
        const dash_servo *v = &s->servo[i];
        if (v->present)
            printf("servo %d pos=%.3f vel=%.3f load=%.1f volt=%.1f temp=%d err=%d\n", DASH_JOINT_IDS[i], v->pos, v->vel,
                   v->load, v->volt, v->temp, v->err);
    }
    printf("battery=%s volt=%.2f pct=%d\n", stw(s->batt_st), s->batt_v, s->batt_pct);
    printf("imu=%s msg=%s acc=%.2f,%.2f,%.2f roll=%.1f pitch=%.1f\n", stw(s->imu_st), s->imu_msg, s->acc[0], s->acc[1],
           s->acc[2], s->roll, s->pitch);
    printf("feet=%s left=%d right=%d msg=%s\n", stw(s->feet_st), s->foot[0], s->foot[1], s->feet_msg);
    printf("camera=%s msg=%s fps=%.1f\n", stw(s->cam_st), s->cam_msg, s->cam_fps);
    printf("policy=%s backend=%s msg=%s\n", stw(s->npu_st), s->backend, s->npu_msg);
    printf("detector=%s msg=%s\n", stw(s->vis_st), s->vis_msg);
    printf("walk_live=%d mode=%s state=%s loop_ms=%.2f policy_ms=%.2f steps=%d bus_errors=%u temp_max=%d\n",
           s->walk_live, s->walk_mode, s->walk_state, s->loop_mean_ms, s->policy_ms, s->policy_steps, s->bus_errors,
           s->servo_temp_max);
    printf("config_loaded=%d path=%s\n", s->config_loaded, s->config_path);
}

/* --stats: how the dashboard holds up (frame time, sampler rate, memory) */
typedef struct {
    double t0, win_t0;
    long frames, win_frames;
    double draw_sum, draw_max, win_draw_sum, win_draw_max;
    unsigned seq0, win_seq0;
    int switches, win_switches, last_source, last_live;
    long rss0_kb, rss_max_kb;
} dash_stats;

static long rss_kb(void)
{
    long pages = 0, rss = 0;
    FILE *f = fopen("/proc/self/statm", "r");
    if (!f) return 0;
    if (fscanf(f, "%ld %ld", &pages, &rss) != 2) rss = 0;
    fclose(f);
    return rss * (sysconf(_SC_PAGESIZE) / 1024);
}

static void stats_frame(dash_stats *st, const dash_snapshot *s, double draw_ms)
{
    if (!st->frames) {
        st->t0 = st->win_t0 = dash_now();
        st->seq0 = st->win_seq0 = s->seq;
        st->last_source = s->source;
        st->last_live = s->walk_live;
        st->rss0_kb = st->rss_max_kb = rss_kb();
    }
    st->frames++;
    st->win_frames++;
    st->draw_sum += draw_ms;
    st->win_draw_sum += draw_ms;
    if (draw_ms > st->draw_max) st->draw_max = draw_ms;
    if (draw_ms > st->win_draw_max) st->win_draw_max = draw_ms;
    if ((int)s->source != st->last_source || s->walk_live != st->last_live) {
        st->switches++;
        st->win_switches++;
        st->last_source = s->source;
        st->last_live = s->walk_live;
    }
}

static void stats_report(dash_stats *st, const dash_snapshot *s, FILE *out, int final)
{
    double now = dash_now();
    long rss = rss_kb();
    if (rss > st->rss_max_kb) st->rss_max_kb = rss;
    if (final) {
        double dt = now - st->t0;
        fprintf(out,
                "stats_total seconds=%.1f frames=%ld fps=%.1f draw_ms_mean=%.2f draw_ms_max=%.2f sampler_hz=%.1f "
                "source_switches=%d rss_kb=%ld rss_start_kb=%ld rss_max_kb=%ld\n",
                dt, st->frames, dt > 0 ? st->frames / dt : 0, st->frames ? st->draw_sum / st->frames : 0, st->draw_max,
                dt > 0 ? (s->seq - st->seq0) / dt : 0, st->switches, rss, st->rss0_kb, st->rss_max_kb);
        return;
    }
    double dt = now - st->win_t0;
    fprintf(out,
            "stats t=%.0fs fps=%.1f draw_ms=%.2f/%.2f sampler_hz=%.1f switches=%d source=%s%s rss_kb=%ld\n",
            now - st->t0, dt > 0 ? st->win_frames / dt : 0, st->win_frames ? st->win_draw_sum / st->win_frames : 0,
            st->win_draw_max, dt > 0 ? (s->seq - st->win_seq0) / dt : 0, st->win_switches,
            s->source == SRC_SIM ? "sim" : s->source == SRC_FEED ? "feed" : "probe", s->walk_live ? "+walk" : "", rss);
    fflush(out);
    st->win_t0 = now;
    st->win_frames = 0;
    st->win_draw_sum = st->win_draw_max = 0;
    st->win_seq0 = s->seq;
    st->win_switches = 0;
}

static void sleep_until(struct timespec *next, long period_ns)
{
    next->tv_nsec += period_ns;
    while (next->tv_nsec >= 1000000000L) {
        next->tv_nsec -= 1000000000L;
        next->tv_sec++;
    }
    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);
    if (now.tv_sec > next->tv_sec || (now.tv_sec == next->tv_sec && now.tv_nsec > next->tv_nsec)) *next = now;
    else clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, next, NULL);
}

int dash_main(int argc, char **argv, const dash_platform *plat)
{
    dash_config cfg;
    dash_options opt = {0};
    const char *config_path = NULL, *snapshot = NULL, *servo_port = NULL;
    int fps = 30, ui_hz = 10, frames = -1, page = 0, do_dump = 0, imu_bus = -2;
    int cam_fps = 30, cam_max_exp_us = 33000;
    double stats_every = 0;
    opt.camera_mode = 1;
    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
#define NEXT (i + 1 < argc ? argv[++i] : "")
        if (!strcmp(a, "--config")) config_path = NEXT;
        else if (!strcmp(a, "--servo-port")) servo_port = NEXT;
        else if (!strcmp(a, "--imu-bus")) imu_bus = atoi(NEXT);
        else if (!strcmp(a, "--feed")) opt.feed_path = NEXT;
        else if (!strcmp(a, "--camera")) opt.camera_mode = strcmp(NEXT, "off") != 0;
        else if (!strcmp(a, "--no-bus")) opt.no_bus = 1;
        else if (!strcmp(a, "--no-imu")) opt.no_imu = 1;
        else if (!strcmp(a, "--no-feet")) opt.no_feet = 1;
        else if (!strcmp(a, "--fps")) fps = atoi(NEXT);
        else if (!strcmp(a, "--ui-hz")) ui_hz = atoi(NEXT);
        else if (!strcmp(a, "--cam-fps")) cam_fps = atoi(NEXT);
        else if (!strcmp(a, "--cam-max-exp")) cam_max_exp_us = (int)(atof(NEXT) * 1000);
        else if (!strcmp(a, "--page")) page = atoi(NEXT);
        else if (!strcmp(a, "--frames")) frames = atoi(NEXT);
        else if (!strcmp(a, "--snapshot")) snapshot = NEXT;
        else if (!strcmp(a, "--dump")) do_dump = 1;
        else if (!strcmp(a, "--stats")) stats_every = 10;
        else if (!strncmp(a, "--stats=", 8)) stats_every = atof(a + 8) > 0 ? atof(a + 8) : 10;
        else if (!strcmp(a, "--sim")) opt.sim_faults = "";
        else if (!strncmp(a, "--sim=", 6)) opt.sim_faults = a + 6;
        else if (!strcmp(a, "-h") || !strcmp(a, "--help")) {
            usage();
            return 0;
        } else {
            fprintf(stderr, "unknown option %s (see --help)\n", a);
            return 2;
        }
#undef NEXT
    }
    if (fps < 1) fps = 1;
    if (fps > 60) fps = 60;
    if (ui_hz < 1) ui_hz = 1;
    if (ui_hz > fps) ui_hz = fps;
    if (cam_fps < 0) cam_fps = 0;
    if (cam_max_exp_us < 0) cam_max_exp_us = 0;
    page = page < 0 ? 0 : page % dash_ui_pages();

    dash_config_defaults(&cfg);
    int cr = dash_config_load(&cfg, config_path);
    if (cr < 0) fprintf(stderr, "duck_dash: %s is not valid JSON, using defaults\n", cfg.path);
    if (servo_port) dash_strlcpy(cfg.servo_port, servo_port, sizeof cfg.servo_port);
    if (imu_bus >= -1) cfg.imu_bus = imu_bus;

    signal(SIGINT, on_signal);
    signal(SIGTERM, on_signal);

    int dw = CANVAS_W, dh = CANVAS_H, have_display = 0;
    if (plat->display_open && plat->display_open(plat->ctx, &dw, &dh) == 0) have_display = 1;
    gfx_canvas canvas = {calloc((size_t)CANVAS_W * CANVAS_H, 3), CANVAS_W, CANVAS_H};
    uint8_t *cam = calloc((size_t)CAM_W * CAM_H, 3);
    if (!canvas.px || !cam) return 1;

    FILE *pf = have_display && !opt.sim_faults ? fopen(PID_FILE, "w") : NULL;
    if (pf) {
        fprintf(pf, "%d\n", getpid());
        fclose(pf);
    }

    dash_sampler *smp = dash_sampler_start(&cfg, &opt, plat);
    if (!smp) return 1;

    int cam_open = 0, cam_frames = 0, have_cam = 0;
    double cam_t0 = dash_now();
    int use_cam = opt.camera_mode && !opt.sim_faults && plat->camera_open && plat->camera_read;
    if (!opt.sim_faults && opt.camera_mode && !plat->camera_open)
        dash_sampler_set_camera(smp, ST_UNKNOWN, "no camera access in this build", 0);
    if (use_cam && plat->camera_config) plat->camera_config(plat->ctx, cam_fps, cam_max_exp_us);

    dash_snapshot snap, drawn_snap;
    memset(&drawn_snap, 0, sizeof drawn_snap);
    double last_full = -1e9;
    int last_page = -1;
    dash_stats stats = {0};
    const long period = 1000000000L / fps;
    struct timespec next;
    clock_gettime(CLOCK_MONOTONIC, &next);
    for (int f = 0; !g_stop && (frames < 0 || f < frames); f++) {
        if (plat->need_exit && plat->need_exit(plat->ctx)) break;
        dash_sampler_get(smp, &snap);

        /* the camera: live preview, handed over while the walk loop runs */
        int new_cam = 0;
        if (use_cam) {
            if (snap.walk_live && cam_open) {
                plat->camera_close(plat->ctx);
                cam_open = have_cam = 0;
                dash_sampler_set_camera(smp, ST_OFF, "released to the walk loop", 0);
            } else if (!snap.walk_live && !cam_open && f % (fps * 2) == 0) {
                if (plat->camera_open(plat->ctx, CAM_W, CAM_H) == 0) {
                    cam_open = 1;
                    cam_frames = 0;
                    cam_t0 = dash_now();
                    dash_sampler_set_camera(smp, ST_UNKNOWN, "waiting for frames", 0);
                } else {
                    dash_sampler_set_camera(smp, ST_FAIL, "camera open failed", 0);
                }
            }
            if (cam_open) {
                if (plat->camera_read(plat->ctx, cam, CAM_W, CAM_H) == 0) {
                    have_cam = new_cam = 1;
                    cam_frames++;
                }
                double dt = dash_now() - cam_t0;
                if (dt > 1.0) {
                    char m[DASH_STR];
                    snprintf(m, sizeof m, "%dx%d preview", CAM_W, CAM_H);
                    dash_sampler_set_camera(smp, cam_frames ? ST_OK : ST_FAIL, cam_frames ? m : "no frames", cam_frames / dt);
                    cam_frames = 0;
                    cam_t0 = dash_now();
                }
            }
        }

        int tx, ty;
        if (plat->touch_tap && plat->touch_tap(plat->ctx, &tx, &ty)) {
            /* touch coordinates are in panel pixels; the canvas is scaled to fit */
            tx = dw > 0 ? tx * CANVAS_W / dw : tx;
            ty = dh > 0 ? ty * CANVAS_H / dh : ty;
            if (tx < 150 && ty < 32) break; /* tap the "DUCK TELEMETRY" title to exit */
            page = (page + 1) % dash_ui_pages();
        }

        /* The readings change at the sampler's 20 Hz and nobody reads text faster
         * than that, so the whole screen is redrawn at most ui_hz times a second
         * (and at once on a page change); in between, only a new camera frame is
         * drawn into its box. A frame with neither is not sent to the display. */
        double t_draw = dash_now();
        int redraw = page != last_page || t_draw - last_full >= 1.0 / ui_hz || (frames >= 0 && f == frames - 1);
        int dirty = 0;
        if (redraw) {
            dash_ui_draw(&canvas, &snap, page, have_cam ? cam : NULL, CAM_W, CAM_H);
            last_full = t_draw;
            last_page = page;
            drawn_snap = snap;
            dirty = 1;
        } else if (new_cam) {
            dirty = dash_ui_draw_camera(&canvas, &drawn_snap, page, cam, CAM_W, CAM_H);
        }
        /* while the vision process draws its eyes or boxes, leave the screen to it */
        if (dirty && have_display && !snap.screen_busy) plat->display_show(plat->ctx, canvas.px, CANVAS_W, CANVAS_H);
        if (stats_every > 0) {
            stats_frame(&stats, &snap, (dash_now() - t_draw) * 1000);
            if (dash_now() - stats.win_t0 >= stats_every) stats_report(&stats, &snap, stderr, 0);
        }
        if (frames >= 0 && f == frames - 1) break;
        sleep_until(&next, period);
    }

    if (do_dump) {
        dash_sampler_get(smp, &snap);
        dump(&snap);
    }
    if (stats_every > 0 && stats.frames) {
        dash_sampler_get(smp, &snap);
        stats_report(&stats, &snap, stdout, 1);
    }
    if (snapshot) {
        dash_sampler_get(smp, &snap);
        dash_ui_draw(&canvas, &snap, page, have_cam ? cam : NULL, CAM_W, CAM_H);
        if (gfx_save_ppm(&canvas, snapshot) != 0) fprintf(stderr, "could not write %s\n", snapshot);
    }
    if (cam_open && plat->camera_close) plat->camera_close(plat->ctx);
    dash_sampler_stop(smp);
    if (pf) unlink(PID_FILE);
    free(canvas.px);
    free(cam);
    return 0;
}
