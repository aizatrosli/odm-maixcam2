/* Unit tests for the C core: packets, unit conversions, config, feed, sim. */
#include <fcntl.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include "dash.h"
#include "hw.h"

static int failures, checks;
#define CHECK(c)                                                         \
    do {                                                                 \
        checks++;                                                        \
        if (!(c)) {                                                      \
            failures++;                                                  \
            fprintf(stderr, "%s:%d: CHECK(%s) failed\n", __FILE__, __LINE__, #c); \
        }                                                                \
    } while (0)
#define NEAR(a, b, eps) CHECK(fabs((double)(a) - (double)(b)) < (eps))

static void test_packets(void)
{
    uint8_t p[16];
    /* odm_maix/feetech.py build_packet(1, INST_PING) == ff ff 01 02 01 fb */
    int n = ft_build_packet(p, 1, FT_INST_PING, NULL, 0);
    CHECK(n == 6);
    const uint8_t ping[] = {0xFF, 0xFF, 0x01, 0x02, 0x01, 0xFB};
    CHECK(memcmp(p, ping, 6) == 0);
    uint8_t par[2] = {56, 8};
    n = ft_build_packet(p, 23, FT_INST_READ, par, 2);
    CHECK(n == 8 && p[3] == 4 && p[4] == 2);
    CHECK(p[7] == (uint8_t)~(23 + 4 + 2 + 56 + 8));

    /* a status reply with 2 params, preceded by noise */
    uint8_t st[] = {0x00, 0xFF, 0xFF, 0x0D, 0x04, 0x20, 0x00, 0x08, 0};
    st[8] = (uint8_t)~(0x0D + 0x04 + 0x20 + 0x00 + 0x08);
    int id, err, np;
    const uint8_t *params;
    int used = ft_parse_status(st, sizeof st, &id, &err, &params, &np);
    CHECK(used == (int)sizeof st);
    CHECK(id == 13 && err == 0x20 && np == 2 && params[1] == 0x08);
    CHECK(ft_parse_status(st, 6, &id, &err, &params, &np) == 0); /* truncated */
    st[8] ^= 1;
    CHECK(ft_parse_status(st, sizeof st, &id, &err, &params, &np) == -1);
}

static void test_units(void)
{
    NEAR(ft_raw_to_rad_pos(2048), 0, 1e-6);
    NEAR(ft_raw_to_rad_pos(0), -M_PI, 1e-6);
    NEAR(ft_raw_to_rad_vel(0x8000 | 4095), -2 * M_PI, 1e-4);
    NEAR(ft_raw_to_load_pct(0x400 | 250), -25.0, 1e-4);
    NEAR(ft_raw_to_load_pct(1000), 100.0, 1e-4);
    uint8_t cfg, sign;
    bno_remap_bytes(0, &cfg, &sign);
    CHECK(cfg == 0x21 && sign == 0x04); /* bno055.axis_remap_bytes(*REMAP_UPRIGHT) */
    bno_remap_bytes(1, &cfg, &sign);
    CHECK(cfg == 0x21 && sign == 0x07);
    int chip, line;
    CHECK(gpio_parse("GPIOB21", &chip, &line) == 0 && chip == 1 && line == 21);
    CHECK(gpio_parse("A4", &chip, &line) == 0 && chip == 0 && line == 4);
    CHECK(gpio_parse("GPIO", &chip, &line) < 0);
    NEAR(dash_batt_pct(8.4f), 100, 1e-3);
    NEAR(dash_batt_pct(5.0f), 0, 1e-3);
    CHECK(dash_batt_pct(7.4f) > 40 && dash_batt_pct(7.4f) < 50);
    CHECK(dash_batt_status(6.2f) == ST_FAIL && dash_batt_status(6.8f) == ST_WARN && dash_batt_status(7.8f) == ST_OK);
    CHECK(dash_batt_status(0) == ST_UNKNOWN);
}

static void test_config(void)
{
    char path[] = "/tmp/duck_dash_cfgXXXXXX";
    int fd = mkstemp(path);
    const char *js = "{\n"
                     "  \"imu_upside_down\": true,\n"
                     "  \"expression_features\": {\"eyes\": false, \"screen_eyes\": true},\n"
                     "  \"joints_offsets\": {\"left_knee\": 0.12, \"head_yaw\": -0.05},\n"
                     "  \"maixcam2\": {\n"
                     "    \"servo_port\": \"/dev/ttyS4\", \"servo_baudrate\": 1000000,\n"
                     "    \"imu_i2c_bus\": 7, \"imu_address\": 41,\n"
                     "    \"note\": \"a \\\"quoted\\\" {brace}\",\n"
                     "    \"pinmap\": {\"A8\": \"I2C7_SCL\", \"A9\": \"I2C7_SDA\", \"A21\": \"UART4_TX\", \"A22\": \"UART4_RX\"},\n"
                     "    \"policy_backend\": \"npu\", \"policy_path\": \"~/out/walk_policy.mud\",\n"
                     "    \"vision\": {\"model_path\": \"/root/models/yolo11n.mud\", \"show\": false},\n"
                     "    \"joint_limits\": {\"left_knee\": [-0.2, 1.9]}\n"
                     "  }\n"
                     "}\n";
    CHECK(write(fd, js, strlen(js)) == (ssize_t)strlen(js));
    close(fd);
    setenv("HOME", "/home/duck", 1);
    dash_config c;
    dash_config_defaults(&c);
    CHECK(c.imu_bus == -1 && !strcmp(c.servo_port, "/dev/ttyACM0") && c.pinmap_n == 2);
    CHECK(dash_config_load(&c, path) == 0 && c.loaded);
    CHECK(!strcmp(c.servo_port, "/dev/ttyS4"));
    CHECK(c.imu_bus == 7 && c.imu_addr == 41 && c.imu_upside_down == 1);
    CHECK(c.pinmap_n == 4 && !strcmp(c.pinmap_pin[3], "A22") && !strcmp(c.pinmap_func[2], "UART4_TX"));
    CHECK(!strcmp(c.policy_backend, "npu"));
    CHECK(!strcmp(c.policy_path, "/home/duck/out/walk_policy.mud"));
    CHECK(c.screen_eyes == 1 && c.vision_show == 0);
    NEAR(c.joint_offset[3], 0.12, 1e-6);
    NEAR(c.joint_offset[7], -0.05, 1e-6);
    NEAR(c.joint_lo[3], -0.2, 1e-6);
    NEAR(c.joint_hi[3], 1.9, 1e-6);
    CHECK(c.joint_lo[0] == c.joint_hi[0]);
    unlink(path);

    dash_config d;
    dash_config_defaults(&d);
    CHECK(dash_config_load(&d, "/nonexistent/duck_config.json") == 1 && !d.loaded);
}

static void test_feed(void)
{
    char path[] = "/tmp/duck_dash_feedXXXXXX";
    int fd = mkstemp(path);
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    char buf[2048];
    int n = snprintf(buf, sizeof buf,
                     "duck_telemetry 1\ntime %.3f\nmode follow\nstate tracking\npaused 0\nbackend npu\n"
                     "loop_ms 6.5 9.25\npolicy_ms 0.31\noverruns 2\ncmd 0.1 0 0.2 0 0 0 0\n"
                     "pos 0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1 1.1 1.2 1.3\n"
                     "vel 0 0 0 0 0 0 0 0 0 0 0 0 0 0\ngyro 0 0 0\nacc 0 0 9.81\nfeet 1 0\n"
                     "volt 7.6\ntemp_max 44\nbus_errors 0\ndet_fps 21.5\nscreen 1\n",
                     ts.tv_sec + ts.tv_nsec / 1e9);
    CHECK(write(fd, buf, n) == n);
    close(fd);
    dash_config c;
    dash_config_defaults(&c);
    dash_snapshot s;
    memset(&s, 0, sizeof s);
    CHECK(dash_feed_read(path, 1.0, &s, &c) == 1);
    CHECK(s.walk_live && s.source == SRC_FEED && !strcmp(s.walk_mode, "follow") && !strcmp(s.walk_state, "tracking"));
    CHECK(s.servos_found == DASH_NJ && s.bus_st == ST_OK);
    NEAR(s.servo[13].pos, 1.3, 1e-5);
    NEAR(s.loop_p99_ms, 9.25, 1e-5);
    CHECK(s.overruns == 2 && s.foot[0] == 1 && s.foot[1] == 0 && s.screen_busy == 1);
    NEAR(s.batt_v, 7.6, 1e-5);
    CHECK(s.servo_temp_max == 44 && s.imu_st == ST_OK);
    NEAR(s.roll, 0, 1e-3);

    /* stale */
    fd = open(path, O_WRONLY | O_TRUNC);
    n = snprintf(buf, sizeof buf, "duck_telemetry 1\ntime %.3f\nmode walk\n", ts.tv_sec - 10.0);
    CHECK(write(fd, buf, n) == n);
    close(fd);
    memset(&s, 0, sizeof s);
    CHECK(dash_feed_read(path, 1.0, &s, &c) == 0);
    /* ...but a "starting" announcement lasts while the servos come up */
    fd = open(path, O_WRONLY | O_TRUNC);
    n = snprintf(buf, sizeof buf, "duck_telemetry 1\ntime %.3f\nmode walk\nstate starting\n", ts.tv_sec - 10.0);
    CHECK(write(fd, buf, n) == n);
    close(fd);
    CHECK(dash_feed_read(path, 1.0, &s, &c) == 1);
    unlink(path);
    CHECK(dash_feed_read(path, 1.0, &s, &c) == 0);
}

static void test_sim_and_ui(void)
{
    dash_snapshot s;
    memset(&s, 0, sizeof s);
    dash_sim_fill(&s, "missing:23,missing:24,lowvolt:6.6,no_imu", 1.0);
    CHECK(s.servos_found == 12 && s.bus_st == ST_FAIL && !s.servo[3].present && !s.servo[4].present);
    CHECK(s.batt_st == ST_WARN && s.imu_st == ST_FAIL);
    memset(&s, 0, sizeof s);
    dash_sim_fill(&s, "", 2.0);
    CHECK(s.servos_found == DASH_NJ && s.bus_st == ST_OK && s.batt_st == ST_OK && !s.walk_live);

    gfx_canvas c = {calloc(640 * 480, 3), 640, 480};
    for (int page = 0; page < dash_ui_pages(); page++) {
        dash_ui_draw(&c, &s, page, NULL, 0, 0);
        long lit = 0;
        for (int i = 0; i < 640 * 480 * 3; i++) lit += c.px[i] > 200;
        CHECK(lit > 1000); /* something was drawn */
    }
    free(c.px);
}

int main(void)
{
    test_packets();
    test_units();
    test_config();
    test_feed();
    test_sim_and_ui();
    printf("%d checks, %d failed\n", checks, failures);
    return failures != 0;
}
