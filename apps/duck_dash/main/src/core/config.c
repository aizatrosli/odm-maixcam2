/* duck_config.json reader: the few keys the dashboard needs, with the same
 * defaults as odm_maix/config.py. A tiny JSON walker, no dependencies. */
#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "dash.h"

const char *const DASH_JOINT_NAMES[DASH_NJ] = {
    "left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle",
    "neck_pitch", "head_pitch", "head_yaw", "head_roll",
    "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle",
};
const char *const DASH_JOINT_SHORT[DASH_NJ] = {
    "L hip yaw", "L hip roll", "L hip pit", "L knee", "L ankle",
    "neck pit", "head pit", "head yaw", "head roll",
    "R hip yaw", "R hip roll", "R hip pit", "R knee", "R ankle",
};
/* odm_maix/hwi.py JOINT_IDS, policy order */
const int DASH_JOINT_IDS[DASH_NJ] = {20, 21, 22, 23, 24, 30, 31, 32, 33, 10, 11, 12, 13, 14};

void dash_strlcpy(char *dst, const char *src, int n)
{
    if (n <= 0) return;
    size_t len = src ? strlen(src) : 0;
    if (len > (size_t)n - 1) len = n - 1;
    if (len) memcpy(dst, src, len);
    dst[len] = 0;
}

/* ---------------------------------------------------------------- json walk */

static const char *ws(const char *p)
{
    while (*p && isspace((unsigned char)*p)) p++;
    return p;
}

static const char *skip_string(const char *p)
{
    /* p at opening quote */
    for (p++; *p && *p != '"'; p++)
        if (*p == '\\' && p[1]) p++;
    return *p ? p + 1 : p;
}

static const char *skip_value(const char *p)
{
    p = ws(p);
    if (*p == '"') return skip_string(p);
    if (*p == '{' || *p == '[') {
        int depth = 0;
        while (*p) {
            if (*p == '"') {
                p = skip_string(p);
                continue;
            }
            if (*p == '{' || *p == '[') depth++;
            else if (*p == '}' || *p == ']') {
                if (--depth == 0) return p + 1;
            }
            p++;
        }
        return p;
    }
    while (*p && *p != ',' && *p != '}' && *p != ']') p++;
    return p;
}

/* Finds `key` among the members of the object starting at `obj` ('{'). */
static const char *member(const char *obj, const char *key)
{
    if (!obj) return NULL;
    const char *p = ws(obj);
    if (*p != '{') return NULL;
    p++;
    size_t kl = strlen(key);
    for (;;) {
        p = ws(p);
        if (*p != '"') return NULL;
        const char *k = p + 1, *ke = skip_string(p) - 1;
        p = ws(ke + 1);
        if (*p != ':') return NULL;
        p = ws(p + 1);
        if ((size_t)(ke - k) == kl && strncmp(k, key, kl) == 0) return p;
        p = ws(skip_value(p));
        if (*p != ',') return NULL;
        p++;
    }
}

/* Dotted path lookup: "maixcam2.vision.model_path" */
static const char *jpath(const char *root, const char *path)
{
    char part[64];
    const char *p = root;
    while (p && *path) {
        const char *dot = strchr(path, '.');
        size_t n = dot ? (size_t)(dot - path) : strlen(path);
        if (n >= sizeof part) return NULL;
        memcpy(part, path, n);
        part[n] = 0;
        p = member(p, part);
        path += n + (dot ? 1 : 0);
    }
    return p;
}

static int jstr(const char *root, const char *path, char *out, int n)
{
    const char *v = jpath(root, path);
    if (!v || *v != '"') return 0;
    int i = 0;
    for (v++; *v && *v != '"' && i < n - 1; v++) {
        if (*v == '\\' && v[1]) v++;
        out[i++] = *v;
    }
    out[i] = 0;
    return 1;
}

static int jnum(const char *root, const char *path, double *out)
{
    const char *v = jpath(root, path);
    if (!v || !(*v == '-' || isdigit((unsigned char)*v))) return 0;
    *out = strtod(v, NULL);
    return 1;
}

static int jbool(const char *root, const char *path, int *out)
{
    const char *v = jpath(root, path);
    if (!v) return 0;
    if (!strncmp(v, "true", 4)) *out = 1;
    else if (!strncmp(v, "false", 5)) *out = 0;
    else return 0;
    return 1;
}

static int jnull(const char *root, const char *path)
{
    const char *v = jpath(root, path);
    return v && !strncmp(v, "null", 4);
}

static void expand_home(char *s, int n)
{
    if (s[0] != '~') return;
    const char *home = getenv("HOME");
    char tmp[256];
    snprintf(tmp, sizeof tmp, "%s%s", home ? home : "/root", s + 1);
    dash_strlcpy(s, tmp, n);
}

/* --------------------------------------------------------------- config */

void dash_config_defaults(dash_config *c)
{
    memset(c, 0, sizeof *c);
    const char *home = getenv("HOME");
    if (!home) home = "/root";
    snprintf(c->path, sizeof c->path, "%s/duck_config.json", home);
    dash_strlcpy(c->servo_port, "/dev/ttyACM0", sizeof c->servo_port);
    c->servo_baud = 1000000;
    c->imu_bus = -1;
    c->imu_addr = 0x28;
    snprintf(c->imu_offsets_path, sizeof c->imu_offsets_path, "%s/imu_offsets.json", home);
    dash_strlcpy(c->left_foot, "GPIOB21", sizeof c->left_foot);
    dash_strlcpy(c->right_foot, "GPIOB20", sizeof c->right_foot);
    c->pinmap_n = 2;
    dash_strlcpy(c->pinmap_pin[0], "A8", 8);
    dash_strlcpy(c->pinmap_func[0], "I2C7_SCL", 24);
    dash_strlcpy(c->pinmap_pin[1], "A9", 8);
    dash_strlcpy(c->pinmap_func[1], "I2C7_SDA", 24);
    dash_strlcpy(c->policy_backend, "onnx", sizeof c->policy_backend);
    snprintf(c->policy_path, sizeof c->policy_path, "%s/BEST_WALK_ONNX_2.onnx", home);
    dash_strlcpy(c->vision_model, "/root/models/yolo11n.mud", sizeof c->vision_model);
    for (int i = 0; i < DASH_NJ; i++) c->joint_lo[i] = c->joint_hi[i] = 0;
}

int dash_config_load(dash_config *c, const char *path)
{
    if (path) dash_strlcpy(c->path, path, sizeof c->path);
    FILE *f = fopen(c->path, "rb");
    if (!f) return 1;
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (n <= 0 || n > (1 << 20)) {
        fclose(f);
        return -1;
    }
    char *js = malloc(n + 1);
    if (!js) {
        fclose(f);
        return -1;
    }
    size_t got = fread(js, 1, n, f);
    fclose(f);
    js[got] = 0;
    if (*ws(js) != '{') {
        free(js);
        return -1;
    }
    c->loaded = 1;

    double d;
    int b;
    jstr(js, "maixcam2.servo_port", c->servo_port, sizeof c->servo_port);
    if (jnum(js, "maixcam2.servo_baudrate", &d)) c->servo_baud = (int)d;
    if (jnum(js, "maixcam2.imu_i2c_bus", &d)) c->imu_bus = (int)d;
    else if (jnull(js, "maixcam2.imu_i2c_bus")) c->imu_bus = -1;
    if (jnum(js, "maixcam2.imu_address", &d)) c->imu_addr = (int)d;
    if (jstr(js, "maixcam2.imu_offsets_path", c->imu_offsets_path, sizeof c->imu_offsets_path))
        expand_home(c->imu_offsets_path, sizeof c->imu_offsets_path);
    if (jbool(js, "imu_upside_down", &b)) c->imu_upside_down = b;
    jstr(js, "maixcam2.left_foot_gpio", c->left_foot, sizeof c->left_foot);
    jstr(js, "maixcam2.right_foot_gpio", c->right_foot, sizeof c->right_foot);
    jstr(js, "maixcam2.policy_backend", c->policy_backend, sizeof c->policy_backend);
    if (jstr(js, "maixcam2.policy_path", c->policy_path, sizeof c->policy_path))
        expand_home(c->policy_path, sizeof c->policy_path);
    if (jstr(js, "maixcam2.vision.model_path", c->vision_model, sizeof c->vision_model))
        expand_home(c->vision_model, sizeof c->vision_model);
    if (jbool(js, "expression_features.screen_eyes", &b)) c->screen_eyes = b;
    if (jbool(js, "maixcam2.vision.show", &b)) c->vision_show = b;

    /* pinmap replaces the default one, as in config.py */
    const char *pm = jpath(js, "maixcam2.pinmap");
    if (pm && *pm == '{') {
        c->pinmap_n = 0;
        const char *p = ws(pm + 1);
        while (*p == '"' && c->pinmap_n < 8) {
            const char *ke = skip_string(p);
            int kl = (int)(ke - p - 2);
            if (kl > 7) kl = 7;
            memcpy(c->pinmap_pin[c->pinmap_n], p + 1, kl);
            c->pinmap_pin[c->pinmap_n][kl] = 0;
            p = ws(ws(ke) + 1); /* past ':' */
            if (*p == '"') {
                const char *ve = skip_string(p);
                int vl = (int)(ve - p - 2);
                if (vl > 23) vl = 23;
                memcpy(c->pinmap_func[c->pinmap_n], p + 1, vl);
                c->pinmap_func[c->pinmap_n][vl] = 0;
                c->pinmap_n++;
            }
            p = ws(skip_value(p));
            if (*p == ',') p = ws(p + 1);
        }
    }

    for (int i = 0; i < DASH_NJ; i++) {
        char key[64];
        snprintf(key, sizeof key, "joints_offsets.%s", DASH_JOINT_NAMES[i]);
        if (jnum(js, key, &d)) c->joint_offset[i] = (float)d;
        snprintf(key, sizeof key, "maixcam2.joint_limits.%s", DASH_JOINT_NAMES[i]);
        const char *a = jpath(js, key);
        if (a && *a == '[') {
            char *e;
            double lo = strtod(ws(a + 1), &e);
            const char *q = ws(e);
            if (*q == ',') {
                double hi = strtod(ws(q + 1), NULL);
                c->joint_lo[i] = (float)lo;
                c->joint_hi[i] = (float)hi;
            }
        }
    }
    free(js);
    return 0;
}
