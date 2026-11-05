/* Board health and links from /proc and /sys: CPU, SoC temperature, RAM,
 * AX CMM memory, governor, WiFi, gamepad, sound cards. */
#define _GNU_SOURCE
#include <arpa/inet.h>
#include <dirent.h>
#include <ifaddrs.h>
#include <net/if.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#include "dash.h"

double dash_now(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec / 1e9;
}

/* 2S Li-ion, resting voltage to charge (rough; under load it reads lower) */
float dash_batt_pct(float v)
{
    static const float tab[][2] = {{6.0f, 0}, {6.6f, 8}, {7.0f, 20}, {7.4f, 45}, {7.7f, 65}, {8.0f, 82}, {8.4f, 100}};
    const int n = sizeof tab / sizeof tab[0];
    if (v <= tab[0][0]) return 0;
    for (int i = 1; i < n; i++)
        if (v <= tab[i][0])
            return tab[i - 1][1] + (v - tab[i - 1][0]) / (tab[i][0] - tab[i - 1][0]) * (tab[i][1] - tab[i - 1][1]);
    return 100;
}

/* same thresholds as odm_maix/diag/checks.py: VOLT_FAIL 6.4, VOLT_WARN 7.0, VOLT_HIGH 8.5 */
dash_status dash_batt_status(float v)
{
    if (v <= 0) return ST_UNKNOWN;
    if (v < 6.4f || v > 8.5f) return ST_FAIL;
    if (v < 7.0f) return ST_WARN;
    return ST_OK;
}

static int read_file(const char *path, char *buf, int n)
{
    FILE *f = fopen(path, "r");
    if (!f) return -1;
    int got = (int)fread(buf, 1, n - 1, f);
    fclose(f);
    buf[got < 0 ? 0 : got] = 0;
    return got;
}

static void cpu_usage(dash_snapshot *s, dash_cpu_state *st)
{
    FILE *f = fopen("/proc/stat", "r");
    if (!f) return;
    char line[256];
    while (fgets(line, sizeof line, f)) {
        int idx;
        if (!strncmp(line, "cpu ", 4)) idx = 0;
        else if (!strncmp(line, "cpu0 ", 5)) idx = 1;
        else if (!strncmp(line, "cpu1 ", 5)) idx = 2;
        else continue;
        unsigned long long v[8] = {0};
        char *p = strchr(line, ' ');
        for (int i = 0; i < 8; i++) v[i] = strtoull(p, &p, 10);
        unsigned long long idle = v[3] + v[4], total = 0;
        for (int i = 0; i < 8; i++) total += v[i];
        unsigned long long busy = total - idle;
        unsigned long long db = busy - st->prev[idx][0], dt = total - st->prev[idx][1];
        if (st->prev[idx][1] && dt) s->cpu_pct[idx] = 100.0f * db / dt;
        st->prev[idx][0] = busy;
        st->prev[idx][1] = total;
    }
    fclose(f);
}

static void soc_temp(dash_snapshot *s)
{
    float best = 0;
    for (int i = 0; i < 8; i++) {
        char path[64], buf[32];
        snprintf(path, sizeof path, "/sys/class/thermal/thermal_zone%d/temp", i);
        if (read_file(path, buf, sizeof buf) <= 0) continue;
        float t = atoi(buf) / 1000.0f;
        if (t > best) best = t;
    }
    s->soc_temp = best;
}

static void memory(dash_snapshot *s)
{
    char buf[4096];
    if (read_file("/proc/meminfo", buf, sizeof buf) > 0) {
        char *p;
        if ((p = strstr(buf, "MemTotal:"))) s->mem_total_mb = atoi(p + 9) / 1024;
        if ((p = strstr(buf, "MemAvailable:"))) s->mem_avail_mb = atoi(p + 13) / 1024;
    }
    /* AX630C media memory (camera, NPU), as in diag check_board_system */
    if (read_file("/proc/ax_proc/mem_cmm_info", buf, sizeof buf) > 0) {
        char *t = strstr(buf, "total="), *u = strstr(buf, "used=");
        if (t && u) {
            s->cmm_total_mb = atoi(t + 6) / 1024;
            s->cmm_used_mb = atoi(u + 5) / 1024;
        }
    }
    if (read_file("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor", buf, sizeof buf) > 0) {
        buf[strcspn(buf, "\n")] = 0;
        dash_strlcpy(s->governor, buf, sizeof s->governor);
    }
    if (read_file("/proc/uptime", buf, sizeof buf) > 0) s->uptime_s = atof(buf);
}

static void wifi(dash_snapshot *s)
{
    struct ifaddrs *ifa, *i;
    s->ip[0] = 0;
    char any[24] = "";
    if (getifaddrs(&ifa) == 0) {
        for (i = ifa; i; i = i->ifa_next) {
            if (!i->ifa_addr || i->ifa_addr->sa_family != AF_INET || (i->ifa_flags & IFF_LOOPBACK)) continue;
            char ip[24];
            inet_ntop(AF_INET, &((struct sockaddr_in *)i->ifa_addr)->sin_addr, ip, sizeof ip);
            if (!strncmp(i->ifa_name, "wlan", 4)) dash_strlcpy(s->ip, ip, sizeof s->ip);
            else if (!any[0]) dash_strlcpy(any, ip, sizeof any);
        }
        freeifaddrs(ifa);
    }
    s->wifi_dbm = 0;
    char buf[1024];
    if (read_file("/proc/net/wireless", buf, sizeof buf) > 0) {
        char *p = strstr(buf, "wlan");
        if (p && (p = strchr(p, ':'))) {
            /* status link level noise */
            float st, link, level;
            if (sscanf(p + 1, "%f %f. %f", &st, &link, &level) == 3) s->wifi_dbm = (int)level;
        }
    }
    if (s->ip[0]) s->wifi_st = ST_OK;
    else if (any[0]) {
        dash_strlcpy(s->ip, any, sizeof s->ip);
        s->wifi_st = ST_OK;
    } else
        s->wifi_st = ST_WARN;
}

/* A joystick handler (jsN) in /proc/bus/input/devices = a gamepad pygame can open. */
static void gamepad(dash_snapshot *s)
{
    char buf[16384];
    s->pad_st = ST_OFF;
    s->pad_name[0] = 0;
    if (read_file("/proc/bus/input/devices", buf, sizeof buf) <= 0) return;
    char name[64] = "";
    for (char *line = strtok(buf, "\n"); line; line = strtok(NULL, "\n")) {
        if (!strncmp(line, "N: Name=", 8)) {
            char *q = line + 8;
            if (*q == '"') q++;
            dash_strlcpy(name, q, sizeof name);
            name[strcspn(name, "\"")] = 0;
        } else if (!strncmp(line, "H: Handlers=", 12) && strstr(line, "js")) {
            dash_strlcpy(s->pad_name, name, sizeof s->pad_name);
            s->pad_st = ST_OK;
            return;
        }
    }
}

static void audio(dash_snapshot *s)
{
    char buf[2048];
    s->audio_cards = 0;
    if (read_file("/proc/asound/cards", buf, sizeof buf) > 0) {
        for (char *p = buf; (p = strstr(p, "]: ")); p += 3) s->audio_cards++;
    }
    s->audio_st = s->audio_cards ? ST_OK : ST_WARN;
}

void dash_sysinfo(dash_snapshot *s, dash_cpu_state *cpu)
{
    cpu_usage(s, cpu);
    soc_temp(s);
    memory(s);
    wifi(s);
    gamepad(s);
    audio(s);
}

/* Scans /proc/<pid>/fd for another process holding `dev` open. */
int dash_dev_in_use(const char *dev, char *who, int wholen)
{
    struct stat want;
    if (stat(dev, &want) < 0) return 0;
    DIR *d = opendir("/proc");
    if (!d) return 0;
    int me = getpid(), found = 0;
    struct dirent *e;
    while (!found && (e = readdir(d))) {
        int pid = atoi(e->d_name);
        if (pid <= 0 || pid == me) continue;
        char fdpath[64];
        snprintf(fdpath, sizeof fdpath, "/proc/%d/fd", pid);
        DIR *fds = opendir(fdpath);
        if (!fds) continue;
        struct dirent *f;
        while ((f = readdir(fds))) {
            if (f->d_name[0] == '.') continue;
            char p[64 + 256];
            struct stat st;
            snprintf(p, sizeof p, "%s/%s", fdpath, f->d_name);
            if (stat(p, &st) == 0 && st.st_rdev == want.st_rdev && S_ISCHR(st.st_mode) && st.st_ino == want.st_ino) {
                found = pid;
                break;
            }
        }
        closedir(fds);
    }
    closedir(d);
    if (found && who) {
        char p[64], comm[64] = "?";
        snprintf(p, sizeof p, "/proc/%d/comm", found);
        if (read_file(p, comm, sizeof comm) > 0) comm[strcspn(comm, "\n")] = 0;
        snprintf(who, wholen, "%s %d", comm, found);
    }
    return found;
}
