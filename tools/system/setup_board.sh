#!/bin/sh
# Debloat a stock MaixCAM2 image for the duck, and undo it.
#
#   sh tools/system/setup_board.sh audit                 # show what runs and how memory is split; changes nothing
#   sh tools/system/setup_board.sh apply [options]       # apply the changes below
#   sh tools/system/setup_board.sh revert                # undo everything apply did
#
# apply options:
#   --autostart          boot into duck.service instead of the Sipeed launcher
#   --args "..."         arguments for scripts/walk.py in duck.service (default: none, i.e. duck_config.json)
#   --cmm MIB            set maix_memory_cmm in /boot/configs (NPU/camera pool; -1 = Sipeed default). Needs a reboot.
#   --keep-journal       keep the journal on disk (default: RAM only, 16 MB)
#   --dry-run            print what would change, change nothing
#
# Every change is recorded under /root/.odm-setup so revert can undo exactly that.
# Re-running apply is safe. See README "Debloat the MaixCAM2" for what each step does.

set -eu

REPO=$(cd "$(dirname "$0")/../.." && pwd)
# ODM_ROOT only exists so the script can be tested off-board; leave it unset.
R=${ODM_ROOT:-}
STATE=$R/root/.odm-setup
UNIT=$R/etc/systemd/system/duck.service
JOURNAL_DROPIN=$R/etc/systemd/journald.conf.d/90-odm-volatile.conf
BOOTCFG=$R/boot/configs
AUTOSTART_TXT=$R/maixapp/auto_start.txt

# Background jobs and daemons the duck never needs. Only units that exist and are
# enabled get disabled. Anything that loads NPU/camera/USB drivers is deliberately
# absent: never add a unit here without reading `systemctl cat <unit>` first.
UNITS_TO_DISABLE="
apt-daily.timer
apt-daily-upgrade.timer
unattended-upgrades.service
man-db.timer
motd-news.timer
e2scrub_all.timer
fstrim.timer
ModemManager.service
avahi-daemon.service
avahi-daemon.socket
cups.service
cups.socket
cups-browsed.service
"

DRY=0
run() {
    if [ "$DRY" = 1 ]; then echo "  [dry-run] $*"; else "$@"; fi
}
say() { echo "==> $*"; }

need_root() {
    [ -n "$R" ] || [ "$(id -u)" = 0 ] || { echo "run as root" >&2; exit 1; }
}

unit_exists() {
    systemctl list-unit-files "$1" --no-legend 2>/dev/null | grep -q "^$1"
}

cmm_info() {
    if [ -r /proc/ax_proc/mem_cmm_info ]; then
        grep -A1 "CMM_USE_INFO" /proc/ax_proc/mem_cmm_info | tail -1
    else
        echo "  (no /proc/ax_proc/mem_cmm_info)"
    fi
}

cmd_audit() {
    say "Enabled units"
    systemctl list-unit-files --state=enabled --no-legend --no-pager | awk '{print "  " $1}'
    say "Timers"
    systemctl list-timers --no-legend --no-pager | awk '{print "  " $NF}' | sort -u
    say "Slowest boot units"
    systemd-analyze blame --no-pager 2>/dev/null | head -10 || true
    say "Top CPU"
    ps -eo pid,pcpu,rss,comm --sort=-pcpu | head -8
    say "Top memory (RSS, KB)"
    ps -eo pid,pcpu,rss,comm --sort=-rss | head -8
    say "Linux RAM"
    free -m
    say "CMM (NPU/camera pool)"
    cmm_info
    [ -r "$BOOTCFG" ] && grep -H "maix_memory_cmm" "$BOOTCFG" || true
    say "CPU governor"
    cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null || echo "  (no cpufreq)"
    say "Candidates apply would disable"
    for u in $UNITS_TO_DISABLE; do
        unit_exists "$u" && echo "  $u: $(systemctl is-enabled "$u" 2>/dev/null || true)"
    done
    return 0
}

cmd_apply() {
    AUTOSTART=0; ARGS=""; CMM=""; KEEP_JOURNAL=0
    while [ $# -gt 0 ]; do
        case "$1" in
            --autostart) AUTOSTART=1 ;;
            --args) ARGS="$2"; shift ;;
            --cmm) CMM="$2"; shift ;;
            --keep-journal) KEEP_JOURNAL=1 ;;
            --dry-run) DRY=1 ;;
            *) echo "unknown option: $1" >&2; exit 2 ;;
        esac
        shift
    done
    if [ -n "$CMM" ]; then
        case "$CMM" in
            -1) ;;
            ''|*[!0-9]*) echo "--cmm must be a whole number of MiB, or -1" >&2; exit 2 ;;
            *) [ "$CMM" -ge 512 ] || { echo "refusing --cmm below 512 MiB: camera + NPU models would not fit" >&2; exit 2; } ;;
        esac
    fi
    need_root
    run mkdir -p "$STATE"

    say "Disabling unused services"
    for u in $UNITS_TO_DISABLE; do
        unit_exists "$u" || continue
        if [ "$(systemctl is-enabled "$u" 2>/dev/null || true)" = enabled ]; then
            echo "  $u"
            run systemctl disable --now "$u"
            [ "$DRY" = 1 ] || echo "$u" >> "$STATE/disabled_units"
        fi
    done

    if [ "$KEEP_JOURNAL" = 0 ]; then
        say "Journal in RAM only ($JOURNAL_DROPIN)"
        run mkdir -p "$(dirname "$JOURNAL_DROPIN")"
        if [ "$DRY" = 0 ]; then
            printf '[Journal]\nStorage=volatile\nRuntimeMaxUse=16M\n' > "$JOURNAL_DROPIN"
        fi
        run systemctl restart systemd-journald
    fi

    say "Installing $UNIT (walk.py $ARGS)"
    if [ "$DRY" = 0 ]; then
        esc() { printf '%s' "$1" | sed 's/[\\|&]/\\&/g'; }
        sed -e "s|@REPO@|$(esc "$REPO")|g" -e "s|@ARGS@|$(esc "$ARGS")|g" "$REPO/tools/system/duck.service" > "$UNIT"
    fi
    run systemctl daemon-reload

    if [ "$AUTOSTART" = 1 ]; then
        say "Booting into duck.service instead of the launcher"
        if unit_exists launcher.service && [ "$(systemctl is-enabled launcher.service 2>/dev/null || true)" = enabled ]; then
            run systemctl disable launcher.service
            [ "$DRY" = 1 ] || touch "$STATE/launcher_disabled"
        fi
        run systemctl enable duck.service
        if [ -f "$AUTOSTART_TXT" ]; then
            run mv "$AUTOSTART_TXT" "$STATE/auto_start.txt"
        fi
    fi

    if [ -n "$CMM" ]; then
        say "Setting maix_memory_cmm=$CMM in $BOOTCFG (takes effect after reboot)"
        [ -f "$BOOTCFG" ] || { echo "  $BOOTCFG not found, skipping"; CMM=""; }
    fi
    if [ -n "$CMM" ]; then
        [ -f "$STATE/configs.orig" ] || run cp "$BOOTCFG" "$STATE/configs.orig"
        if grep -q "^maix_memory_cmm=" "$BOOTCFG"; then
            run sed -i "s/^maix_memory_cmm=.*/maix_memory_cmm=$CMM/" "$BOOTCFG"
        else
            if [ "$DRY" = 0 ]; then echo "maix_memory_cmm=$CMM" >> "$BOOTCFG"; else echo "  [dry-run] append maix_memory_cmm=$CMM"; fi
        fi
        run sync
    fi

    say "Done."
    [ "$AUTOSTART" = 1 ] && echo "  The duck starts at next boot. Stop it with: systemctl stop duck   (logs: journalctl -u duck -f)"
    [ "$AUTOSTART" = 0 ] && echo "  duck.service installed but not enabled. Try it with: systemctl start duck"
    [ -n "$CMM" ] && echo "  Reboot for the CMM change, then check: free -m; sh $0 audit"
    echo "  Undo everything with: sh $0 revert"
}

cmd_revert() {
    for a in "$@"; do [ "$a" = --dry-run ] && DRY=1; done
    need_root
    [ -d "$STATE" ] || { echo "nothing to revert ($STATE missing)"; exit 0; }

    say "Removing duck.service"
    if [ -f "$UNIT" ]; then
        run systemctl disable --now duck.service || true
        run rm -f "$UNIT"
    fi
    if [ -f "$STATE/launcher_disabled" ]; then
        say "Re-enabling launcher.service"
        run systemctl enable launcher.service
        run rm -f "$STATE/launcher_disabled"
    fi
    if [ -f "$STATE/auto_start.txt" ]; then
        say "Restoring /maixapp/auto_start.txt"
        run mv "$STATE/auto_start.txt" "$AUTOSTART_TXT"
    fi
    if [ -f "$STATE/disabled_units" ]; then
        say "Re-enabling services"
        sort -u "$STATE/disabled_units" | while read -r u; do
            echo "  $u"
            run systemctl enable --now "$u" || true
        done
        run rm -f "$STATE/disabled_units"
    fi
    if [ -f "$JOURNAL_DROPIN" ]; then
        say "Journal back to disk"
        run rm -f "$JOURNAL_DROPIN"
        run systemctl restart systemd-journald
    fi
    if [ -f "$STATE/configs.orig" ]; then
        say "Restoring $BOOTCFG (takes effect after reboot)"
        run cp "$STATE/configs.orig" "$BOOTCFG"
        run rm -f "$STATE/configs.orig"
        run sync
    fi
    run systemctl daemon-reload
    [ "$DRY" = 1 ] || rmdir "$STATE" 2>/dev/null || true
    say "Reverted. Reboot to get the stock launcher (and CMM size) back."
}

case "${1:-}" in
    audit) shift; cmd_audit "$@" ;;
    apply) shift; cmd_apply "$@" ;;
    revert) shift; cmd_revert "$@" ;;
    *) sed -n '2,19p' "$0" | sed 's/^# \{0,1\}//'; exit 2 ;;
esac
