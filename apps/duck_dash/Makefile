# Host and plain-Linux builds of duck_dash. The full app for the MaixCAM2
# (screen, touch, camera, NPU) is built with MaixCDK: see README.md.
#
#   make              host build (./build/duck_dash), headless or --fb
#   make test         C unit tests
#   make aarch64      cross-compiled for the MaixCAM2 without MaixCDK (fbdev only)
#   make snapshots    PPM screenshots of the simulated duck in build/

CC ?= cc
CFLAGS ?= -O2 -g
DASH_CFLAGS = $(CFLAGS) -std=gnu99 -Wall -Wextra -Wno-unused-parameter -Imain/include
LDLIBS += -lpthread -lm
CORE := $(wildcard main/src/core/*.c)
BUILD ?= build

all: $(BUILD)/duck_dash

$(BUILD)/duck_dash: $(CORE) host/plat_host.c main/include/*.h
	@mkdir -p $(BUILD)
	$(CC) $(DASH_CFLAGS) $(LDFLAGS) -o $@ $(CORE) host/plat_host.c $(LDLIBS)

$(BUILD)/test_core: $(CORE) tests/test_core.c main/include/*.h
	@mkdir -p $(BUILD)
	$(CC) $(DASH_CFLAGS) -o $@ $(filter-out main/src/core/app.c,$(CORE)) tests/test_core.c $(LDLIBS)

test: $(BUILD)/test_core
	$(BUILD)/test_core

aarch64:
	$(MAKE) CC=aarch64-linux-gnu-gcc BUILD=build-aarch64 CFLAGS="-O2" LDFLAGS=-static all

snapshots: $(BUILD)/duck_dash
	$(BUILD)/duck_dash --sim --frames 3 --snapshot $(BUILD)/sim_ok.ppm
	$(BUILD)/duck_dash --sim=missing:23,missing:24,lowvolt:6.7,no_feet,no_pad --frames 3 --snapshot $(BUILD)/sim_faults.ppm
	$(BUILD)/duck_dash --sim=walk --frames 3 --snapshot $(BUILD)/sim_walk.ppm
	$(BUILD)/duck_dash --sim=hot:24,overload:13 --page 1 --frames 3 --snapshot $(BUILD)/sim_servos.ppm

clean:
	rm -rf build build-aarch64

.PHONY: all test aarch64 snapshots clean
