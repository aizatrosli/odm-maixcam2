/* MaixCDK platform for duck_dash: the MaixCAM2 screen, touch panel, camera,
 * NPU and pin functions. Everything else is the C core in src/core.
 *
 * This file is C++ only because Sipeed's SDK (MaixCDK) is a C++ API; it
 * hands frames and touches to the C core and does no drawing itself. */
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "maix_basic.hpp"
#include "maix_camera.hpp"
#include "maix_display.hpp"
#include "maix_image.hpp"
#include "maix_nn.hpp"
#include "maix_pinmap.hpp"
#include "maix_touchscreen.hpp"

#include "dash.h"

using namespace maix;

namespace {

/* gain() takes 1024 for 1x; the AX630C sensor tops out near 15.5x */
constexpr int GAIN_1X = 1024;
constexpr int GAIN_MAX = 15872;
constexpr int EXP_MIN_US = 100;
constexpr int AE_TARGET_Y = 110; /* mean luma, 0..255 */

struct Ctx {
    std::unique_ptr<display::Display> disp;
    std::unique_ptr<touchscreen::TouchScreen> touch;
    bool was_pressed = false;
    bool disp_native = false; /* panel is the canvas size: show without scaling */

    /* camera: a thread reads every frame (blocking) and keeps the latest one,
     * so the screen loop never waits for the sensor and never misses a frame */
    std::unique_ptr<camera::Camera> cam;
    std::thread cam_thread;
    std::atomic<bool> cam_run{false};
    std::mutex cam_mu;
    std::vector<uint8_t> cam_buf;
    unsigned cam_seq = 0, cam_seq_read = 0;
    int cam_fps = 30, cam_max_exp_us = 33000;
    int ae_exp = 0, ae_gain = 0;

    ~Ctx()
    { /* a joinable std::thread at destruction would abort the app */
        cam_run = false;
        if (cam_thread.joinable()) cam_thread.join();
    }
};

int display_open(void *p, int *w, int *h)
{
    auto *c = static_cast<Ctx *>(p);
    try {
        c->disp.reset(new display::Display());
        if (!c->disp->is_opened()) return -1;
        *w = c->disp->width();
        *h = c->disp->height();
        c->disp_native = *w == 640 && *h == 480;
        c->touch.reset(new touchscreen::TouchScreen());
    } catch (std::exception &e) {
        log::error("duck_dash: display: %s", e.what());
        return -1;
    }
    return 0;
}

void display_show(void *p, const uint8_t *rgb, int w, int h)
{
    auto *c = static_cast<Ctx *>(p);
    if (!c->disp) return;
    /* wraps our buffer without a copy; show() scales 640x480 to the panel,
     * unless the panel already is 640x480 (MaixCAM2), which saves ~4 ms */
    try {
        image::Image img(w, h, image::FMT_RGB888, const_cast<uint8_t *>(rgb), w * h * 3, false);
        c->disp->show(img, c->disp_native && w == c->disp->width() && h == c->disp->height() ? image::FIT_NONE
                                                                                             : image::FIT_CONTAIN);
    } catch (std::exception &e) {
        log::error("duck_dash: show: %s", e.what());
    }
}

int touch_tap(void *p, int *x, int *y)
{
    auto *c = static_cast<Ctx *>(p);
    if (!c->touch || !c->touch->is_opened()) return 0;
    int tapped = 0;
    while (c->touch->available(0)) {
        bool pressed = false;
        if (c->touch->read(*x, *y, pressed) != err::ERR_NONE) break;
        if (pressed && !c->was_pressed) tapped = 1; /* count the touch-down edge only */
        c->was_pressed = pressed;
    }
    return tapped;
}

int need_exit(void *) { return app::need_exit(); }

void camera_config(void *p, int fps, int max_exposure_us)
{
    auto *c = static_cast<Ctx *>(p);
    c->cam_fps = fps;
    c->cam_max_exp_us = max_exposure_us;
}

/* The SDK's auto-exposure stretches the exposure to ~100 ms in dim light,
 * which caps the sensor at ~10 fps whatever frame rate was asked for. This
 * one keeps the exposure at or under cam_max_exp_us and makes up the rest
 * with gain: a darker, noisier picture in a dim room, but a steady frame rate.
 * It runs on every 3rd frame from a coarse luma sample (~1200 pixels). */
static void auto_exposure(Ctx *c, const uint8_t *rgb, int w, int h)
{
    long sum = 0, n = 0;
    for (int y = 4; y < h; y += 8)
        for (int x = 4; x < w; x += 8) {
            const uint8_t *px = rgb + ((size_t)y * w + x) * 3;
            sum += (77 * px[0] + 150 * px[1] + 29 * px[2]) >> 8;
            n++;
        }
    if (!n) return;
    int luma = (int)(sum / n);
    if (abs(luma - AE_TARGET_Y) < 12) return;
    double ratio = (double)AE_TARGET_Y / (luma < 1 ? 1 : luma);
    if (ratio > 1.4) ratio = 1.4; /* damped: the sensor applies changes a few frames late */
    if (ratio < 0.7) ratio = 0.7;
    double total = (double)c->ae_exp * c->ae_gain / GAIN_1X * ratio; /* exposure x gain, in us */
    double top = (double)c->cam_max_exp_us * GAIN_MAX / GAIN_1X;
    if (total > top) total = top;
    if (total < EXP_MIN_US) total = EXP_MIN_US;
    int exp = total > c->cam_max_exp_us ? c->cam_max_exp_us : (int)total;
    if (exp < EXP_MIN_US) exp = EXP_MIN_US;
    int gain = (int)(total * GAIN_1X / exp);
    if (gain < GAIN_1X) gain = GAIN_1X;
    if (gain > GAIN_MAX) gain = GAIN_MAX;
    if (abs(exp - c->ae_exp) * 33 > c->ae_exp) { /* only changes over 3% */
        c->cam->exposure(exp);
        c->ae_exp = exp;
    }
    if (abs(gain - c->ae_gain) * 33 > c->ae_gain) {
        c->cam->gain(gain);
        c->ae_gain = gain;
    }
}

static void camera_loop(Ctx *c, int w, int h)
{
    unsigned n = 0;
    bool ae = c->cam_max_exp_us > 0;
    while (c->cam_run) {
        std::unique_ptr<image::Image> img;
        try {
            img.reset(c->cam->read(true, 500)); /* blocking, so no frame is missed */
        } catch (std::exception &e) {
            img.reset();
        }
        if (!img || img->format() != image::FMT_RGB888) continue;
        int sw = img->width(), sh = img->height();
        const uint8_t *src = static_cast<const uint8_t *>(img->data());
        if (ae && ++n % 3 == 0) {
            try {
                auto_exposure(c, src, sw, sh);
            } catch (std::exception &e) {
                log::warn("duck_dash: auto-exposure off: %s", e.what());
                ae = false;
            }
        }
        std::lock_guard<std::mutex> lk(c->cam_mu);
        if (sw == w && sh == h) {
            memcpy(c->cam_buf.data(), src, (size_t)w * h * 3);
        } else { /* nearest-neighbour resize */
            for (int y = 0; y < h; y++)
                for (int x = 0; x < w; x++)
                    memcpy(&c->cam_buf[(y * w + x) * 3], src + ((y * sh / h) * sw + x * sw / w) * 3, 3);
        }
        c->cam_seq++;
    }
}

void camera_close(void *p);

int camera_open(void *p, int w, int h)
{
    auto *c = static_cast<Ctx *>(p);
    camera_close(p);
    try {
        c->cam.reset(new camera::Camera(w, h, image::FMT_RGB888, nullptr, c->cam_fps > 0 ? c->cam_fps : -1));
        if (!c->cam->is_opened()) {
            c->cam.reset();
            return -1;
        }
        if (c->cam_max_exp_us > 0) {
            c->cam->exp_mode(camera::AeMode::Manual);
            c->ae_exp = c->cam_max_exp_us;
            c->ae_gain = 4 * GAIN_1X; /* a middle start; the loop settles in a few frames */
            c->cam->exposure(c->ae_exp);
            c->cam->gain(c->ae_gain);
        }
    } catch (std::exception &e) {
        log::warn("duck_dash: camera: %s", e.what());
        c->cam.reset();
        return -1;
    }
    c->cam_buf.assign((size_t)w * h * 3, 0);
    c->cam_seq = c->cam_seq_read = 0;
    c->cam_run = true;
    c->cam_thread = std::thread(camera_loop, c, w, h);
    return 0;
}

int camera_read(void *p, uint8_t *rgb, int w, int h)
{
    auto *c = static_cast<Ctx *>(p);
    if (!c->cam) return -1;
    std::lock_guard<std::mutex> lk(c->cam_mu);
    if (c->cam_seq == c->cam_seq_read || c->cam_buf.size() != (size_t)w * h * 3) return 1; /* nothing new */
    memcpy(rgb, c->cam_buf.data(), c->cam_buf.size());
    c->cam_seq_read = c->cam_seq;
    return 0;
}

void camera_close(void *p)
{
    auto *c = static_cast<Ctx *>(p);
    c->cam_run = false;
    if (c->cam_thread.joinable()) c->cam_thread.join();
    c->cam.reset();
}

int npu_probe(void *, const char *mud, int in_size, float *load_ms, float *infer_ms, char *msg, int msglen)
{
    try {
        uint64_t t0 = time::ticks_us();
        nn::NN model(mud);
        if (!model.loaded()) {
            snprintf(msg, msglen, "NPU could not load the model");
            return -1;
        }
        *load_ms = (time::ticks_us() - t0) / 1000.0f;
        std::vector<nn::LayerInfo> ins = model.inputs_info();
        if (ins.empty()) {
            snprintf(msg, msglen, "model has no inputs");
            return -1;
        }
        std::vector<int> shape = ins[0].shape;
        int n = 1;
        for (int d : shape) n *= d;
        if (n != in_size) {
            snprintf(msg, msglen, "model wants %d inputs, policy has %d", n, in_size);
            return -1;
        }
        std::vector<float> zeros(n, 0.0f);
        float best = 1e9f;
        for (int i = 0; i < 5; i++) { /* first run warms up */
            tensor::Tensors inputs;
            inputs.add_tensor(ins[0].name, new tensor::Tensor(shape, tensor::FLOAT32, zeros.data(), true), false, true);
            uint64_t t = time::ticks_us();
            tensor::Tensors *out = model.forward(inputs);
            float ms = (time::ticks_us() - t) / 1000.0f;
            if (!out) {
                snprintf(msg, msglen, "NPU forward failed");
                return -1;
            }
            delete out;
            if (i && ms < best) best = ms;
        }
        *infer_ms = best;
    } catch (std::exception &e) {
        snprintf(msg, msglen, "%s", e.what());
        return -1;
    }
    return 0;
}

int pinmap(void *, const char *pin, const char *func)
{
    return peripheral::pinmap::set_pin_function(pin, func) == err::ERR_NONE ? 0 : -1;
}

} // namespace

static int run(int argc, char **argv)
{
    Ctx ctx;
    dash_platform plat{};
    plat.ctx = &ctx;
    plat.name = "maixcam2";
    plat.display_open = display_open;
    plat.display_show = display_show;
    plat.touch_tap = touch_tap;
    plat.need_exit = need_exit;
    plat.camera_config = camera_config;
    plat.camera_open = camera_open;
    plat.camera_read = camera_read;
    plat.camera_close = camera_close;
    plat.npu_probe = npu_probe;
    plat.pinmap = pinmap;
    return dash_main(argc, argv, &plat);
}

int main(int argc, char *argv[])
{
    sys::register_default_signal_handle();
    CATCH_EXCEPTION_RUN_RETURN(run, -1, argc, argv);
}
