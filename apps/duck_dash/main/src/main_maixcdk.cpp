/* MaixCDK platform for duck_dash: the MaixCAM2 screen, touch panel, camera,
 * NPU and pin functions. Everything else is the C core in src/core.
 *
 * This file is C++ only because Sipeed's SDK (MaixCDK) is a C++ API; it
 * hands frames and touches to the C core and does no drawing itself. */
#include <cstring>
#include <memory>
#include <string>
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

struct Ctx {
    std::unique_ptr<display::Display> disp;
    std::unique_ptr<touchscreen::TouchScreen> touch;
    std::unique_ptr<camera::Camera> cam;
    bool was_pressed = false;
};

int display_open(void *p, int *w, int *h)
{
    auto *c = static_cast<Ctx *>(p);
    try {
        c->disp.reset(new display::Display());
        if (!c->disp->is_opened()) return -1;
        *w = c->disp->width();
        *h = c->disp->height();
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
    /* wraps our buffer without a copy; show() scales 640x480 to the panel */
    try {
        image::Image img(w, h, image::FMT_RGB888, const_cast<uint8_t *>(rgb), w * h * 3, false);
        c->disp->show(img, image::FIT_CONTAIN);
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

int camera_open(void *p, int w, int h)
{
    auto *c = static_cast<Ctx *>(p);
    try {
        c->cam.reset(new camera::Camera(w, h, image::FMT_RGB888));
        if (!c->cam->is_opened()) {
            c->cam.reset();
            return -1;
        }
    } catch (std::exception &e) {
        log::warn("duck_dash: camera: %s", e.what());
        c->cam.reset();
        return -1;
    }
    return 0;
}

int camera_read(void *p, uint8_t *rgb, int w, int h)
{
    auto *c = static_cast<Ctx *>(p);
    if (!c->cam) return -1;
    std::unique_ptr<image::Image> img;
    try {
        img.reset(c->cam->read(false)); /* non-blocking: never stall the screen */
    } catch (std::exception &e) {
        return -1;
    }
    if (!img || img->format() != image::FMT_RGB888) return -1;
    int sw = img->width(), sh = img->height();
    const uint8_t *src = static_cast<const uint8_t *>(img->data());
    if (sw == w && sh == h) {
        memcpy(rgb, src, (size_t)w * h * 3);
    } else { /* nearest-neighbour resize */
        for (int y = 0; y < h; y++)
            for (int x = 0; x < w; x++) memcpy(rgb + (y * w + x) * 3, src + ((y * sh / h) * sw + x * sw / w) * 3, 3);
    }
    return 0;
}

void camera_close(void *p)
{
    static_cast<Ctx *>(p)->cam.reset();
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
