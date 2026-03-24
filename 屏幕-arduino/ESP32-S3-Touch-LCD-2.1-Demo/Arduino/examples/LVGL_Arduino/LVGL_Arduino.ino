/*
 * Blinking Eyes Demo – ESP32-S3-Touch-LCD-2.1 (480×480)
 * Only this file is modified; all other driver files are untouched.
 */

// ── Hardware drivers (unchanged from original project) ──────────────
#include "BAT_Driver.h"
#include "Gyro_QMI8658.h"
#include "LVGL_Driver.h" // provides LCD_Init(), Lvgl_Init(), Lvgl_Loop()
#include "RTC_PCF85063.h"
#include "SD_Card.h"
#include "Wireless.h"

// ════════════════════════════════════════════════════════════════════
//  Eye geometry  — display is 480×480
// ════════════════════════════════════════════════════════════════════
#define EYE_W 160       // eye width
#define EYE_H_OPEN 160  // eye height when fully open
#define EYE_H_CLOSED 10 // eye height when fully closed (thin slit)

#define EYE_L_OFS -100 // left  eye X offset from screen centre
#define EYE_R_OFS 100  // right eye X offset from screen centre

static lv_obj_t *left_eye, *right_eye; // eye containers (height animated)

// ════════════════════════════════════════════════════════════════════
//  Animation callback – squishes eye height to simulate blink
// ════════════════════════════════════════════════════════════════════
static void eye_height_anim_cb(void *var, int32_t v) {
  lv_obj_t *eye = (lv_obj_t *)var;
  lv_obj_set_height(eye, (lv_coord_t)v);

  // Re-align so the eye stays centred vertically while squishing
  lv_coord_t xofs = (eye == left_eye) ? EYE_L_OFS : EYE_R_OFS;
  lv_obj_align(eye, LV_ALIGN_CENTER, xofs, 0);
}

// ════════════════════════════════════════════════════════════════════
//  Trigger one blink on both eyes simultaneously
// ════════════════════════════════════════════════════════════════════
static void blink_action(void) {
  lv_anim_t a;
  lv_anim_init(&a);
  lv_anim_set_exec_cb(&a, eye_height_anim_cb);
  lv_anim_set_values(&a, EYE_H_OPEN, EYE_H_CLOSED); // squish
  lv_anim_set_time(&a, 120);                        // 120 ms close
  lv_anim_set_playback_time(&a, 120);               // 120 ms re-open
  lv_anim_set_path_cb(&a, lv_anim_path_ease_in_out);

  lv_anim_set_var(&a, left_eye);
  lv_anim_start(&a);

  lv_anim_set_var(&a, right_eye);
  lv_anim_start(&a);
}

// ════════════════════════════════════════════════════════════════════
//  Periodic timer – blinks at a random interval [1.5 s … 4.5 s]
// ════════════════════════════════════════════════════════════════════
static void blink_timer_cb(lv_timer_t *timer) {
  blink_action();
  lv_timer_set_period(timer, (uint32_t)random(1500, 4500));
}

// ════════════════════════════════════════════════════════════════════
//  Build helper: one eye container + pupil, centred at xofs
// ════════════════════════════════════════════════════════════════════
static lv_obj_t *make_eye(lv_coord_t xofs) {
  // Outer white circle (the sclera / white of the eye)
  lv_obj_t *eye = lv_obj_create(lv_scr_act());
  lv_obj_set_size(eye, EYE_W, EYE_H_OPEN);
  lv_obj_align(eye, LV_ALIGN_CENTER, xofs, 0);
  lv_obj_set_style_radius(eye, LV_RADIUS_CIRCLE, 0);
  lv_obj_set_style_bg_color(eye, lv_color_white(), 0);
  lv_obj_set_style_bg_opa(eye, LV_OPA_COVER, 0);
  lv_obj_set_style_border_width(eye, 0, 0);
  lv_obj_set_style_shadow_color(eye, lv_color_make(0, 0, 100), 0);
  lv_obj_set_style_shadow_width(eye, 20, 0);
  lv_obj_set_style_shadow_opa(eye, LV_OPA_50, 0);
  lv_obj_set_style_pad_all(eye, 0, 0);
  lv_obj_clear_flag(eye, LV_OBJ_FLAG_SCROLLABLE);

  // (no pupil — pure white sclera only)

  return eye;
}

// ════════════════════════════════════════════════════════════════════
//  Create the full eye scene
// ════════════════════════════════════════════════════════════════════
static void setup_eyes(void) {
  randomSeed(42);

  // Dark background
  lv_obj_set_style_bg_color(lv_scr_act(), lv_color_make(18, 18, 35), 0);
  lv_obj_set_style_bg_opa(lv_scr_act(), LV_OPA_COVER, 0);

  left_eye = make_eye(EYE_L_OFS);
  right_eye = make_eye(EYE_R_OFS);

  // Caption
  lv_obj_t *lbl = lv_label_create(lv_scr_act());
  lv_label_set_text(lbl, "Hello!  ( ^ . ^ )");
  lv_obj_set_style_text_color(lbl, lv_color_make(180, 180, 255), 0);
  lv_obj_set_style_text_font(lbl, &lv_font_montserrat_14, 0);
  lv_obj_align(lbl, LV_ALIGN_BOTTOM_MID, 0, -30);

  // Start blink timer (first blink after 2 s)
  lv_timer_create(blink_timer_cb, 2000, NULL);
}

// ════════════════════════════════════════════════════════════════════
//  Original driver boilerplate (unchanged)
// ════════════════════════════════════════════════════════════════════
void Driver_Loop(void *parameter) {
  while (1) {
    QMI8658_Loop();
    RTC_Loop();
    BAT_Get_Volts();
    vTaskDelay(pdMS_TO_TICKS(100));
  }
}

void Driver_Init() {
  Flash_test();
  BAT_Init();
  I2C_Init();
  TCA9554PWR_Init(0x00);
  Set_EXIO(EXIO_PIN8, Low);
  PCF85063_Init();
  QMI8658_Init();
  xTaskCreatePinnedToCore(Driver_Loop, "Other Driver task", 4096, NULL, 3, NULL,
                          0);
}

void setup() {
  Wireless_Test2();
  Driver_Init();
  LCD_Init(); // ← hardware display init (ST7701 via LVGL_Driver.h)
  SD_Init();
  Lvgl_Init(); // ← registers LVGL display & touch drivers

  setup_eyes(); // ← our eye animation (replaces Lvgl_Example1)
}

void loop() {
  Lvgl_Loop(); // ← drives the LVGL task handler
  vTaskDelay(pdMS_TO_TICKS(5));
}
