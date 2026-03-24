#pragma once

// lv_conf.h is included internally by lvgl.h
#include <lvgl.h>

// #include <demos/lv_demos.h>  // Uncomment if using lv_demo_* functions
// directly
#include "Display_ST7701.h"
#include "Touch_CST820.h"
#include <esp_heap_caps.h>

#define LCD_WIDTH ESP_PANEL_LCD_WIDTH
#define LCD_HEIGHT ESP_PANEL_LCD_HEIGHT
#define LVGL_BUF_LEN (ESP_PANEL_LCD_WIDTH * ESP_PANEL_LCD_HEIGHT / 5)

#define EXAMPLE_LVGL_TICK_PERIOD_MS 2

extern lv_disp_drv_t disp_drv;

void Lvgl_print(const char *buf);
void Lvgl_Display_LCD(
    lv_disp_drv_t *disp_drv, const lv_area_t *area,
    lv_color_t *color_p); // Displays LVGL content on the LCD.    This function
                          // implements associating LVGL data to the LCD screen
void Lvgl_Touchpad_Read(lv_indev_drv_t *indev_drv,
                        lv_indev_data_t *data); // Read the touchpad
void example_increase_lvgl_tick(void *arg);

void Lvgl_Init(void);
void Lvgl_Loop(void);
