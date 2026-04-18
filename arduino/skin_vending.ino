/*
 * ============================================================================
 *  SKIN ANALYSIS SMART VENDING SYSTEM
 * ============================================================================
 *  Hardware:
 *    - Arduino Mega 2560
 *    - TFT 3.95" ILI9488 (LCDWIKI shield)
 *    - Resistive touch panel (on same shield)
 *    - L298N motor driver x4 channels (2x L298N boards)
 *
 *  Serial protocol (newline-terminated):
 *    Python -> Arduino:   "RESULT:OILY,ACNE,DARK_SPOTS"
 *    Arduino -> Python:   "REQ:SCAN"   "READY"   "DISPENSING:1"   "DONE:1"
 * ============================================================================
 */

#include <LCDWIKI_GUI.h>
#include <LCDWIKI_KBV.h>
#include <LCDWIKI_TOUCH.h>
#include <string.h>

// ── Motor pins (4 products) ────────────────────────────────────────────
struct MotorPins { uint8_t en, in1, in2; };

const MotorPins MOTORS[4] = {
    {2, 22, 23},   // Motor 1: Cleanser
    {3, 24, 25},   // Motor 2: Toner
    {4, 26, 27},   // Motor 3: Moisturizer
    {5, 28, 29},   // Motor 4: Sunscreen
};

const char* PRODUCT_NAMES[4] = {"Cleanser", "Toner", "Moisturizer", "Sunscreen"};
const uint8_t  NUM_PRODUCTS     = 4;
const uint8_t  MOTOR_SPEED      = 255;
const uint16_t DISPENSE_TIME_MS = 2500;
const uint16_t MOTOR_PAUSE_MS   = 500;

// ── TFT + Touch ────────────────────────────────────────────────────────
LCDWIKI_KBV lcd(ILI9488, 40, 38, 39, 43, 41);
LCDWIKI_TOUCH touch(53, 52, 50, 51, 44);

// ── Colors ─────────────────────────────────────────────────────────────
#define C_BG      0xEF3C
#define C_PANEL   0xFFFF
#define C_NAVY    0x0010
#define C_WHITE   0xFFFF
#define C_TEXT    0x18C6
#define C_BORDER  0x9CD3
#define C_GREEN   0x3D8E
#define C_RED     0xE0C3
#define C_BLUE    0x041F
#define C_SKY     0x867D
#define C_YELLOW  0xFEA0
#define C_ORANGE  0xFCA0
#define C_PURPLE  0x780F
#define C_DARK    0x2104

// ── Touch config ───────────────────────────────────────────────────────
const uint8_t  TOUCH_SAMPLES      = 16;
const uint8_t  TOUCH_SAMPLE_DELAY = 2;
const uint8_t  TOUCH_MIN_GOOD     = 5;
const uint8_t  TOUCH_ROTATION     = 3;

// Set these to true if axes are mirrored on your panel
bool TOUCH_SWAP_XY  = false;
bool TOUCH_INVERT_X = false;
bool TOUCH_INVERT_Y = false;

// Debug: set true to see touch crosshair + coordinates on screen
bool TOUCH_DEBUG = false;

unsigned long lastTouchMs = 0;
const unsigned long DEBOUNCE_MS = 350;

// ── Screen state ───────────────────────────────────────────────────────
enum Screen {
    SCR_WELCOME   = 0,
    SCR_MAIN      = 1,
    SCR_SCANNING  = 2,
    SCR_RESULT    = 3,
    SCR_RECOMMEND = 4,
    SCR_SELECT    = 5,
    SCR_DISPENSING= 6,
};

Screen currentScreen = SCR_WELCOME;
int16_t px = -1, py = -1;
int16_t LCD_W = 0, LCD_H = 0;

// ── Skin analysis data ─────────────────────────────────────────────────
String skinType      = "";
String issues[5];
uint8_t issueCount   = 0;
bool recommended[4]  = {false, false, false, false};
uint8_t recommendCount = 0;

String serialBuf = "";

// ══════════════════════════════════════════════════════════════════════
//  MOTOR CONTROL
// ══════════════════════════════════════════════════════════════════════

void motorSetup() {
    for (uint8_t i = 0; i < NUM_PRODUCTS; i++) {
        pinMode(MOTORS[i].en,  OUTPUT);
        pinMode(MOTORS[i].in1, OUTPUT);
        pinMode(MOTORS[i].in2, OUTPUT);
        motorStop(i);
    }
}
void motorStop(uint8_t id) {
    digitalWrite(MOTORS[id].in1, LOW);
    digitalWrite(MOTORS[id].in2, LOW);
    analogWrite(MOTORS[id].en, 0);
}
void motorForward(uint8_t id) {
    digitalWrite(MOTORS[id].in1, HIGH);
    digitalWrite(MOTORS[id].in2, LOW);
    analogWrite(MOTORS[id].en, MOTOR_SPEED);
}
void dispenseProduct(uint8_t id) {
    if (id >= NUM_PRODUCTS) return;
    Serial.print("DISPENSING:"); Serial.println(id + 1);
    motorForward(id);
    delay(DISPENSE_TIME_MS);
    motorStop(id);
    delay(MOTOR_PAUSE_MS);
    Serial.print("DONE:"); Serial.println(id + 1);
}

// ══════════════════════════════════════════════════════════════════════
//  RECOMMENDATION ENGINE
// ══════════════════════════════════════════════════════════════════════

bool hasIssue(const char* name) {
    for (uint8_t i = 0; i < issueCount; i++)
        if (issues[i] == name) return true;
    return false;
}

void buildRecommendations() {
    recommendCount = 0;
    for (uint8_t i = 0; i < NUM_PRODUCTS; i++) recommended[i] = false;
    if (skinType == "OILY" || hasIssue("ACNE") || hasIssue("LARGE_PORES")) recommended[0] = true;
    if (skinType == "OILY" || skinType == "NORMAL" || hasIssue("REDNESS"))  recommended[1] = true;
    if (skinType == "DRY"  || hasIssue("WRINKLES") || hasIssue("DARK_SPOTS")) recommended[2] = true;
    recommended[3] = true;  // sunscreen always
    for (uint8_t i = 0; i < NUM_PRODUCTS; i++) if (recommended[i]) recommendCount++;
}

// ══════════════════════════════════════════════════════════════════════
//  SERIAL PARSER
// ══════════════════════════════════════════════════════════════════════

void parseSkinResult(const String& payload) {
    skinType = ""; issueCount = 0;
    for (uint8_t i = 0; i < 5; i++) issues[i] = "";
    int idx = 0, start = 0;
    for (int i = 0; i <= (int)payload.length(); i++) {
        if (i == (int)payload.length() || payload.charAt(i) == ',') {
            String token = payload.substring(start, i);
            token.trim(); token.toUpperCase();
            if (idx == 0) skinType = token;
            else if (issueCount < 5) { issues[issueCount] = token; issueCount++; }
            idx++; start = i + 1;
        }
    }
}

void parseSerialCommand(const String& cmd) {
    String c = cmd; c.trim();
    Serial.print("RX:"); Serial.println(c);

    if (c.startsWith("RESULT:")) {
        parseSkinResult(c.substring(7));
        buildRecommendations();
        currentScreen = SCR_RESULT;
        drawResultScreen();
    } else if (c == "TEST") {
        parseSkinResult("OILY,ACNE,LARGE_PORES");
        buildRecommendations();
        currentScreen = SCR_RESULT;
        drawResultScreen();
        Serial.println("OK:TEST");
    } else if (c == "DEBUG") {
        TOUCH_DEBUG = !TOUCH_DEBUG;
        Serial.print("TOUCH_DEBUG="); Serial.println(TOUCH_DEBUG ? "ON" : "OFF");
    } else if (c.startsWith("VEND:")) {
        String what = c.substring(5); what.trim();
        if (what == "ALL") { dispenseAllRecommended(); }
        else {
            int id = what.toInt();
            if (id >= 1 && id <= NUM_PRODUCTS) {
                drawDispensingScreen(id - 1);
                dispenseProduct(id - 1);
                currentScreen = SCR_MAIN; drawMainMenu();
            }
        }
    }
}

// ══════════════════════════════════════════════════════════════════════
//  DISPENSING LOGIC
// ══════════════════════════════════════════════════════════════════════

void dispenseAllRecommended() {
    for (uint8_t i = 0; i < NUM_PRODUCTS; i++) {
        if (!recommended[i]) continue;
        drawDispensingScreen(i);
        dispenseProduct(i);
    }
    Serial.println("DONE:ALL");
    currentScreen = SCR_MAIN;
    drawMainMenu();
}

void dispenseSingleProduct(uint8_t id) {
    drawDispensingScreen(id);
    dispenseProduct(id);
    currentScreen = SCR_SELECT;
    drawSelectScreen();
}

// ══════════════════════════════════════════════════════════════════════
//  UI DRAWING PRIMITIVES
// ══════════════════════════════════════════════════════════════════════

void drawHeader(const char* title) {
    lcd.Set_Draw_color(C_NAVY);
    lcd.Fill_Rectangle(0, 0, LCD_W, 44);
    lcd.Set_Text_Size(2);
    lcd.Set_Text_colour(C_WHITE);
    lcd.Set_Text_Back_colour(C_NAVY);
    lcd.Print_String((char*)title, 12, 12);
}

void drawCard(int x1, int y1, int x2, int y2, uint16_t fill, uint16_t border) {
    lcd.Set_Draw_color(fill);
    lcd.Fill_Round_Rectangle(x1, y1, x2, y2, 8);
    lcd.Set_Draw_color(border);
    lcd.Draw_Round_Rectangle(x1, y1, x2, y2, 8);
}

void drawButton(int x, int y, int w, int h,
                uint16_t fill, uint16_t border, uint16_t textCol, const char* label) {
    drawCard(x, y, x + w, y + h, fill, border);
    int textW = (int)strlen(label) * 12;
    int tx = x + (w - textW) / 2;
    if (tx < x + 4) tx = x + 4;
    int ty = y + (h - 16) / 2;
    lcd.Set_Text_Size(2);
    lcd.Set_Text_colour(textCol);
    lcd.Set_Text_Back_colour(fill);
    lcd.Print_String((char*)label, tx, ty);
}

void drawLabel(int x, int y, const char* text, uint8_t sz, uint16_t col, uint16_t bg) {
    lcd.Set_Text_Size(sz);
    lcd.Set_Text_colour(col);
    lcd.Set_Text_Back_colour(bg);
    lcd.Print_String((char*)text, x, y);
}

// ══════════════════════════════════════════════════════════════════════
//  SCREEN DRAWING — each screen defines its own button layout struct
// ══════════════════════════════════════════════════════════════════════

// A simple button rectangle for hit testing
struct BtnRect { int x, y, w, h; };

// ── Welcome ─────────────────────────────────────────────────────────
const BtnRect BTN_WELCOME_START = {60, 300, 200, 60};

void drawWelcomeScreen() {
    lcd.Fill_Screen(C_BG);
    drawHeader("Skin Smart Vend");
    drawCard(18, 80, 302, 420, C_PANEL, C_BORDER);
    drawLabel(80, 110, "Welcome!", 2, C_NAVY, C_PANEL);
    drawLabel(40, 170, "Smart skincare vending", 1, C_TEXT, C_PANEL);
    drawLabel(40, 190, "system with AI-powered", 1, C_TEXT, C_PANEL);
    drawLabel(40, 210, "skin analysis.", 1, C_TEXT, C_PANEL);
    drawButton(BTN_WELCOME_START.x, BTN_WELCOME_START.y,
               BTN_WELCOME_START.w, BTN_WELCOME_START.h,
               C_GREEN, C_BORDER, C_WHITE, "START");
}

// ── Main Menu ───────────────────────────────────────────────────────
// 3 buttons with large 40px gaps between them
const BtnRect BTN_MAIN[3] = {
    {35,  95, 250, 60},   // Scan My Skin     (y  95 → 155)
    {35, 195, 250, 60},   // Pick Product      (y 195 → 255)  gap = 40px
    {35, 295, 250, 60},   // Take All (Auto)   (y 295 → 355)  gap = 40px
};

void drawMainMenu() {
    lcd.Fill_Screen(C_BG);
    drawHeader("Main Menu");
    drawCard(18, 56, 302, 460, C_PANEL, C_BORDER);
    drawLabel(50, 66, "Choose an option:", 2, C_TEXT, C_PANEL);

    drawButton(BTN_MAIN[0].x, BTN_MAIN[0].y, BTN_MAIN[0].w, BTN_MAIN[0].h,
               C_GREEN, C_BORDER, C_WHITE, "Scan My Skin");
    drawButton(BTN_MAIN[1].x, BTN_MAIN[1].y, BTN_MAIN[1].w, BTN_MAIN[1].h,
               C_BLUE, C_BORDER, C_WHITE, "Pick Product");
    drawButton(BTN_MAIN[2].x, BTN_MAIN[2].y, BTN_MAIN[2].w, BTN_MAIN[2].h,
               C_YELLOW, C_BORDER, C_NAVY, "Take All (Auto)");

    drawLabel(38, 380, "Scan: AI skin analysis", 1, C_TEXT, C_PANEL);
    drawLabel(38, 398, "Pick: choose a product", 1, C_TEXT, C_PANEL);
    drawLabel(38, 416, "Auto: dispense default set", 1, C_TEXT, C_PANEL);
}

// ── Scanning ────────────────────────────────────────────────────────
const BtnRect BTN_SCAN_CANCEL = {35, 360, 250, 55};

void drawScanningScreen() {
    lcd.Fill_Screen(C_BG);
    drawHeader("Scanning...");
    drawCard(18, 100, 302, 330, C_PANEL, C_BORDER);
    drawLabel(56, 150, "Analyzing skin...", 2, C_NAVY, C_PANEL);
    drawLabel(40, 200, "Please look at the camera.", 1, C_TEXT, C_PANEL);
    drawLabel(40, 220, "Result will appear shortly.", 1, C_TEXT, C_PANEL);
    drawButton(BTN_SCAN_CANCEL.x, BTN_SCAN_CANCEL.y,
               BTN_SCAN_CANCEL.w, BTN_SCAN_CANCEL.h,
               C_RED, C_BORDER, C_WHITE, "CANCEL");
}

// ── Result ──────────────────────────────────────────────────────────
const BtnRect BTN_RESULT[2] = {
    {35, 320, 250, 55},   // See Products      (y 320 → 375)
    {35, 410, 250, 55},   // Back to Menu       (y 410 → 465)  gap = 35px
};

void drawResultScreen() {
    lcd.Fill_Screen(C_BG);
    drawHeader("Skin Analysis");
    drawCard(18, 56, 302, 305, C_PANEL, C_BORDER);

    drawLabel(35, 72, "Skin Type:", 2, C_TEXT, C_PANEL);
    uint16_t typeColor = C_NAVY;
    if (skinType == "OILY")   typeColor = C_BLUE;
    if (skinType == "DRY")    typeColor = C_ORANGE;
    if (skinType == "NORMAL") typeColor = C_GREEN;
    drawLabel(170, 72, skinType.c_str(), 2, typeColor, C_PANEL);

    drawLabel(35, 110, "Conditions:", 2, C_TEXT, C_PANEL);
    if (issueCount == 0) {
        drawLabel(35, 140, "  None detected", 1, C_GREEN, C_PANEL);
    } else {
        int yy = 140;
        for (uint8_t i = 0; i < issueCount && i < 5; i++) {
            String nice = "  - " + formatIssue(issues[i]);
            drawLabel(35, yy, nice.c_str(), 1, C_RED, C_PANEL);
            yy += 18;
        }
    }
    drawLabel(35, 268, "Products recommended:", 1, C_TEXT, C_PANEL);
    char buf[8]; sprintf(buf, "%d of 4", recommendCount);
    drawLabel(200, 268, buf, 1, C_NAVY, C_PANEL);

    drawButton(BTN_RESULT[0].x, BTN_RESULT[0].y, BTN_RESULT[0].w, BTN_RESULT[0].h,
               C_GREEN, C_BORDER, C_WHITE, "See Products");
    drawButton(BTN_RESULT[1].x, BTN_RESULT[1].y, BTN_RESULT[1].w, BTN_RESULT[1].h,
               C_SKY, C_BORDER, C_NAVY, "Back to Menu");
}

// ── Recommend ───────────────────────────────────────────────────────
const BtnRect BTN_REC[3] = {
    {35,  385, 120, 50},  // TAKE ALL
    {165, 385, 120, 50},  // SELECT
    {95,  450, 130, 35},  // BACK
};

void drawRecommendScreen() {
    lcd.Fill_Screen(C_BG);
    drawHeader("Recommended");
    drawCard(18, 56, 302, 375, C_PANEL, C_BORDER);
    drawLabel(35, 72, "For your skin:", 2, C_TEXT, C_PANEL);

    int yy = 108;
    for (uint8_t i = 0; i < NUM_PRODUCTS; i++) {
        uint16_t tc = recommended[i] ? C_NAVY : C_BORDER;
        char num[4]; sprintf(num, "%d.", i + 1);
        String line = String(num) + " " + PRODUCT_NAMES[i];
        if (recommended[i]) line += "  *";
        drawLabel(45, yy, line.c_str(), 2, tc, C_PANEL);
        yy += 34;
    }
    drawLabel(45, yy + 6, "* = recommended", 1, C_TEXT, C_PANEL);

    drawButton(BTN_REC[0].x, BTN_REC[0].y, BTN_REC[0].w, BTN_REC[0].h,
               C_GREEN, C_BORDER, C_WHITE, "TAKE ALL");
    drawButton(BTN_REC[1].x, BTN_REC[1].y, BTN_REC[1].w, BTN_REC[1].h,
               C_BLUE, C_BORDER, C_WHITE, "SELECT");
    drawButton(BTN_REC[2].x, BTN_REC[2].y, BTN_REC[2].w, BTN_REC[2].h,
               C_RED, C_BORDER, C_WHITE, "BACK");
}

// ── Select Product ──────────────────────────────────────────────────
// 4 product buttons + BACK, spaced 80px apart with 50px height = 30px gap
const int SEL_Y0 = 95;
const int SEL_SP = 80;
const int SEL_H  = 50;
const int SEL_X  = 35;
const int SEL_W  = 250;
const BtnRect BTN_SEL_BACK = {95, 430, 130, 35};

void drawSelectScreen() {
    lcd.Fill_Screen(C_BG);
    drawHeader("Pick Product");
    drawCard(18, 56, 302, 470, C_PANEL, C_BORDER);
    drawLabel(40, 66, "Tap to dispense:", 2, C_TEXT, C_PANEL);

    uint16_t btnC[4] = {C_GREEN, C_BLUE, C_PURPLE, C_YELLOW};
    uint16_t txtC[4] = {C_WHITE, C_WHITE, C_WHITE, C_NAVY};

    for (uint8_t i = 0; i < NUM_PRODUCTS; i++) {
        int y = SEL_Y0 + i * SEL_SP;
        drawButton(SEL_X, y, SEL_W, SEL_H, btnC[i], C_BORDER, txtC[i], PRODUCT_NAMES[i]);
    }

    drawButton(BTN_SEL_BACK.x, BTN_SEL_BACK.y, BTN_SEL_BACK.w, BTN_SEL_BACK.h,
               C_RED, C_BORDER, C_WHITE, "BACK");
}

// ── Dispensing ──────────────────────────────────────────────────────
void drawDispensingScreen(uint8_t productId) {
    lcd.Fill_Screen(C_BG);
    drawHeader("Dispensing...");
    drawCard(18, 100, 302, 320, C_PANEL, C_BORDER);
    drawLabel(60, 150, "Dispensing:", 2, C_NAVY, C_PANEL);
    drawLabel(60, 185, PRODUCT_NAMES[productId], 2, C_GREEN, C_PANEL);
    lcd.Set_Draw_color(C_BORDER);
    lcd.Fill_Rectangle(50, 240, 270, 258);
    lcd.Set_Draw_color(C_GREEN);
    lcd.Fill_Rectangle(50, 240, 160, 258);
    drawLabel(70, 280, "Please wait...", 1, C_TEXT, C_PANEL);
}

// ══════════════════════════════════════════════════════════════════════
//  HELPERS
// ══════════════════════════════════════════════════════════════════════

String formatIssue(const String& raw) {
    String out = raw;
    out.toLowerCase();
    for (int i = 0; i < (int)out.length(); i++)
        if (out.charAt(i) == '_') out.setCharAt(i, ' ');
    if (out.length() > 0) {
        out.setCharAt(0, toupper(out.charAt(0)));
        for (int i = 1; i < (int)out.length(); i++)
            if (out.charAt(i - 1) == ' ') out.setCharAt(i, toupper(out.charAt(i)));
    }
    return out;
}

// ── Touch ───────────────────────────────────────────────────────────

bool readTouch() {
    if (millis() - lastTouchMs < DEBOUNCE_MS) return false;

    touch.TP_Scan(0);
    if (!(touch.TP_Get_State() & TP_PRES_DOWN)) return false;

    int32_t ax = 0, ay = 0;
    uint8_t good = 0;

    for (uint8_t i = 0; i < TOUCH_SAMPLES; i++) {
        touch.TP_Scan(0);
        if (!(touch.TP_Get_State() & TP_PRES_DOWN)) break;
        uint16_t rx = touch.x, ry = touch.y;
        if (rx == 0xFFFF || ry == 0xFFFF) { delay(TOUCH_SAMPLE_DELAY); continue; }
        ax += rx; ay += ry; good++;
        delay(TOUCH_SAMPLE_DELAY);
    }
    if (good < TOUCH_MIN_GOOD) return false;

    int16_t tx = (int16_t)(ax / good);
    int16_t ty = (int16_t)(ay / good);

    if (TOUCH_SWAP_XY) { int16_t t = tx; tx = ty; ty = t; }
    if (TOUCH_INVERT_X) tx = LCD_W - tx;
    if (TOUCH_INVERT_Y) ty = LCD_H - ty;

    px = constrain(tx, 0, LCD_W);
    py = constrain(ty, 0, LCD_H);
    lastTouchMs = millis();

    // Debug: show where we detected the touch
    if (TOUCH_DEBUG) {
        Serial.print("TOUCH x="); Serial.print(px);
        Serial.print(" y="); Serial.println(py);
        // Draw crosshair at touch point
        lcd.Set_Draw_color(C_RED);
        lcd.Draw_Fast_HLine(px - 10, py, 20);
        lcd.Draw_Fast_VLine(px, py - 10, 20);
    }

    return true;
}

bool hitBtn(const BtnRect& b) {
    int x2 = b.x + b.w;
    int y2 = b.y + b.h;
    return (px >= b.x && px <= x2 && py >= b.y && py <= y2);
}

bool hitRect(int x, int y, int w, int h) {
    return (px >= x && px <= x + w && py >= y && py <= y + h);
}

void waitRelease() {
    delay(180);
    unsigned long t0 = millis();
    do {
        touch.TP_Scan(0);
        if (millis() - t0 > 2000) break;
    } while (touch.TP_Get_State() & TP_PRES_DOWN);
    delay(120);
    lastTouchMs = millis();
}

void flashBtn(const BtnRect& b) {
    lcd.Set_Draw_color(C_WHITE);
    lcd.Draw_Round_Rectangle(b.x, b.y, b.x + b.w, b.y + b.h, 8);
    delay(80);
}

void flashRect(int x, int y, int w, int h) {
    lcd.Set_Draw_color(C_WHITE);
    lcd.Draw_Round_Rectangle(x, y, x + w, y + h, 8);
    delay(80);
}

// ══════════════════════════════════════════════════════════════════════
//  SETUP
// ══════════════════════════════════════════════════════════════════════

void setup() {
    Serial.begin(9600);
    motorSetup();

    lcd.Init_LCD();
    lcd.Set_Rotation(0);
    LCD_W = lcd.Get_Display_Width() - 1;
    LCD_H = lcd.Get_Display_Height() - 1;

    touch.TP_Set_Rotation(TOUCH_ROTATION);
    touch.TP_Init(lcd.Get_Rotation(), lcd.Get_Display_Width(), lcd.Get_Display_Height());

    drawWelcomeScreen();
    Serial.println("READY");
    Serial.print("LCD: "); Serial.print(LCD_W); Serial.print("x"); Serial.println(LCD_H);
}

// ══════════════════════════════════════════════════════════════════════
//  MAIN LOOP
// ══════════════════════════════════════════════════════════════════════

void loop() {
    // ── Serial (non-blocking) ──────────────────────────────────────
    while (Serial.available() > 0) {
        char ch = (char)Serial.read();
        if (ch == '\n' || ch == '\r') {
            if (serialBuf.length() > 0) {
                parseSerialCommand(serialBuf);
                serialBuf = "";
            }
        } else {
            serialBuf += ch;
        }
    }

    // ── Touch ──────────────────────────────────────────────────────
    px = -1; py = -1;
    if (!readTouch()) return;

    switch (currentScreen) {

    // ── Welcome ────────────────────────────────────────────────────
    case SCR_WELCOME:
        if (hitBtn(BTN_WELCOME_START)) {
            flashBtn(BTN_WELCOME_START);
            currentScreen = SCR_MAIN;
            drawMainMenu();
            waitRelease();
        }
        break;

    // ── Main Menu ──────────────────────────────────────────────────
    case SCR_MAIN:
        if (hitBtn(BTN_MAIN[0])) {
            // Scan My Skin
            flashBtn(BTN_MAIN[0]);
            currentScreen = SCR_SCANNING;
            drawScanningScreen();
            Serial.println("REQ:SCAN");
            waitRelease();
        }
        else if (hitBtn(BTN_MAIN[1])) {
            // Pick Product
            flashBtn(BTN_MAIN[1]);
            if (skinType == "") {
                for (uint8_t i = 0; i < NUM_PRODUCTS; i++) recommended[i] = true;
                recommendCount = NUM_PRODUCTS;
            }
            currentScreen = SCR_SELECT;
            drawSelectScreen();
            waitRelease();
        }
        else if (hitBtn(BTN_MAIN[2])) {
            // Take All (Auto)
            flashBtn(BTN_MAIN[2]);
            for (uint8_t i = 0; i < NUM_PRODUCTS; i++) recommended[i] = true;
            dispenseAllRecommended();
            waitRelease();
        }
        break;

    // ── Scanning ───────────────────────────────────────────────────
    case SCR_SCANNING:
        if (hitBtn(BTN_SCAN_CANCEL)) {
            flashBtn(BTN_SCAN_CANCEL);
            currentScreen = SCR_MAIN;
            drawMainMenu();
            waitRelease();
        }
        break;

    // ── Result ─────────────────────────────────────────────────────
    case SCR_RESULT:
        if (hitBtn(BTN_RESULT[0])) {
            flashBtn(BTN_RESULT[0]);
            currentScreen = SCR_RECOMMEND;
            drawRecommendScreen();
            waitRelease();
        }
        else if (hitBtn(BTN_RESULT[1])) {
            flashBtn(BTN_RESULT[1]);
            currentScreen = SCR_MAIN;
            drawMainMenu();
            waitRelease();
        }
        break;

    // ── Recommendations ────────────────────────────────────────────
    case SCR_RECOMMEND:
        if (hitBtn(BTN_REC[0])) {
            flashBtn(BTN_REC[0]);
            dispenseAllRecommended();
            waitRelease();
        }
        else if (hitBtn(BTN_REC[1])) {
            flashBtn(BTN_REC[1]);
            currentScreen = SCR_SELECT;
            drawSelectScreen();
            waitRelease();
        }
        else if (hitBtn(BTN_REC[2])) {
            flashBtn(BTN_REC[2]);
            currentScreen = SCR_RESULT;
            drawResultScreen();
            waitRelease();
        }
        break;

    // ── Select Product ─────────────────────────────────────────────
    case SCR_SELECT: {
        bool handled = false;
        for (uint8_t i = 0; i < NUM_PRODUCTS; i++) {
            int by = SEL_Y0 + i * SEL_SP;
            if (hitRect(SEL_X, by, SEL_W, SEL_H)) {
                flashRect(SEL_X, by, SEL_W, SEL_H);
                dispenseSingleProduct(i);
                waitRelease();
                handled = true;
                break;
            }
        }
        if (!handled && hitBtn(BTN_SEL_BACK)) {
            flashBtn(BTN_SEL_BACK);
            currentScreen = (skinType.length() > 0) ? SCR_RECOMMEND : SCR_MAIN;
            if (currentScreen == SCR_RECOMMEND) drawRecommendScreen();
            else drawMainMenu();
            waitRelease();
        }
        break;
    }

    case SCR_DISPENSING:
        break;
    }
}
