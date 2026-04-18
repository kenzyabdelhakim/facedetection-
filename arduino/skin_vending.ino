/*
 * ============================================================================
 *  SKIN ANALYSIS SMART VENDING SYSTEM
 * ============================================================================
 *  Hardware:
 *    - Arduino Mega 2560
 *    - TFT 3.95" ILI9488 (LCDWIKI shield)
 *    - Resistive touch panel (on same shield)
 *    - L298N motor driver  ×4 channels (2× L298N boards, or 1 board per 2 motors)
 *
 *  Architecture:
 *    Python (laptop/RPi) runs ViT model → sends result over Serial →
 *    Arduino displays result, recommends products, drives motors.
 *
 *  Serial protocol (newline-terminated):
 *    From Python → Arduino:
 *      "RESULT:OILY,ACNE,DARK_SPOTS"       skin type + comma-separated issues
 *      "RESULT:DRY,WRINKLES"               skin type + issues
 *      "RESULT:NORMAL"                      skin type only, no issues
 *      "VEND:1"                             dispense product 1 directly
 *      "VEND:ALL"                           dispense all recommended
 *
 *    From Arduino → Python:
 *      "READY"                              Arduino booted, waiting
 *      "REQ:SCAN"                           user pressed Scan — Python should capture+classify
 *      "DISPENSING:1"                       motor 1 running
 *      "DONE:1"                             motor 1 finished
 *      "DONE:ALL"                           all dispensing complete
 * ============================================================================
 */

#include <LCDWIKI_GUI.h>
#include <LCDWIKI_KBV.h>
#include <LCDWIKI_TOUCH.h>
#include <string.h>

/* ═══════════════════════════════════════════════════════════════════════════
 *  PIN DEFINITIONS
 * ═══════════════════════════════════════════════════════════════════════════
 *
 *  TFT + Touch (directly on Mega shield — directly wired):
 *    CS=40  CD=38  WR=39  RD=43  RST=41
 *    Touch: TCS=53  TCLK=52  TDOUT=50  TDIN=51  TIRQ=44
 *
 *  L298N Motor Driver (4 products = 4 DC motors):
 *    Board 1:  ENA=2  IN1=22  IN2=23   (Motor 1 — Cleanser)
 *              ENB=3  IN3=24  IN4=25   (Motor 2 — Toner)
 *    Board 2:  ENA=4  IN1=26  IN2=27   (Motor 3 — Moisturizer)
 *              ENB=5  IN3=28  IN4=29   (Motor 4 — Sunscreen)
 */

// ── Motor pins (4 products) ────────────────────────────────────────────
struct MotorPins {
    uint8_t en;
    uint8_t in1;
    uint8_t in2;
};

const MotorPins MOTORS[4] = {
    {2, 22, 23},   // Motor 1: Cleanser
    {3, 24, 25},   // Motor 2: Toner
    {4, 26, 27},   // Motor 3: Moisturizer
    {5, 28, 29},   // Motor 4: Sunscreen
};

const char* PRODUCT_NAMES[4] = {
    "Cleanser",
    "Toner",
    "Moisturizer",
    "Sunscreen"
};

const uint8_t  NUM_PRODUCTS       = 4;
const uint8_t  MOTOR_SPEED        = 255;
const uint16_t DISPENSE_TIME_MS   = 2500;  // forward spin per product
const uint16_t MOTOR_PAUSE_MS     = 500;

// ── TFT + Touch ────────────────────────────────────────────────────────
LCDWIKI_KBV lcd(ILI9488, 40, 38, 39, 43, 41);
LCDWIKI_TOUCH touch(53, 52, 50, 51, 44);

// ── Colors ─────────────────────────────────────────────────────────────
#define C_BG         0xEF3C
#define C_PANEL      0xFFFF
#define C_NAVY       0x0010
#define C_WHITE      0xFFFF
#define C_TEXT       0x18C6
#define C_BORDER     0x9CD3
#define C_GREEN      0x3D8E
#define C_RED        0xE0C3
#define C_BLUE       0x041F
#define C_SKY        0x867D
#define C_YELLOW     0xFEA0
#define C_ORANGE     0xFCA0
#define C_PURPLE     0x780F

// ── Touch config ───────────────────────────────────────────────────────
const int16_t  TOUCH_NUDGE_X       = 0;
const int16_t  TOUCH_NUDGE_Y       = 0;
const int      TOUCH_PAD           = 4;    // keep tight to prevent overlap
const uint8_t  TOUCH_SAMPLES       = 12;
const uint8_t  TOUCH_SAMPLE_DELAY  = 3;
const uint8_t  TOUCH_MIN_GOOD      = 4;
const uint8_t  TOUCH_ROTATION      = 3;
bool           TOUCH_SWAP_XY       = false;
bool           TOUCH_INVERT_X      = false;
bool           TOUCH_INVERT_Y      = false;

unsigned long  lastTouchMs         = 0;
const unsigned long DEBOUNCE_MS    = 300;  // ignore taps within this window

// ── Screen state ───────────────────────────────────────────────────────
enum Screen {
    SCR_WELCOME     = 0,
    SCR_MAIN        = 1,
    SCR_SCANNING    = 2,
    SCR_RESULT      = 3,
    SCR_RECOMMEND   = 4,
    SCR_SELECT      = 5,
    SCR_DISPENSING  = 6,
};

Screen currentScreen = SCR_WELCOME;
int16_t px = -1, py = -1;
int16_t LCD_W = 0, LCD_H = 0;
const int HEADER_H = 44;

// ── Skin analysis data ─────────────────────────────────────────────────
String skinType       = "";
String issues[5];
uint8_t issueCount    = 0;
bool recommended[4]   = {false, false, false, false};
uint8_t recommendCount = 0;

// ── Serial buffer ──────────────────────────────────────────────────────
String serialBuf = "";


/* ═══════════════════════════════════════════════════════════════════════════
 *  MOTOR CONTROL
 * ═══════════════════════════════════════════════════════════════════════════ */

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

    Serial.print("DISPENSING:");
    Serial.println(id + 1);

    motorForward(id);
    delay(DISPENSE_TIME_MS);
    motorStop(id);
    delay(MOTOR_PAUSE_MS);

    Serial.print("DONE:");
    Serial.println(id + 1);
}


/* ═══════════════════════════════════════════════════════════════════════════
 *  PRODUCT RECOMMENDATION ENGINE
 *  Maps skin type + issues → which of the 4 products to recommend.
 * ═══════════════════════════════════════════════════════════════════════════ */

void buildRecommendations() {
    recommendCount = 0;
    for (uint8_t i = 0; i < NUM_PRODUCTS; i++) recommended[i] = false;

    // Cleanser: always recommended for oily, or if acne/large_pores detected
    if (skinType == "OILY" || hasIssue("ACNE") || hasIssue("LARGE_PORES")) {
        recommended[0] = true;
    }

    // Toner: recommended for oily or normal, or if redness detected
    if (skinType == "OILY" || skinType == "NORMAL" || hasIssue("REDNESS")) {
        recommended[1] = true;
    }

    // Moisturizer: always recommended for dry, or if wrinkles detected
    if (skinType == "DRY" || hasIssue("WRINKLES") || hasIssue("DARK_SPOTS")) {
        recommended[2] = true;
    }

    // Sunscreen: always recommended (universal protection)
    recommended[3] = true;

    for (uint8_t i = 0; i < NUM_PRODUCTS; i++) {
        if (recommended[i]) recommendCount++;
    }
}

bool hasIssue(const char* name) {
    for (uint8_t i = 0; i < issueCount; i++) {
        if (issues[i] == name) return true;
    }
    return false;
}


/* ═══════════════════════════════════════════════════════════════════════════
 *  SERIAL PROTOCOL PARSER
 *  Expects: "RESULT:OILY,ACNE,DARK_SPOTS\n"
 * ═══════════════════════════════════════════════════════════════════════════ */

void parseSerialCommand(const String& cmd) {
    String c = cmd;
    c.trim();

    Serial.print("RX:");
    Serial.println(c);

    if (c.startsWith("RESULT:")) {
        String payload = c.substring(7);
        parseSkinResult(payload);
        buildRecommendations();
        currentScreen = SCR_RESULT;
        drawResultScreen();

    } else if (c == "TEST") {
        // Quick test: simulate an oily+acne result from Serial Monitor
        parseSkinResult("OILY,ACNE,LARGE_PORES");
        buildRecommendations();
        currentScreen = SCR_RESULT;
        drawResultScreen();
        Serial.println("OK:TEST");

    } else if (c.startsWith("VEND:")) {
        String what = c.substring(5);
        what.trim();
        if (what == "ALL") {
            dispenseAllRecommended();
        } else {
            int id = what.toInt();
            if (id >= 1 && id <= NUM_PRODUCTS) {
                currentScreen = SCR_DISPENSING;
                drawDispensingScreen(id - 1);
                dispenseProduct(id - 1);
                currentScreen = SCR_RECOMMEND;
                drawRecommendScreen();
            }
        }
    }
}

void parseSkinResult(const String& payload) {
    skinType = "";
    issueCount = 0;
    for (uint8_t i = 0; i < 5; i++) issues[i] = "";

    int idx = 0;
    int start = 0;

    for (int i = 0; i <= (int)payload.length(); i++) {
        if (i == (int)payload.length() || payload.charAt(i) == ',') {
            String token = payload.substring(start, i);
            token.trim();
            token.toUpperCase();

            if (idx == 0) {
                skinType = token;      // first token is always the skin type
            } else if (issueCount < 5) {
                issues[issueCount] = token;
                issueCount++;
            }
            idx++;
            start = i + 1;
        }
    }
}


/* ═══════════════════════════════════════════════════════════════════════════
 *  DISPENSING LOGIC
 * ═══════════════════════════════════════════════════════════════════════════ */

void dispenseAllRecommended() {
    for (uint8_t i = 0; i < NUM_PRODUCTS; i++) {
        if (!recommended[i]) continue;
        currentScreen = SCR_DISPENSING;
        drawDispensingScreen(i);
        dispenseProduct(i);
    }
    Serial.println("DONE:ALL");
    currentScreen = SCR_MAIN;
    drawMainMenu();
}

void dispenseSingleProduct(uint8_t id) {
    currentScreen = SCR_DISPENSING;
    drawDispensingScreen(id);
    dispenseProduct(id);
    currentScreen = SCR_SELECT;
    drawSelectScreen();
}


/* ═══════════════════════════════════════════════════════════════════════════
 *  UI DRAWING PRIMITIVES
 * ═══════════════════════════════════════════════════════════════════════════ */

void drawHeader(const char* title) {
    lcd.Set_Draw_color(C_NAVY);
    lcd.Fill_Rectangle(0, 0, LCD_W, HEADER_H);
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

void drawSmallButton(int x, int y, int w, int h,
                     uint16_t fill, uint16_t border, uint16_t textCol, const char* label) {
    drawCard(x, y, x + w, y + h, fill, border);
    int textW = (int)strlen(label) * 6;
    int tx = x + (w - textW) / 2;
    int ty = y + (h - 8) / 2;
    lcd.Set_Text_Size(1);
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


/* ═══════════════════════════════════════════════════════════════════════════
 *  SCREEN DRAWING FUNCTIONS
 * ═══════════════════════════════════════════════════════════════════════════ */

// Screen 0: Welcome
void drawWelcomeScreen() {
    lcd.Fill_Screen(C_BG);
    drawHeader("Skin Smart Vend");
    drawCard(18, 80, 302, 400, C_PANEL, C_BORDER);

    drawLabel(80, 110, "Welcome!", 2, C_NAVY, C_PANEL);
    drawLabel(40, 160, "Smart skincare vending", 1, C_TEXT, C_PANEL);
    drawLabel(40, 180, "system with AI-powered", 1, C_TEXT, C_PANEL);
    drawLabel(40, 200, "skin analysis.", 1, C_TEXT, C_PANEL);

    drawButton(60, 280, 200, 60, C_GREEN, C_BORDER, C_WHITE, "START");
}

// Screen 1: Main menu
// Button Y positions: 100, 190, 280  (90px apart, 55px tall = 35px clear gap)
#define MAIN_BTN_X   35
#define MAIN_BTN_W  250
#define MAIN_BTN_H   55
#define MAIN_BTN1_Y 100
#define MAIN_BTN2_Y 190
#define MAIN_BTN3_Y 280

void drawMainMenu() {
    lcd.Fill_Screen(C_BG);
    drawHeader("Main Menu");
    drawCard(18, 56, 302, 450, C_PANEL, C_BORDER);

    drawLabel(52, 68, "Choose an option:", 2, C_TEXT, C_PANEL);

    drawButton(MAIN_BTN_X, MAIN_BTN1_Y, MAIN_BTN_W, MAIN_BTN_H, C_GREEN,  C_BORDER, C_WHITE, "Scan My Skin");
    drawButton(MAIN_BTN_X, MAIN_BTN2_Y, MAIN_BTN_W, MAIN_BTN_H, C_BLUE,   C_BORDER, C_WHITE, "Pick Product");
    drawButton(MAIN_BTN_X, MAIN_BTN3_Y, MAIN_BTN_W, MAIN_BTN_H, C_YELLOW, C_BORDER, C_NAVY,  "Take All (Auto)");

    drawLabel(38, 360, "Scan: AI skin analysis", 1, C_TEXT, C_PANEL);
    drawLabel(38, 378, "Pick: choose a product", 1, C_TEXT, C_PANEL);
    drawLabel(38, 396, "Auto: dispense default set", 1, C_TEXT, C_PANEL);
}

// Screen 2: Scanning (waiting for Python result)
void drawScanningScreen() {
    lcd.Fill_Screen(C_BG);
    drawHeader("Scanning...");
    drawCard(18, 100, 302, 320, C_PANEL, C_BORDER);

    drawLabel(56, 150, "Analyzing skin...", 2, C_NAVY, C_PANEL);
    drawLabel(40, 200, "Please look at the camera.", 1, C_TEXT, C_PANEL);
    drawLabel(40, 220, "Result will appear shortly.", 1, C_TEXT, C_PANEL);

    drawButton(35, 350, 250, 50, C_RED, C_BORDER, C_WHITE, "CANCEL");
}

// Screen 3: Result display
void drawResultScreen() {
    lcd.Fill_Screen(C_BG);
    drawHeader("Skin Analysis");
    drawCard(18, 56, 302, 310, C_PANEL, C_BORDER);

    // Skin type with color coding
    drawLabel(35, 72, "Skin Type:", 2, C_TEXT, C_PANEL);

    uint16_t typeColor = C_NAVY;
    if (skinType == "OILY")    typeColor = C_BLUE;
    else if (skinType == "DRY") typeColor = C_ORANGE;
    else if (skinType == "NORMAL") typeColor = C_GREEN;

    drawLabel(170, 72, skinType.c_str(), 2, typeColor, C_PANEL);

    // Issues
    drawLabel(35, 110, "Conditions:", 2, C_TEXT, C_PANEL);

    if (issueCount == 0) {
        drawLabel(35, 140, "  None detected", 1, C_GREEN, C_PANEL);
    } else {
        int yPos = 140;
        for (uint8_t i = 0; i < issueCount && i < 5; i++) {
            String nice = "  - " + formatIssue(issues[i]);
            drawLabel(35, yPos, nice.c_str(), 1, C_RED, C_PANEL);
            yPos += 18;
        }
    }

    drawLabel(35, 270, "Products recommended:", 1, C_TEXT, C_PANEL);
    char buf[4];
    sprintf(buf, "%d", recommendCount);
    String recText = String(buf) + " of 4";
    drawLabel(200, 270, recText.c_str(), 1, C_NAVY, C_PANEL);

    drawButton(35, 320, 250, 50, C_GREEN, C_BORDER, C_WHITE, "See Products");
    drawButton(35, 400, 250, 50, C_SKY,   C_BORDER, C_NAVY,  "Back to Menu");
}

// Screen 4: Recommendation screen
void drawRecommendScreen() {
    lcd.Fill_Screen(C_BG);
    drawHeader("Recommended");
    drawCard(18, 56, 302, 370, C_PANEL, C_BORDER);

    drawLabel(35, 72, "For your skin:", 2, C_TEXT, C_PANEL);

    int yPos = 105;
    for (uint8_t i = 0; i < NUM_PRODUCTS; i++) {
        uint16_t col = recommended[i] ? C_GREEN : C_BORDER;
        uint16_t textCol = recommended[i] ? C_NAVY : C_BORDER;
        char num[4];
        sprintf(num, "%d.", i + 1);
        String line = String(num) + " " + PRODUCT_NAMES[i];
        if (recommended[i]) line += "  *";
        drawLabel(45, yPos, line.c_str(), 2, textCol, C_PANEL);
        yPos += 32;
    }

    drawLabel(45, yPos + 10, "* = recommended for you", 1, C_TEXT, C_PANEL);

    drawButton(35, 385, 120, 48, C_GREEN, C_BORDER, C_WHITE, "TAKE ALL");
    drawButton(165, 385, 120, 48, C_BLUE,  C_BORDER, C_WHITE, "SELECT");
    drawSmallButton(110, 445, 100, 28, C_RED, C_BORDER, C_WHITE, "BACK");
}

// Screen 5: Select individual product
// 4 buttons: y = 100, 175, 250, 325 (75px apart, 50px tall = 25px gap)
#define SEL_BTN_X   35
#define SEL_BTN_W  250
#define SEL_BTN_H   50
#define SEL_BTN_SPACING 75
#define SEL_BTN_Y0 100

void drawSelectScreen() {
    lcd.Fill_Screen(C_BG);
    drawHeader("Select Product");
    drawCard(18, 56, 302, 460, C_PANEL, C_BORDER);

    drawLabel(40, 68, "Tap to dispense:", 2, C_TEXT, C_PANEL);

    uint16_t btnColors[4] = {C_GREEN, C_BLUE, C_PURPLE, C_YELLOW};
    uint16_t txtColors[4] = {C_WHITE, C_WHITE, C_WHITE, C_NAVY};

    for (uint8_t i = 0; i < NUM_PRODUCTS; i++) {
        int y = SEL_BTN_Y0 + i * SEL_BTN_SPACING;
        String label = String(PRODUCT_NAMES[i]);
        if (recommended[i]) label += " *";
        drawButton(SEL_BTN_X, y, SEL_BTN_W, SEL_BTN_H, btnColors[i], C_BORDER, txtColors[i], label.c_str());
    }

    drawSmallButton(110, 420, 100, 30, C_RED, C_BORDER, C_WHITE, "BACK");
}

// Screen 6: Dispensing animation
void drawDispensingScreen(uint8_t productId) {
    lcd.Fill_Screen(C_BG);
    drawHeader("Dispensing...");
    drawCard(18, 100, 302, 320, C_PANEL, C_BORDER);

    drawLabel(60, 150, "Dispensing:", 2, C_NAVY, C_PANEL);
    drawLabel(60, 185, PRODUCT_NAMES[productId], 2, C_GREEN, C_PANEL);

    // Simple progress bar background
    lcd.Set_Draw_color(C_BORDER);
    lcd.Fill_Rectangle(50, 240, 270, 258);
    lcd.Set_Draw_color(C_GREEN);
    lcd.Fill_Rectangle(50, 240, 160, 258);

    drawLabel(70, 280, "Please wait...", 1, C_TEXT, C_PANEL);
}


/* ═══════════════════════════════════════════════════════════════════════════
 *  HELPERS
 * ═══════════════════════════════════════════════════════════════════════════ */

String formatIssue(const String& raw) {
    // "DARK_SPOTS" → "Dark Spots"
    String out = raw;
    out.toLowerCase();
    for (int i = 0; i < (int)out.length(); i++) {
        if (out.charAt(i) == '_') out.setCharAt(i, ' ');
    }
    if (out.length() > 0) {
        out.setCharAt(0, toupper(out.charAt(0)));
        for (int i = 1; i < (int)out.length(); i++) {
            if (out.charAt(i - 1) == ' ')
                out.setCharAt(i, toupper(out.charAt(i)));
        }
    }
    return out;
}

// ── Touch reading ──────────────────────────────────────────────────────

bool readTouch() {
    // Debounce: ignore taps too close together
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

    int16_t tx = (int16_t)(ax / good) + TOUCH_NUDGE_X;
    int16_t ty = (int16_t)(ay / good) + TOUCH_NUDGE_Y;

    if (TOUCH_SWAP_XY) { int16_t t = tx; tx = ty; ty = t; }
    if (TOUCH_INVERT_X) tx = LCD_W - tx;
    if (TOUCH_INVERT_Y) ty = LCD_H - ty;

    px = constrain(tx, 0, LCD_W);
    py = constrain(ty, 0, LCD_H);
    lastTouchMs = millis();
    return true;
}

bool hit(int x1, int y1, int x2, int y2) {
    return (px >= x1 - TOUCH_PAD && px <= x2 + TOUCH_PAD &&
            py >= y1 - TOUCH_PAD && py <= y2 + TOUCH_PAD);
}

void waitRelease() {
    delay(150);
    unsigned long t0 = millis();
    do {
        touch.TP_Scan(0);
        if (millis() - t0 > 2000) break; // safety timeout
    } while (touch.TP_Get_State() & TP_PRES_DOWN);
    delay(100);
    lastTouchMs = millis(); // reset debounce after release
}

void flash(int x1, int y1, int x2, int y2) {
    lcd.Set_Draw_color(C_WHITE);
    lcd.Draw_Round_Rectangle(x1, y1, x2, y2, 8);
    delay(70);
}


/* ═══════════════════════════════════════════════════════════════════════════
 *  SETUP
 * ═══════════════════════════════════════════════════════════════════════════ */

void setup() {
    Serial.begin(9600);
    motorSetup();

    lcd.Init_LCD();
    lcd.Set_Rotation(0);  // portrait
    LCD_W = lcd.Get_Display_Width() - 1;
    LCD_H = lcd.Get_Display_Height() - 1;

    touch.TP_Set_Rotation(TOUCH_ROTATION);
    touch.TP_Init(lcd.Get_Rotation(), lcd.Get_Display_Width(), lcd.Get_Display_Height());

    drawWelcomeScreen();
    Serial.println("READY");
}


/* ═══════════════════════════════════════════════════════════════════════════
 *  MAIN LOOP  (non-blocking)
 * ═══════════════════════════════════════════════════════════════════════════ */

void loop() {
    // ── Serial input (non-blocking) ────────────────────────────────
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

    // ── Touch input ────────────────────────────────────────────────
    px = -1; py = -1;
    if (!readTouch()) return;

    switch (currentScreen) {

    // ── Welcome ────────────────────────────────────────────────────
    case SCR_WELCOME:
        if (hit(60, 280, 260, 340)) {
            flash(60, 280, 260, 340);
            currentScreen = SCR_MAIN;
            drawMainMenu();
            waitRelease();
        }
        break;

    // ── Main Menu ──────────────────────────────────────────────────
    case SCR_MAIN: {
        int bx2 = MAIN_BTN_X + MAIN_BTN_W;
        // Scan My Skin
        if (hit(MAIN_BTN_X, MAIN_BTN1_Y, bx2, MAIN_BTN1_Y + MAIN_BTN_H)) {
            flash(MAIN_BTN_X, MAIN_BTN1_Y, bx2, MAIN_BTN1_Y + MAIN_BTN_H);
            currentScreen = SCR_SCANNING;
            drawScanningScreen();
            Serial.println("REQ:SCAN");
            waitRelease();
        }
        // Pick Product
        else if (hit(MAIN_BTN_X, MAIN_BTN2_Y, bx2, MAIN_BTN2_Y + MAIN_BTN_H)) {
            flash(MAIN_BTN_X, MAIN_BTN2_Y, bx2, MAIN_BTN2_Y + MAIN_BTN_H);
            currentScreen = SCR_SELECT;
            if (skinType == "") {
                for (uint8_t i = 0; i < NUM_PRODUCTS; i++) recommended[i] = true;
                recommendCount = NUM_PRODUCTS;
            }
            drawSelectScreen();
            waitRelease();
        }
        // Take All (Auto)
        else if (hit(MAIN_BTN_X, MAIN_BTN3_Y, bx2, MAIN_BTN3_Y + MAIN_BTN_H)) {
            flash(MAIN_BTN_X, MAIN_BTN3_Y, bx2, MAIN_BTN3_Y + MAIN_BTN_H);
            for (uint8_t i = 0; i < NUM_PRODUCTS; i++) recommended[i] = true;
            dispenseAllRecommended();
            waitRelease();
        }
        break;
    }

    // ── Scanning (waiting) ─────────────────────────────────────────
    case SCR_SCANNING:
        // Cancel button
        if (hit(35, 350, 285, 400)) {
            flash(35, 350, 285, 400);
            currentScreen = SCR_MAIN;
            drawMainMenu();
            waitRelease();
        }
        break;

    // ── Result ─────────────────────────────────────────────────────
    case SCR_RESULT:
        // See Products
        if (hit(35, 320, 285, 370)) {
            flash(35, 320, 285, 370);
            currentScreen = SCR_RECOMMEND;
            drawRecommendScreen();
            waitRelease();
        }
        // Back to Menu
        else if (hit(35, 400, 285, 450)) {
            flash(35, 400, 285, 450);
            currentScreen = SCR_MAIN;
            drawMainMenu();
            waitRelease();
        }
        break;

    // ── Recommendations ────────────────────────────────────────────
    case SCR_RECOMMEND:
        // Take All
        if (hit(35, 385, 155, 433)) {
            flash(35, 385, 155, 433);
            dispenseAllRecommended();
            waitRelease();
        }
        // Select individual
        else if (hit(165, 385, 285, 433)) {
            flash(165, 385, 285, 433);
            currentScreen = SCR_SELECT;
            drawSelectScreen();
            waitRelease();
        }
        // Back
        else if (hit(110, 445, 210, 473)) {
            flash(110, 445, 210, 473);
            currentScreen = SCR_RESULT;
            drawResultScreen();
            waitRelease();
        }
        break;

    // ── Select individual product ──────────────────────────────────
    case SCR_SELECT: {
        bool handled = false;
        for (uint8_t i = 0; i < NUM_PRODUCTS; i++) {
            int y1 = SEL_BTN_Y0 + i * SEL_BTN_SPACING;
            int y2 = y1 + SEL_BTN_H;
            if (hit(SEL_BTN_X, y1, SEL_BTN_X + SEL_BTN_W, y2)) {
                flash(SEL_BTN_X, y1, SEL_BTN_X + SEL_BTN_W, y2);
                dispenseSingleProduct(i);
                waitRelease();
                handled = true;
                break;
            }
        }
        // Back
        if (!handled && hit(110, 420, 210, 450)) {
            flash(110, 420, 210, 450);
            currentScreen = (skinType.length() > 0) ? SCR_RECOMMEND : SCR_MAIN;
            if (currentScreen == SCR_RECOMMEND) drawRecommendScreen();
            else drawMainMenu();
            waitRelease();
        }
        break;
    }

    case SCR_DISPENSING:
        // No touch during dispensing — motor is running
        break;
    }
}
