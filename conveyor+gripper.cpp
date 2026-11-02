#include <Arduino.h>
#include <ESP32Servo.h>

// =========================
// [1] Pin Conveyor (DM556)
// =========================
#define STEP_PIN      18
#define DIR_PIN       19
#define ENABLE_PIN    21      // DM556 ไม่ได้ต่อขา ENA -> ไม่มีผล (ปล่อยไว้ได้)
#define LIMIT_HOME    23      // Home อยู่ฝั่งขวา
#define LIMIT_END     22      // End อยู่ฝั่งซ้าย
// ทิศมอเตอร์ตามที่ทดสอบจริงกับ DM556: เดินหน้า = ออกจาก Home ไปทาง End
#define DIR_FORWARD   LOW
#define DIR_HOME      HIGH

// =========================
// [2] Pin Angle Stepper (TB6600)
// =========================
#define STEP_PIN_ANGLE 25
#define DIR_PIN_ANGLE  26
#define ENA_PIN_ANGLE  27

// =========================
// [3] Pin Servo
// =========================
#define SERVO1_PIN 13
#define SERVO2_PIN 12
#define SERVO3_PIN 14
#define SERVO4_PIN 32
#define SERVO5_PIN 33
#define SERVO6_PIN 4

Servo servo1, servo2, servo3, servo4, servo5, servo6;

// =========================
// Conveyor config: DM556 800 ppr, GT2 20 ฟัน (pitch 2mm)  -> 1 step = 0.05 mm
// =========================
const float STEPS_PER_REV = 800.0;
const float MM_PER_REV    = 40.0;
const float STEPS_PER_MM  = STEPS_PER_REV / MM_PER_REV;   // = 20
const int   STEP_DELAY    = 400;   // ใช้ตอน Home (µs) ช้าลงเพื่อให้จุด Home ซ้ำได้แม่น
long        currentPosition = 0;   // ตำแหน่งปัจจุบันเป็น step นับจาก Home (open-loop)
float       lastTargetMM    = 0;   // เป้าหมายล่าสุด (ใช้กับคำสั่งแบบ relative)

// ---- ความเร็ว (ปรับด้วยคำสั่ง F<mm/s>) ----
float       speedMMS          = 40.0;   // ความเร็วเดินทาง (mm/s) ค่าเริ่มต้น
const float MIN_SPEED_MMS     = 5.0;    // ความเร็วตอนออกตัว/หยุด
const float MAX_SPEED_MMS     = 100.0;  // เพดานความเร็ว (ถ้า step หลุดให้ลดลง)
const float ACCEL_MMS2        = 300.0;  // ความเร่ง/หน่วง (mm/s^2)

// ---- การเข้าหาเป้าหมาย ----
const float APPROACH_MM       = 2.0;    // ช่วงสุดท้ายก่อนถึงเป้า เข้าหาจาก "ฝั่ง Home" ทิศเดียวเสมอ
const float APPROACH_SPEED_MMS= 10.0;   // ความเร็วช่วงเข้าหาสุดท้าย (ช้า = นิ่งและแม่น)
const float HOME_MARGIN_MM    = 1.0;    // เป้าใกล้ Home กว่านี้ -> Home ใหม่แล้วเดินหน้าเข้าหา
const int   SETTLE_MS         = 50;     // รอให้สายพานนิ่งหลังถึงเป้า (ก่อนถ่ายภาพ)

// ---- Warm-up ทุกครั้งที่ออกจาก Home ----
const float WARMUP_MM         = 30.0;   // ระยะที่เลื่อนออกจาก Home ตอน warm-up

// =========================
// Angle config
// =========================
const int PULSES_PER_45DEG = 200;
const int PULSES_PER_15DEG  = 67;
const int PULSES_PER_90DEG = PULSES_PER_45DEG * 2;
const int PULSE_PER_REV    = PULSES_PER_45DEG * 8;
const int pulseDelay       = 150;
long      currentAnglePosition = 0;

// =========================
// [5] Angle Home sensor (F249) - คำสั่ง O
// =========================
// GPIO34 เป็นขา input-only ไม่มี pull-up/pull-down ภายใน จ่ายไฟโมดูลที่ 3.3V
#define HOME_SENSOR_PIN      34
#define HOME_SENSOR_ACTIVE   HIGH   // ระดับสัญญาณตอนเซ็นเซอร์ "เจอ home" (ตามสเก็ตช์ทดสอบ HIGH = มีวัตถุ) ถ้าตรงข้ามเปลี่ยนเป็น LOW
#define ANGLE_HOME_DIR_LEVEL LOW    // ทิศหมุนตอนหา home = ทิศเดียวกับคำสั่ง V
const int ANGLE_HOME_DELAY_US = 300;  // ความเร็วตอนหา home: ครึ่งคาบพัลส์ (µs) ยิ่งมากยิ่งช้า ช้า = ตำแหน่งหยุดซ้ำได้แม่นกว่า
const int ANGLE_HOME_DEBOUNCE = 3;    // ต้องเห็นสัญญาณเปลี่ยนต่อเนื่องกี่พัลส์ถึงยอมรับ

// Servo Wave Mode
bool waveMode = false;

// =========================
// CONVEYOR Functions (open-loop, DM556)
// =========================

// วิ่ง n step ในทิศที่กำหนด ด้วย profile เร่ง-คงที่-หน่วง (trapezoid)
// vMax = ความเร็วสูงสุดของช่วงนี้ (mm/s)
bool runSteps(long n, bool forward, float vMax) {
    if (n <= 0) return true;

    digitalWrite(DIR_PIN, forward ? DIR_FORWARD : DIR_HOME);
    delayMicroseconds(100);                        // DM556 ต้องการ DIR setup ≥ 5µs

    vMax = constrain(vMax, MIN_SPEED_MMS, MAX_SPEED_MMS);

    for (long i = 0; i < n; i++) {
        if (forward  && digitalRead(LIMIT_END)  == LOW) { Serial.println("ชน End limit");  return false; }
        if (!forward && digitalRead(LIMIT_HOME) == LOW) { Serial.println("ชน Home limit"); return false; }

        float dAcc = i / STEPS_PER_MM;             // ระยะที่เดินมาแล้ว
        float dDec = (n - i) / STEPS_PER_MM;       // ระยะที่เหลือ
        float vUp  = sqrtf(MIN_SPEED_MMS * MIN_SPEED_MMS + 2.0f * ACCEL_MMS2 * dAcc);
        float vDn  = sqrtf(MIN_SPEED_MMS * MIN_SPEED_MMS + 2.0f * ACCEL_MMS2 * dDec);
        float v    = min(vMax, min(vUp, vDn));

        int half = (int)(1e6f / (2.0f * v * STEPS_PER_MM));
        half = constrain(half, 100, 20000);

        digitalWrite(STEP_PIN, HIGH);
        delayMicroseconds(half);
        digitalWrite(STEP_PIN, LOW);
        delayMicroseconds(half);

        currentPosition += forward ? 1 : -1;
    }
    return true;
}

void stepPulse(int delay_us) {
    if (digitalRead(LIMIT_HOME) == LOW && digitalRead(DIR_PIN) == DIR_HOME)    return;
    if (digitalRead(LIMIT_END)  == LOW && digitalRead(DIR_PIN) == DIR_FORWARD) return;

    digitalWrite(STEP_PIN, HIGH);
    delayMicroseconds(delay_us);
    digitalWrite(STEP_PIN, LOW);
    delayMicroseconds(delay_us);
}

void goHome() {
    Serial.println("กำลัง Home...");
    digitalWrite(ENABLE_PIN, LOW);
    digitalWrite(DIR_PIN, DIR_HOME);
    delay(10);

    // เริ่มจากปลายราง (End ถูกกดอยู่) ต้องถอยออกได้: หยุดเฉพาะเมื่อ End "เพิ่งถูกกด" ระหว่างทางกลับ Home
    bool endWasPressed = digitalRead(LIMIT_END) == LOW;
    if (endWasPressed) Serial.println("เริ่มจาก End -> ถอยกลับ Home");

    while (digitalRead(LIMIT_HOME) == HIGH) {
        bool endPressed = digitalRead(LIMIT_END) == LOW;
        if (!endPressed) endWasPressed = false;
        if (endPressed && !endWasPressed) {
            Serial.println("ชน End ตอน Home ต่อสายผิดหรือเปล่า?");
            digitalWrite(ENABLE_PIN, HIGH);
            return;
        }
        stepPulse(STEP_DELAY);
        currentPosition--;
    }

    currentPosition = 0;
    digitalWrite(ENABLE_PIN, HIGH);
    delay(100);          // รอให้นิ่งก่อนตั้งศูนย์
    lastTargetMM = 0;
    Serial.println("Home เสร็จ ตำแหน่ง = 0mm");
}

// Warm-up: เลื่อนออกจาก Home ระยะหนึ่ง -> กลับ Home (ให้มอเตอร์/สายพานเข้าที่ก่อนวัดจริง)
void warmUp() {
    Serial.print("Warm-up: เลื่อนออก "); Serial.print(WARMUP_MM, 1); Serial.println(" mm แล้วกลับ Home");
    runSteps((long)lroundf(WARMUP_MM * STEPS_PER_MM), true, speedMMS);
    delay(200);
    goHome();
}

// เคลื่อนไปตำแหน่งสัมบูรณ์ (mm จาก Home) แบบ open-loop
// ช่วงสุดท้ายเข้าหาเป้าจากฝั่ง Home ด้วยความเร็วต่ำเสมอ
bool moveToMM(float targetMM) {
    if (targetMM < 0) { Serial.println("เป้าหมายติดลบ ยกเลิก"); return false; }

    long targetSteps = (long)lroundf(targetMM * STEPS_PER_MM);
    long apSteps     = (long)lroundf(APPROACH_MM * STEPS_PER_MM);
    float approachV  = min(speedMMS, APPROACH_SPEED_MMS);
    bool ok = true;

    // 1) ถ้าเป้าอยู่ถอยหลัง (ใกล้ Home กว่า): ถอยไปเลยเป้าเล็กน้อย แล้วค่อยเดินหน้าเข้าหา
    if (targetSteps < currentPosition) {
        if (targetMM < HOME_MARGIN_MM) {
            goHome();                              // เป้าใกล้ Home มาก: Home ใหม่ แล้วเดินหน้าเข้าหา
        } else {
            long pre = max(targetSteps - apSteps, (long)lroundf(HOME_MARGIN_MM * STEPS_PER_MM));
            ok = runSteps(currentPosition - pre, false, speedMMS);
        }
    }

    // 2) ออกจาก Home ทุกครั้ง -> warm-up ก่อน (เลื่อนออก -> กลับ Home) แล้วค่อยไปเป้าจริง
    if (ok && currentPosition <= 0 && targetSteps > 0) warmUp();

    // 3) เดินหน้าเข้าหาเป้า: ช่วงไกลใช้ความเร็วที่เลือก ช่วงสุดท้ายช้าลง
    if (ok) {
        long d = targetSteps - currentPosition;
        if (d > apSteps) {
            ok = runSteps(d - apSteps, true, speedMMS);
            if (ok) ok = runSteps(apSteps, true, approachV);
        } else {
            ok = runSteps(d, true, approachV);
        }
    }

    delay(SETTLE_MS);
    lastTargetMM = currentPosition / STEPS_PER_MM;

    if (ok) {
        Serial.print("✔ ถึงเป้าหมาย "); Serial.print(lastTargetMM, 3); Serial.println(" mm");
    } else {
        Serial.println("⚠ เคลื่อนที่ไม่สำเร็จ");
    }
    return ok;
}

void printStatus() {
    Serial.print("Pos: "); Serial.print(currentPosition / STEPS_PER_MM, 3);
    Serial.print("mm | Speed: "); Serial.print(speedMMS, 1);
    Serial.print("mm/s | Home: "); Serial.print(digitalRead(LIMIT_HOME) == LOW ? "ON" : "OFF");
    Serial.print(" | End: "); Serial.println(digitalRead(LIMIT_END) == LOW ? "ON" : "OFF");
}

void setSpeed(float v) {
    speedMMS = constrain(v, MIN_SPEED_MMS, MAX_SPEED_MMS);
    Serial.print("ความเร็ว = "); Serial.print(speedMMS, 1); Serial.println(" mm/s");
}

void handleConveyorMove(float mm) {
    Serial.print("สายพาน: "); Serial.print(mm);
    Serial.print("mm = "); Serial.print((long)round(abs(mm * STEPS_PER_MM)));
    Serial.println(" steps");

    if (mm != 0) {
        float target = lastTargetMM + mm;
        Serial.print("▶ -> "); Serial.print(target, 2); Serial.println("mm");
        moveToMM(target);
    } else {
        Serial.println("ค่าเป็น 0");
    }
    printStatus();
}

// =========================
// ANGLE STEPPER Functions
// =========================
void moveAngle(long pulses, bool direction) {
    digitalWrite(ENA_PIN_ANGLE, LOW);
    digitalWrite(DIR_PIN_ANGLE, direction ? HIGH : LOW);
    delayMicroseconds(5);

    for (long i = 0; i < pulses; i++) {
        digitalWrite(STEP_PIN_ANGLE, HIGH);
        delayMicroseconds(pulseDelay);
        digitalWrite(STEP_PIN_ANGLE, LOW);
        delayMicroseconds(pulseDelay);
    }
    currentAnglePosition += direction ? pulses : -pulses;
}

void angleZ() {
    long target = PULSES_PER_90DEG / 2;
    if (currentAnglePosition < target) {
        moveAngle(target - currentAnglePosition, true);
    } else if (currentAnglePosition > target) {
        moveAngle(currentAnglePosition - target, false);
    }
    currentAnglePosition = target;
    Serial.println("Position: 90 องศา");
}

void angleX() {
    if (currentAnglePosition > 0) {
        moveAngle(currentAnglePosition, false);
    } else if (currentAnglePosition < 0) {
        moveAngle(-currentAnglePosition, true);
    }
    currentAnglePosition = 0;
    Serial.println("Position: 0 องศา");
}

void printAngleStatus() {
    float deg = (currentAnglePosition * 45.0) / PULSES_PER_45DEG;
    Serial.print("มุมปัจจุบัน: "); Serial.print(deg); Serial.println(" องศา");
}

// หมุน Angle ไปทางเดียวกับ V จนเซ็นเซอร์ F249 เจอ home แล้วตั้งตำแหน่ง = 0
// หยุดที่ "ขอบขาเข้า" ขอบเดียวเสมอ: ถ้าเริ่มต้นอยู่ในแผ่นบัง/แม่เหล็กอยู่แล้ว จะหมุนออกก่อน แล้วค่อยวนมาเจอใหม่
bool angleHomeSearch() {
    bool startActive = (digitalRead(HOME_SENSOR_PIN) == HOME_SENSOR_ACTIVE);
    Serial.print("Angle Home: เซ็นเซอร์ตอนเริ่ม = ");
    Serial.println(startActive ? "ACTIVE (อยู่ในจุด home แล้ว จะหมุนออกแล้ววนกลับมาเจอ)" : "INACTIVE");

    digitalWrite(ENA_PIN_ANGLE, LOW);
    digitalWrite(DIR_PIN_ANGLE, ANGLE_HOME_DIR_LEVEL);
    delayMicroseconds(5);

    const long maxPulses = (long)PULSE_PER_REV * 2;    // ไม่เกิน 2 รอบ กันหมุนไม่รู้จบ
    bool leaving = startActive;                        // true = ยังต้องหมุนออกจากจุด home ก่อน
    int  streak  = 0;                                  // นับพัลส์ต่อเนื่องที่สัญญาณเปลี่ยน (debounce)
    long done    = 0;
    bool found   = false;

    while (done < maxPulses) {
        digitalWrite(STEP_PIN_ANGLE, HIGH);
        delayMicroseconds(ANGLE_HOME_DELAY_US);
        digitalWrite(STEP_PIN_ANGLE, LOW);
        delayMicroseconds(ANGLE_HOME_DELAY_US);
        done++;

        bool active = (digitalRead(HOME_SENSOR_PIN) == HOME_SENSOR_ACTIVE);
        if (active != leaving) {                       // leaving: รอ INACTIVE / ไม่งั้น: รอ ACTIVE
            if (++streak >= ANGLE_HOME_DEBOUNCE) {
                if (leaving) { leaving = false; streak = 0; }   // ออกจากจุด home แล้ว เริ่มหาขอบขาเข้า
                else         { found = true; break; }
            }
        } else {
            streak = 0;
        }
    }

    if (found) {
        currentAnglePosition = 0;
        Serial.print("Position: 0 องศา (Home sensor, หมุน "); Serial.print(done); Serial.println(" พัลส์)");
        return true;
    }

    currentAnglePosition -= done;                      // ยังเก็บตำแหน่งสะสมให้ตรงกับที่หมุนไปจริง
    if (leaving) Serial.println("⚠ Angle Home: เซ็นเซอร์ ACTIVE ตลอด 2 รอบ ตรวจสาย/ระดับ HOME_SENSOR_ACTIVE");
    else         Serial.println("⚠ Angle Home: ไม่เจอเซ็นเซอร์ภายใน 2 รอบ ตรวจสาย/แผ่นบัง/ไฟ 3.3V");
    return false;
}

// =========================
// SERVO Functions
// =========================
void setServo(Servo &s, int openVal, int closeVal, bool close) {
    s.write(close ? closeVal : openVal);
}

void applyMask(int mask) {
    setServo(servo1, 86+20, 62,  (mask >> 5) & 1);
    setServo(servo2, 101+20, 70, (mask >> 4) & 1);
    setServo(servo3, 99+20, 72,  (mask >> 3) & 1);
    setServo(servo4, 90-20, 114, (mask >> 2) & 1);
    setServo(servo5, 85-20, 110, (mask >> 1) & 1);
    setServo(servo6, 88-20, 114, (mask >> 0) & 1);

    Serial.print("Servo Mask: "); Serial.print(mask); Serial.print(" → Binary: ");
    for (int i = 5; i >= 0; i--) Serial.print((mask >> i) & 1);
    Serial.println();
}

void openAll()  { applyMask(0b000000); }
void closeAll() { applyMask(0b111111); }

void waveServo() {
    closeAll();
    delay(300);
    servo1.write(64); delay(500); servo1.write(70);
    servo2.write(85); delay(500); servo2.write(73);
    servo3.write(99); delay(500); servo3.write(74);
    servo4.write(90); delay(500); servo4.write(112);
    servo5.write(85); delay(500); servo5.write(109);
    servo6.write(90); delay(500); servo6.write(112);
}

// =========================
// Setup
// =========================
void setup() {
    Serial.begin(115200);

    // Setup Conveyor
    pinMode(STEP_PIN,   OUTPUT);
    pinMode(DIR_PIN,    OUTPUT);
    pinMode(ENABLE_PIN, OUTPUT);
    pinMode(LIMIT_HOME, INPUT_PULLUP);
    pinMode(LIMIT_END,  INPUT_PULLUP);
    digitalWrite(ENABLE_PIN, HIGH);
    digitalWrite(DIR_PIN,    DIR_HOME);
    digitalWrite(STEP_PIN,   LOW);

    // Setup Angle
    pinMode(STEP_PIN_ANGLE, OUTPUT);
    pinMode(DIR_PIN_ANGLE,  OUTPUT);
    pinMode(ENA_PIN_ANGLE,  OUTPUT);
    digitalWrite(ENA_PIN_ANGLE,  LOW);
    digitalWrite(DIR_PIN_ANGLE,  LOW);
    digitalWrite(STEP_PIN_ANGLE, LOW);
    pinMode(HOME_SENSOR_PIN, INPUT);        // F249 (GPIO34 ไม่มี pull-up ภายใน)

    // Setup Servos
    ESP32PWM::allocateTimer(0);
    ESP32PWM::allocateTimer(1);
    ESP32PWM::allocateTimer(2);
    ESP32PWM::allocateTimer(3);

    servo1.setPeriodHertz(50); servo1.attach(SERVO1_PIN, 500, 2400);
    servo2.setPeriodHertz(50); servo2.attach(SERVO2_PIN, 500, 2400);
    servo3.setPeriodHertz(50); servo3.attach(SERVO3_PIN, 500, 2400);
    servo4.setPeriodHertz(50); servo4.attach(SERVO4_PIN, 500, 2400);
    servo5.setPeriodHertz(50); servo5.attach(SERVO5_PIN, 500, 2400);
    servo6.setPeriodHertz(50); servo6.attach(SERVO6_PIN, 500, 2400);

    openAll();

    Serial.println("==============================================");
    Serial.println("ESP32 Master Control - Conveyor(DM556), Angle, Servos");
    Serial.println("==============================================");
    Serial.println("[คำสั่งสายพาน (Open-Loop)]");
    Serial.println("  H = Home, S = แสดงสถานะ");
    Serial.println("  M<ค่า> = เคลื่อนที่แบบ relative (mm) เช่น M50, M-20");
    Serial.println("  G<ค่า> = ไปตำแหน่งสัมบูรณ์ (mm) เช่น G120.5");
    Serial.println("  F<ค่า> = ตั้งความเร็ว (mm/s) เช่น F40 (F เฉยๆ = ดูความเร็วปัจจุบัน)");
    Serial.println("  T = Home + Warm-up ทันที (ปกติทำเองอัตโนมัติทุกครั้งที่ออกจาก Home)");
    Serial.println("  ตัวเลขทศนิยม/ติดลบ/เกิน 63 = relative เช่นกัน");
    Serial.println("[คำสั่ง Angle]");
    Serial.println("  Z = ลง 90 องศา, X = ขึ้น 0 องศา");
    Serial.println("  C = หมุน +45 องศา, A = แสดงมุมปัจจุบัน");
    Serial.println("  V = หมุน -45 องศา (ทิศตรงข้าม C ระยะเท่ากัน)");
    Serial.println("  O = หมุนทิศเดียวกับ V จนเซ็นเซอร์ F249 เจอ home แล้วตั้งมุม = 0");
    Serial.println("[คำสั่ง Servo]");
    Serial.println("  Q = ปิดทุกตัว, W = เปิดทุกตัว, R = Wave Mode");
    Serial.println("  ตัวเลข 0-63 = Servo Bitmask");
    Serial.println("  B<ค่า> = สั่ง Servo เจาะจง (เช่น B63)");
    Serial.println("==============================================");

    delay(100);
    goHome();
}

// =========================
// Loop
// =========================
String inputBuffer = "";

void loop() {
    static bool lastHome = HIGH;
    static bool lastEnd  = HIGH;

    bool nowHome = digitalRead(LIMIT_HOME);
    bool nowEnd  = digitalRead(LIMIT_END);

    // เช็ค Limit Switch
    if (nowHome != lastHome) {
        Serial.print("Home Switch: ");
        Serial.println(nowHome == LOW ? "ON ← ถึงจุด Home" : "OFF");
        if (nowHome == LOW) { currentPosition = 0; lastTargetMM = 0; }
        lastHome = nowHome;
    }

    if (nowEnd != lastEnd) {
        Serial.print("End Switch: ");
        Serial.println(nowEnd == LOW ? "ON ← ถึงจุด End" : "OFF");
        lastEnd = nowEnd;
    }

    // รับคำสั่งจาก Serial
    while (Serial.available()) {
        char c = Serial.read();

        if (c == '\n' || c == '\r') {
            inputBuffer.trim();

            if (inputBuffer.length() > 0) {
                String cmd = inputBuffer;
                cmd.toUpperCase();

                // 1. ตรวจสอบคำสั่งตัวอักษรเดี่ยว
                if (cmd == "H") goHome();
                else if (cmd == "S") printStatus();
                else if (cmd == "T") { goHome(); warmUp(); printStatus(); }
                else if (cmd == "F") { Serial.print("ความเร็ว = "); Serial.print(speedMMS, 1); Serial.println(" mm/s"); }
                else if (cmd == "Z") angleZ();
                else if (cmd == "X") angleX();
                else if (cmd == "C") {
                    moveAngle(PULSES_PER_45DEG, true);
                    Serial.print("หมุน +15 องศา → "); printAngleStatus();
                }
                else if (cmd == "V") {
                    moveAngle(PULSES_PER_15DEG, false);
                    Serial.print("หมุน -15 องศา → "); printAngleStatus();
                }
                else if (cmd == "O") angleHomeSearch();
                else if (cmd == "A") printAngleStatus();
                else if (cmd == "Q") { waveMode = false; closeAll(); Serial.println("ALL: CLOSE"); }
                else if (cmd == "W") { waveMode = false; openAll(); Serial.println("ALL: OPEN"); }
                else if (cmd == "R") {
                    waveMode = !waveMode;
                    Serial.println(waveMode ? "WAVE: START" : "WAVE: STOP");
                    if (!waveMode) closeAll();
                }

                // 2. ตรวจสอบคำสั่งที่มีการระบุ Prefix ชัดเจน
                else if (cmd.startsWith("F")) {
                    setSpeed(inputBuffer.substring(1).toFloat());
                }
                else if (cmd.startsWith("G")) {
                    moveToMM(inputBuffer.substring(1).toFloat());
                    printStatus();
                }
                else if (cmd.startsWith("M")) {
                    handleConveyorMove(inputBuffer.substring(1).toFloat());
                }
                else if (cmd.startsWith("B")) {
                    waveMode = false;
                    applyMask(inputBuffer.substring(1).toInt());
                }

                // 3. ระบบแยกแยะตัวเลขอัตโนมัติ (Smart Parser)
                else {
                    // ถ้ามีจุดทศนิยม, ค่าติดลบ, หรือมีค่าอยู่นอกระยะ 0-63 จะมองเป็นสายพานทันที
                    if (inputBuffer.indexOf('.') >= 0 || inputBuffer.toInt() < 0 || inputBuffer.toFloat() > 63 || inputBuffer.toFloat() < -63) {
                        handleConveyorMove(inputBuffer.toFloat());
                    } else {
                        // ถ้าเป็นเลขโดดๆ ระหว่าง 0-63 จะมองเป็น Servo Mask (พฤติกรรมเดิม)
                        int mask = inputBuffer.toInt();
                        if (mask >= 0 && mask <= 63) {
                            waveMode = false;
                            applyMask(mask);
                        }
                    }
                }
            }
            inputBuffer = "";
        } else {
            inputBuffer += c;
        }
    }

    if (waveMode) {
        waveServo();
    }
}