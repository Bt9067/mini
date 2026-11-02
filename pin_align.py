"""
จัดตำแหน่งสายพานให้ตรงกริปเปอร์ด้วยกล้องตัวที่ 2 (ปลายเข็ม 2 อันต้องอยู่ตำแหน่ง x เดียวกันในภาพ)

- เข็มติดสายพาน = ห้อยลงจากแผ่นใสด้านบน ปลายอยู่ล่าง -> เลื่อนตามสายพาน
- เข็มติดกริปเปอร์ = ตั้งขึ้นจากด้านล่าง ปลายอยู่บน    -> อยู่นิ่ง ใช้เป็นจุดอ้างอิง
แยกสองเข็มด้วยตำแหน่ง (ไม่ใช่สี): พื้นหลังเหลืองทำให้เข็มทั้งสองมืดกว่าพื้นเหมือนกัน (พื้นดำเข็มกลับสว่าง)
ในแต่ละ ROI เข็มสายพาน = ก้อนที่เริ่มสูงสุด, เข็มกริปเปอร์ = ก้อนที่ลงต่ำสุด
ROI ทั้งสองไม่ซ้อนกันแนวตั้ง (ตอนตรงกันปลายสองเข็มอยู่คอลัมน์เดียวกันจะต่อเป็นก้อนเดียว) จึงวัด x จากก้านช่วงที่
ใกล้กันที่สุด ไม่ใช่ปลายจริง -> เข็มต้องตั้งตรง
เข็มทั้งสองต้องอยู่ระนาบเดียวกัน (ระยะห่างจากกล้องเท่ากัน) ไม่งั้น x ตรงกันในภาพแต่ไม่ตรงกันจริง (parallax)

หลักการ: หาเข็มด้วย black-hat/top-hat แนวนอน (เข็มเป็นเส้นตั้งบางกว่าหน้าต่าง -> พื้นหลัง/กล่องกว้างๆ หายไป)
แล้วเก็บเส้นแนวตั้ง (open แนวตั้ง) ตำแหน่ง x = ค่ากลางของจุดกึ่งกลางแต่ละแถวช่วงปลายเข็ม
(แต่ละแถวใช้เฉพาะส่วนที่เข้มเกิน core_frac ของจุดเข้มสุด ไม่ให้เงาข้างเข็มดึง x ออกไป)
dx = x เข็มสายพาน - x เข็มกริปเปอร์ (px) -> แปลงเป็น mm ด้วยสเกลที่วัดเอง -> สั่ง M<mm> วนจนเหลือ <= tolerance

Usage:
  python tools/pin_align.py --image <ภาพ>.jpg        # ทดสอบการหาเข็มจากภาพนิ่ง (บันทึกภาพผลไว้ข้างภาพเดิม *_pins.jpg)
  python tools/pin_align.py --live                   # ดูกล้องสด + ระยะห่างเข็ม (ใช้ตอนจัดเข็มให้อยู่ระนาบเดียวกัน)
  python tools/pin_align.py --align                  # ต่อสายพาน (conveyor.serial_port) วัดสเกลแล้วเลื่อนจนเข็มตรงกัน

Controls (--live): Q = ออก, S = บันทึกภาพ
ค่าทั้งหมดอยู่ที่ pin_align: ใน config.yaml
"""
import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

DEFAULTS = {
    "source": 0,
    "frame_width": 1920,
    "frame_height": 1080,
    "conveyor_roi": [0.30, 0.31, 0.87, 0.44],  # x0, y0, x1, y1 (สัดส่วนภาพ) พื้นเหลืองใต้ขอบแผ่นใส ถึงเหนือปลายเข็มกริปเปอร์
    "gripper_roi": [0.30, 0.54, 0.87, 0.66],   # ใต้ปลายเข็มสายพาน ถึงเหนือตัวกริปเปอร์ดำ
    "bg_kernel_px": 61,                # หน้าต่าง top-hat แนวนอน ต้องกว้างกว่าเข็ม (~20 px) หลายเท่า
    "contrast": 30,                    # ต่างจากพื้นหลังเกินเท่านี้ = เป็นเข็ม
    "min_length_px": 120,              # ความสูงก้อนขั้นต่ำ (ต้องน้อยกว่าความสูง ROI)
    "max_width_px": 60,                # กว้างเกินนี้ไม่ใช่เข็ม
    "compare_y": 0.49,                 # เทียบ x สองเข็มที่ระดับนี้ (สัดส่วนความสูงภาพ; 0.49 = y 529 ช่วงปลายซ้อนกัน)
    "core_frac": 0.85,                 # แต่ละแถวใช้เฉพาะส่วนที่ต่างจากพื้น >= 85% ของจุดเข้มสุด (ตัดเงา)
    "conveyor_pin": "gray",            # gray = ไม่มีสีบนพื้นสี (พื้นเหลือง), dark = มืดกว่าพื้น, bright = สว่างกว่า
    "gripper_pin": "gray",
    "pin_sat_max": 60,                 # gray: saturation ต่ำกว่านี้ = เนื้อเข็ม (พื้นเหลือง 90-170, เข็ม 4-50)
    "pin_dark_max": 110,               # gray: มืดกว่านี้ = ขอบเข็ม (ขอบ 37-90, เงาบนพื้น ~150)
    "pin_dark_sat_max": 160,
    "bg_sat_min": 120,                 # พื้นเหลือง: S มากกว่านี้ และ V มากกว่า bg_val_min
    "bg_val_min": 40,
    "bg_hue": [12, 45],                # เฉดเหลือง (OpenCV H 0-180; พื้นเหลือง 21-34)
    "bg_bright_sat_min": 20,           # พื้นเหลืองสว่างจ้า (V > pin_bright_min) สีซีดได้ถึงเท่านี้ (แสงสะท้อนกลางเข็ม S 4-16)
    "side_gap_px": 8,                  # ดูพื้นข้างเข็มห่างจากขอบก้อนเท่านี้ (ข้ามขอบเบลอ/แสงสะท้อนข้างเข็มใต้ไฟ ~6 px) กว้าง 4 px
    "side_bg_min": 0.5,                # สองข้างต้องเป็นพื้นเหลืองอย่างน้อยสัดส่วนนี้           # gray: พิกเซลมืดต้อง S ต่ำกว่านี้ด้วย (เงาเหลืองใต้ไฟ S 195-250)
    "pin_bright_min": 210,             # gray: สว่างเกินนี้ใช้ pin_sat_max_bright แทน (พื้นเหลืองแสงจ้า V ~233)
    "pin_sat_max_bright": 25,
    "tolerance_px": 3.0,              # 1 สเต็ป = 0.05 mm ~ 2-3 px (กล้องเข็ม ~40-67 px/mm)
    "start_mm": 0.0,                   # ที่ Home มองไม่เห็นเข็มสายพาน: เลื่อน M+ รวดเดียวเท่านี้ก่อน (ถ้ารู้ระยะคร่าวๆ)
    "approach_mm": 5.0,                # แล้วเลื่อนหาทีละเท่านี้จนเจอเข็มสายพาน (ต้องน้อยกว่าความกว้าง ROI เป็น mm)
                                       # null = คำนวณเอง approach_roi_frac x ความกว้าง conveyor_roi / |scale_px_per_mm|
    "approach_roi_frac": 0.6,          # ก้าวหาเข็ม = สัดส่วนนี้ของความกว้าง ROI (< 1 = ทุกตำแหน่งเข็มต้องโผล่ใน ROI อย่างน้อย 1 ครั้ง)
    "approach_max_mm": 400.0,          # เลื่อนหารวมไม่เกินเท่านี้ (หรือจนชน End)
    "scale_px_per_mm": None,           # ใส่ค่า = ใช้สเกลคงที่ เลื่อนเข้าใกล้ครั้งเดียว (ไม่วัดด้วย calib_mm)
    "gripper_x_range": [950, 1150],    # เข็มกริปเปอร์ (นิ่ง) ต้องอยู่ช่วงนี้ ไม่งั้นถือว่ามีของบังกล้อง
    "roi_gap_px": 8,                   # --align: ขอบล่าง conveyor_roi อยู่เหนือยอดเข็มกริปเปอร์ (วัดตอน Home) อย่างน้อยเท่านี้
    "coarse_min_px": 80,               # ห่างเกินนี้ = เลื่อนหยาบเล็งเป้าตรงๆ, ไม่เกิน = ขยับทีละนิด
    "fine_px_per_mm": 100,             # สเกลสมมติตอนก้าวละเอียด (มากกว่าจริง -> ก้าวสั้น ไม่เลยเป้า)
    "min_step_mm": 0.05,               # ก้าวเล็กสุด = 1 สเต็ป
    "calib_mm": 2.0,                   # เลื่อนเท่านี้ตอนวัดสเกล px/mm (ต้องไม่พาเข็มหลุดจอ)
    "max_step_mm": 20.0,
    "lost_search_mm": 30.0,
    "transit_min_hits": 3,             # วิ่งไป start_mm แล้วเห็นเข็มสายพานอย่างน้อยกี่เฟรม = เข็มเลยไปแล้ว            # เข็มหลุดจอ: ถอยกลับเท่าที่เพิ่งเลื่อน แล้วถอยหาต่อได้อีกไม่เกินเท่านี้
    "max_iter": 10,
    "settle_sec": 0.5,
}


def load_cfg(path):
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    pcfg = dict(DEFAULTS)
    pcfg.update(cfg.get("pin_align") or {})
    return cfg, pcfg


def _roi_px(frac, shape):
    h, w = shape[:2]
    x0, y0, x1, y1 = frac
    return int(x0 * w), int(y0 * h), int(x1 * w), int(y1 * h)


def _find_pin(diff, roi, pcfg, tip, ref_x=None, bg=None):
    """หาเข็มจากภาพความต่างจากพื้นหลัง (ค่าบวก = เป็นเข็ม) คืน dict หรือ None
    tip="bottom" = เข็มห้อยลง (เลือกก้อนที่เริ่มสูงสุด ใช้ช่วงปลายล่าง)
    tip="top" = เข็มตั้งขึ้น (เลือกก้อนที่ลงต่ำสุด ใช้ช่วงปลายบน)"""
    x0, y0, x1, y1 = roi
    mask = np.zeros(diff.shape, np.uint8)
    mask[y0:y1, x0:x1] = (diff[y0:y1, x0:x1] > pcfg["contrast"]).astype(np.uint8) * 255
    vk = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, vk)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    best = None
    for i in range(1, n):
        x, y, w, h, _ = stats[i]
        if h < pcfg["min_length_px"] or w > pcfg["max_width_px"]:
            continue
        # ติดขอบซ้าย/ขวา ROI = ของที่ยื่นเข้ามา (ที่ ~320 mm ชิ้นส่วนดำของสายพานโผล่ขอบซ้าย) หรือเข็มยังเข้ามาไม่เต็ม
        if x <= x0 or x + w >= x1:
            continue
        # เข็มอยู่กลางพื้นเหลือง: สองข้างต้องเป็นพื้นเหลือง (ขอบนิ้วกริปเปอร์ดำ/ของดำมีสีดำอยู่ข้างหนึ่ง
        # 2026-10-29 หลังตัดพิกเซลมืดสีจัดออก ขอบนิ้วกริปเปอร์เหลือแถบบาง 9 px ถูกเลือกแทนเข็ม)
        if bg is not None:
            gap = int(pcfg["side_gap_px"])
            left = bg[y:y + h, max(x - gap - 4, 0):max(x - gap, 0)]
            right = bg[y:y + h, x + w + gap:x + w + gap + 4]
            if left.size == 0 or right.size == 0 or min(left.mean(), right.mean()) < pcfg["side_bg_min"]:
                continue
        if ref_x is not None:
            # รู้ตำแหน่งเข็มอยู่แล้ว (เข็มกริปเปอร์นิ่ง วัดตอน Home): เลือกก้อนที่ใกล้สุด
            # กรอบเขียวที่ยอดเข็มมีก้านเข็มสายพานผ่านด้วย ลงต่ำสุดเท่ากัน แยกด้วยวิธีเดิมไม่ได้
            if best is None or abs(x + w / 2 - ref_x) < abs(stats[best][0] + stats[best][2] / 2 - ref_x):
                best = i
        elif best is None or (y < stats[best][1] if tip == "bottom" else y + h > stats[best][1] + stats[best][3]):
            best = i
    if best is None:
        return None
    x, y, w, h, _ = stats[best]
    # จุดกึ่งกลางทุกแถว แล้วลากเส้นตรง x = a*y + b (เข็มอาจเอียง/งอ: รอบ 13 เข็มสายพานเอียง ~3 องศา
    # ปลายเยื้องจากก้าน 7-8 px) -> x ของเข็ม = ค่าบนเส้นที่ระดับ compare_y (ช่วงที่ปลายสองเข็มซ้อนกัน)
    ys, xs = [], []
    for row in range(y, y + h):
        cols = np.nonzero(labels[row, x:x + w] == best)[0]
        if len(cols) == 0:
            continue
        wts = diff[row, x + cols].astype(np.float64)
        core = wts >= pcfg["core_frac"] * wts.max()
        ys.append(row)
        xs.append(x + float(np.average(cols[core], weights=wts[core])))
    if len(xs) < 10:
        return None
    ys, xs = np.array(ys, float), np.array(xs, float)
    a, b = np.polyfit(ys, xs, 1)
    keep = np.abs(xs - (a * ys + b)) <= 3.0          # ตัดแถวที่คลาด (ขอบเงา/รอยบนเข็ม) แล้วลากใหม่
    if keep.sum() >= 10:
        a, b = np.polyfit(ys[keep], xs[keep], 1)
    cy = float(pcfg["compare_y"]) * diff.shape[0]
    return {"x": float(a * cy + b), "tip_y": int(y + h - 1 if tip == "bottom" else y),
            "box": (int(x), int(y), int(w), int(h)), "length": int(h), "line": (float(a), float(b))}


def _pin_diffs(frame, pcfg):
    """ภาพ "ความเป็นเข็ม" ของแต่ละแบบ (gray/dark/bright) ค่ามาก = เป็นเข็ม"""
    gray = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (3, 3), 0)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (int(pcfg["bg_kernel_px"]), 1))
    diffs = {}
    for pol in {pcfg["conveyor_pin"], pcfg["gripper_pin"]}:
        if pol == "gray":
            # เข็มโลหะไม่มีสี (saturation ต่ำ) ส่วนขอบเข็มมืดมาก; พื้นเหลืองและเงาบนพื้นยังมีสี -> ไม่ติดเงา
            # ทุกพิกเซลที่เป็นเข็มน้ำหนักเท่ากัน x = กึ่งกลางเนื้อเข็ม (ไม่ขึ้นกับแสงสะท้อนกลางเข็ม)
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            sat = hsv[..., 1]
            # พื้นเหลืองที่สว่างจัด (แสงจ้า) สีซีด S ลดเหลือ 36-55 -> ช่วงสว่างต้อง S ต่ำมาก (แสงสะท้อนกลางเข็ม S 4-16)
            sat_max = np.where(gray < pcfg["pin_bright_min"], pcfg["pin_sat_max"], pcfg["pin_sat_max_bright"])
            # มืดอย่างเดียวไม่พอ: เปิดไฟแล้วแผ่นใสทอดเงาบนพื้นเหลือง (V 50-115 แต่ S 195-250) -> ต้องสีไม่จัดด้วย (ขอบเข็ม S <= ~110)
            pin = (sat < sat_max) | ((gray < pcfg["pin_dark_max"]) & (sat < pcfg["pin_dark_sat_max"]))
            diffs[pol] = pin.astype(np.uint8) * 255
        else:
            diffs[pol] = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT if pol == "dark" else cv2.MORPH_TOPHAT, k)
    return diffs


def _trace_pin(diff, pin, pcfg, step):
    """ไล่ตามแนวเส้นเข็มออกนอก ROI (step -1 = ขึ้นจากขอบบนกรอบ, +1 = ลงจากขอบล่าง) จนเนื้อเข็มหมด คืน y สุดท้ายที่ยังเป็นเข็ม"""
    a, b = pin["line"]
    half = pin["box"][2] // 2 + 3
    x, y, w, h = pin["box"]
    end = y if step < 0 else y + h - 1
    miss = 0
    for row in range(end + step, -1 if step < 0 else diff.shape[0], step):
        cx = int(round(a * row + b))
        if np.count_nonzero(diff[row, max(cx - half, 0):cx + half + 1] > pcfg["contrast"]) >= 2:
            end, miss = row, 0
        else:
            miss += 1
            if miss >= 3:
                break
    return end


def gripper_top(frame, pcfg, grip):
    """y ยอดเข็มกริปเปอร์จริง (ใช้ตอนเข็มสายพานยังไม่อยู่ในภาพเท่านั้น ไม่งั้นจะไล่ต่อเข้าไปในเข็มสายพาน)"""
    return _trace_pin(_pin_diffs(frame, pcfg)[pcfg["gripper_pin"]], grip, pcfg, -1)


def conveyor_tip(frame, pcfg, conv):
    """y ปลายล่างเข็มสายพานจริง (ใช้ตอนเข็มสองอันยังห่างกัน ไม่งั้นจะไล่ต่อเข้าไปในเข็มกริปเปอร์)"""
    return _trace_pin(_pin_diffs(frame, pcfg)[pcfg["conveyor_pin"]], conv, pcfg, 1)


def set_gripper_roi_top(pcfg, y0_px, shape, why, y1_max):
    """ย้าย gripper_roi ให้เริ่มที่ y0_px สูงเท่า conveyor_roi (ผู้ใช้ให้สองกรอบเท่ากัน)
    แต่ไม่ต่ำเกิน y1_max (ขอบล่างใน config = เหนือตัวกริปเปอร์ดำ)"""
    h = shape[0]
    x0, _, x1, _ = pcfg["gripper_roi"]
    conv_h = (pcfg["conveyor_roi"][3] - pcfg["conveyor_roi"][1]) * h
    y1 = min(y0_px + conv_h, y1_max * h) / h
    roi_h = y1 * h - y0_px
    if roi_h < 30:
        raise SystemExit(f"กรอบหาเข็มกริปเปอร์เหลือแค่ {roi_h:.0f} px ({why}) ปลายเข็มสายพานยาวลงมาเกินไป")
    pcfg["gripper_roi"] = [x0, y0_px / h, x1, y1]
    pcfg["min_length_px"] = min(float(pcfg["min_length_px"]), 0.8 * roi_h)
    print(f"[ALIGN] gripper_roi y {y0_px:.0f}-{y1 * h:.0f} px ({why})")


def fit_conveyor_roi(pcfg, top, shape):
    """ตัดขอบล่าง conveyor_roi ให้จบเหนือยอดเข็มกริปเปอร์ (ย้ายเข็มขึ้น/ลงได้ ไม่ต้องแก้ config)
    ROI ซ้อนยอดเข็มกริปเปอร์ = สองเข็มต่อเป็นก้อนเดียว x ที่วัดได้อยู่กลางระหว่างสองเข็ม (2026-10-29 วัดได้ dx 0.3 จริง ~15 px)"""
    h = shape[0]
    x0, y0, x1, y1 = pcfg["conveyor_roi"]
    y1_px = min(y1 * h, top - float(pcfg["roi_gap_px"]))
    roi_h = y1_px - y0 * h
    if roi_h < 30:
        raise SystemExit(f"ยอดเข็มกริปเปอร์สูงเกินไป (y={top}) เหลือที่หาเข็มสายพานแค่ {roi_h:.0f} px "
                         f"ลดเข็มกริปเปอร์ลง หรือยก conveyor_roi y0 ขึ้น")
    pcfg["conveyor_roi"] = [x0, y0, x1, y1_px / h]
    # เข็มสายพานต้องยาวเกิน min_length_px ใน ROI ที่เตี้ยลง
    pcfg["min_length_px"] = min(float(pcfg["min_length_px"]), 0.8 * roi_h)
    print(f"[ALIGN] ยอดเข็มกริปเปอร์ y={top} -> conveyor_roi y {y0 * h:.0f}-{y1_px:.0f} px, "
          f"min_length_px {pcfg['min_length_px']:.0f}")


def find_pins(frame, pcfg):
    """คืน (เข็มสายพาน, เข็มกริปเปอร์) แต่ละตัวเป็น dict {x, tip_y, box, length} หรือ None"""
    diffs = _pin_diffs(frame, pcfg)
    # พื้นเหลือง = เฉดเหลือง และ (สีจัด + ไม่มืดสนิท หรือ สว่างจ้าสีซีด)
    # เงาแผ่นใสใต้ไฟ V 50-115 S 195-250 / ใต้ไฟช่วงล่างสว่างจ้า V ~243 S 28-57 ยังนับ; ของดำ V < 40 สีเพี้ยนไม่นับ
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hue, sat, val = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    h0, h1 = pcfg["bg_hue"]
    bg = ((hue >= h0) & (hue <= h1)
          & (((sat > pcfg["bg_sat_min"]) & (val > pcfg["bg_val_min"]))
             | ((val > pcfg["pin_bright_min"]) & (sat > pcfg["bg_bright_sat_min"])))).astype(np.float32)
    conv = _find_pin(diffs[pcfg["conveyor_pin"]], _roi_px(pcfg["conveyor_roi"], frame.shape), pcfg, "bottom", bg=bg)
    grip = _find_pin(diffs[pcfg["gripper_pin"]], _roi_px(pcfg["gripper_roi"], frame.shape), pcfg, "top",
                     pcfg.get("gripper_ref_x"), bg=bg)
    return conv, grip


def draw(frame, conv, grip, pcfg, text=None):
    out = frame.copy()
    h, w = out.shape[:2]
    for pin, color, name in ((conv, (255, 128, 0), "conveyor"), (grip, (0, 200, 0), "gripper")):
        rx0, ry0, rx1, ry1 = _roi_px(pcfg[f"{name}_roi"], out.shape)
        cv2.rectangle(out, (rx0, ry0), (rx1, ry1), color, 1)
        if pin is None:
            continue
        x, y, bw, bh = pin["box"]
        cv2.rectangle(out, (x, y), (x + bw, y + bh), color, 1)
        # เส้นแนวเข็มที่ลากได้ ต่อไปถึงระดับ compare_y + จุดที่ใช้เทียบ
        a, b = pin["line"]
        cy = int(round(pcfg["compare_y"] * h))
        ya, yb = (y, cy) if cy > y else (cy, y + bh)
        cv2.line(out, (int(round(a * ya + b)), ya), (int(round(a * yb + b)), yb), color, 2)
        cv2.circle(out, (int(round(pin["x"])), cy), 6, color, 2)
        cv2.putText(out, f"{name} x={pin['x']:.1f}", (x - 60, max(y - 12, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    if text is None:
        if conv and grip:
            dx = conv["x"] - grip["x"]
            ok = abs(dx) <= pcfg["tolerance_px"]
            text = f"dx = {dx:+.1f} px  {'ALIGNED' if ok else ''}"
        else:
            text = "pin not found: " + ", ".join(n for p, n in ((conv, "conveyor"), (grip, "gripper")) if p is None)
    cv2.putText(out, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 255), 3)
    return out


def open_camera(pcfg):
    src = pcfg["source"]
    cap = cv2.VideoCapture(int(src) if str(src).isdigit() else src)
    if not cap.isOpened():
        raise SystemExit(f"เปิดกล้องไม่ได้: {src}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, pcfg["frame_width"])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, pcfg["frame_height"])
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    print(f"ความละเอียดกล้อง: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
    return cap


def fresh_frame(cap, flush=5):
    """ทิ้งเฟรมค้างในบัฟเฟอร์ แล้วคืนเฟรมล่าสุด (หลังสายพานหยุด)"""
    for _ in range(flush):
        cap.grab()
    ok, frame = cap.read()
    if not ok:
        raise RuntimeError("อ่านภาพจากกล้องไม่ได้")
    return frame


def measure(cap, pcfg, shots=3):
    """วัดตำแหน่งเข็มหลายภาพแล้วเอาค่ากลาง คืน (x สายพาน, x กริปเปอร์, เฟรมสุดท้าย, เข็มสุดท้าย)"""
    xc, xg = [], []
    for _ in range(shots):
        frame = fresh_frame(cap)
        conv, grip = find_pins(frame, pcfg)
        if conv:
            xc.append(conv["x"])
        if grip:
            xg.append(grip["x"])
    return (float(np.median(xc)) if xc else None, float(np.median(xg)) if xg else None, frame, (conv, grip))


def run_image(path, pcfg):
    frame = cv2.imread(str(path))
    if frame is None:
        raise SystemExit(f"อ่านภาพไม่ได้: {path}")
    conv, grip = find_pins(frame, pcfg)
    for name, pin in (("เข็มสายพาน", conv), ("เข็มกริปเปอร์", grip)):
        print(f"{name}: " + (f"x={pin['x']:.1f} ช่วง y={pin['box'][1]}-{pin['box'][1] + pin['box'][3]} ยาว {pin['length']} px" if pin else "ไม่เจอ"))
    if conv and grip:
        print(f"dx = {conv['x'] - grip['x']:+.1f} px")
    out_path = Path(path).with_name(Path(path).stem + "_pins.jpg")
    cv2.imwrite(str(out_path), draw(frame, conv, grip, pcfg))
    print(f"บันทึกภาพผล: {out_path}")


def run_live(pcfg):
    cap = open_camera(pcfg)
    cv2.namedWindow("pin_align", cv2.WINDOW_NORMAL)
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        conv, grip = find_pins(frame, pcfg)
        cv2.imshow("pin_align", draw(frame, conv, grip, pcfg))
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("s"):
            name = time.strftime("pin_%Y%m%d_%H%M%S.jpg")
            cv2.imwrite(name, frame)
            print(f"บันทึก {name}")
    cap.release()
    cv2.destroyAllWindows()


def run_align(cfg, pcfg):
    from conveyor import Conveyor

    cap = open_camera(pcfg)
    cv2.namedWindow("pin_align", cv2.WINDOW_NORMAL)
    status = ["Home..."]   # ข้อความสถานะบนหน้าต่างสด
    # ระหว่างวิ่งไป start_mm นับเฟรมที่เห็นเข็มสายพาน (เข็มกริปเปอร์ยังอยู่ที่เดิม) -> เคยเห็น = เข็มเลยไปแล้ว
    transit = {"on": False, "hits": 0, "xg": None}

    def live():
        """แสดงภาพกล้องสด + เข็มที่หาได้ในเฟรมนั้น (ระหว่างรอสายพาน/รอนิ่ง ไม่ให้หน้าต่างค้าง)
        ค่าที่ใช้สั่งสายพานยังมาจาก measure() ตอนสายพานหยุดนิ่งเท่านั้น"""
        ok, frame = cap.read()
        if not ok:
            return
        conv, grip = find_pins(frame, pcfg)
        if transit["on"] and conv and grip and abs(grip["x"] - transit["xg"]) <= 10:
            transit["hits"] += 1
        dx = f"  dx={conv['x'] - grip['x']:+.1f}px" if conv and grip else ""
        cv2.imshow("pin_align", draw(frame, conv, grip, pcfg, status[0] + dx))
        cv2.waitKey(1)

    def wait_live(sec):
        end = time.time() + sec
        while time.time() < end:
            live()

    conv_cfg = cfg["conveyor"]
    belt = Conveyor(conv_cfg["serial_port"], conv_cfg.get("baudrate", 115200), on_wait=live)

    def show(frame, pins, text=None):
        cv2.imshow("pin_align", draw(frame, *pins, pcfg, text))
        cv2.waitKey(1)

    try:
        xc0, xg, frame, pins = measure(cap, pcfg)
        show(frame, pins)
        if xg is None:
            raise SystemExit("หาเข็มกริปเปอร์ไม่เจอใน gripper_roi (มีของบังกล้อง/เข็มอยู่นอกแผ่นเหลือง)")
        # gripper_x_range: null = เข็มกริปเปอร์อยู่ตรงไหนก็ได้ใน gripper_roi (ย้ายเข็มได้ ไม่ต้องแก้ config)
        if pcfg.get("gripper_x_range"):
            gx0, gx1 = pcfg["gripper_x_range"]
            if not gx0 <= xg <= gx1:
                # 2026-10-10: นิ้วกริปเปอร์ยกขึ้นบังกล้อง เคยจับขอบแผ่นเหลืองเป็นเข็มทั้งสองอันแล้วสั่งถอยชน Home
                raise SystemExit(f"เข็มกริปเปอร์ไม่อยู่ในช่วง x {gx0}-{gx1} (ได้ {xg}) มีของบังกล้อง/กริปเปอร์ไม่อยู่ท่าปกติ")
        print(f"[ALIGN] เข็มกริปเปอร์ x={xg:.1f}")
        grip_roi_cfg = list(pcfg["gripper_roi"])

        def bottom_y0():
            """ขอบบนกรอบเขียวเมื่อชิดขอบล่าง (y1 ใน config) และสูงเท่ากรอบฟ้า"""
            hh = frame.shape[0]
            return grip_roi_cfg[3] * hh - (pcfg["conveyor_roi"][3] - pcfg["conveyor_roi"][1]) * hh
        if pins[1] is not None:
            top = gripper_top(frame, pcfg, pins[1])
            fit_conveyor_roi(pcfg, top, frame.shape)
            # กรอบเขียวอยู่ด้านล่างตลอด (ผู้ใช้ขอ): ชิดขอบล่างใน config (เหนือตัวกริปเปอร์ดำ) สูงเท่ากรอบฟ้า
            set_gripper_roi_top(pcfg, bottom_y0(), frame.shape, "ด้านล่าง", grip_roi_cfg[3])
            xc0, xg, frame, pins = measure(cap, pcfg)
            if xg is None:
                raise SystemExit("กรอบเขียวที่ยอดเข็มหาเข็มกริปเปอร์ไม่เจอ")
            xc0 = None
            print(f"[ALIGN] เข็มกริปเปอร์ (กรอบที่ยอดเข็ม) x={xg:.1f}")
            pcfg["gripper_ref_x"] = xg   # จากนี้เลือกก้อนในกรอบเขียวที่ใกล้ x นี้สุด (ก้านเข็มสายพานจะผ่านกรอบเขียวด้วย)
        xc0 = None   # ที่ Home เข็มสายพานไม่อยู่ในภาพแน่นอน ถ้าเจอ = จับผิด
        # ระยะก้าวหาเข็ม: null = คำนวณจากความกว้าง ROI เป็น mm (เข็มอยู่ตรงไหนบนสายพานก็ต้องโผล่ใน ROI สักครั้ง)
        approach = pcfg.get("approach_mm")
        if not approach:
            if not pcfg.get("scale_px_per_mm"):
                raise SystemExit("approach_mm: null ต้องตั้ง scale_px_per_mm ด้วย (ใช้แปลงความกว้าง ROI เป็น mm)")
            rx0, _, rx1, _ = _roi_px(pcfg["conveyor_roi"], frame.shape)
            approach = float(pcfg["approach_roi_frac"]) * (rx1 - rx0) / abs(float(pcfg["scale_px_per_mm"]))
        approach = float(approach)
        print(f"[ALIGN] เลื่อนหาเข็มทีละ {approach:.1f} mm")
        # ที่ Home เข็มสายพานไม่อยู่ในภาพ: เลื่อน M+ หาจนเจอ (บันทึกภาพทุกช่วงไว้ดูทีหลัง)
        moved = 0.0
        dbg_dir = ROOT / "results" / "pin_align" / time.strftime("%Y%m%d_%H%M%S")
        dbg_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(dbg_dir / "approach_000.0mm.jpg"), frame)
        def look(pos):
            """วัดหลังเลื่อนช่วงหาเข็ม คืน (x เข็มสายพาน หรือ None, เฟรม, เข็ม)"""
            wait_live(pcfg["settle_sec"])
            xc, xg1, frame, pins = measure(cap, pcfg)
            cv2.imwrite(str(dbg_dir / f"approach_{pos:05.1f}mm.jpg"), frame)
            # ช่วง ~285-305 mm มีชิ้นส่วนดำของสายพานบังกล้อง เคยจับขอบมันเป็นเข็มทั้งสองอัน
            # -> เชื่อเข็มสายพานเฉพาะเมื่อเข็มกริปเปอร์ (นิ่ง) ยังอยู่ที่เดิม
            if xc is not None and (xg1 is None or abs(xg1 - xg) > 10):
                print(f"[ALIGN] {pos:.0f} mm: มีของบังกล้อง (เข็มกริปเปอร์ไม่อยู่ที่เดิม) ไม่นับ")
                xc = None
            print(f"[ALIGN] หาเข็ม {pos:.0f} mm: สายพาน " + (f"x={xc:.1f}" if xc is not None else "ไม่เจอ")
                  + ", กริปเปอร์ " + (f"x={xg1:.1f}" if xg1 is not None else "ไม่เจอ"))
            show(frame, pins, f"searching conveyor pin... {pos:.0f} mm")
            return xc, frame, pins

        def search_back():
            """ถอยกลับหาเข็มทาง Home ทีละก้าว คืน (xc, เฟรม, เข็ม) ไม่เจอ = xc None
            ไม่ลงไปวัดที่ Home (ที่ Home เข็มสายพานไม่อยู่ในภาพ ถ้าเจอ = จับผิด) ถอยถึงแค่ประมาณ 1 ก้าวจาก Home"""
            xc, fr, pn = None, frame, pins
            while xc is None and belt.position_mm - approach > 0:
                status[0] = f"searching back... {belt.position_mm - approach:.0f} mm"
                print(f"[ALIGN] ถอยกลับหา {-approach:+.1f} mm")
                belt.move(-approach)
                xc, fr, pn = look(belt.position_mm)
            return xc, fr, pn

        start = float(pcfg["start_mm"])
        passed = False
        if start > 0:
            print(f"[ALIGN] เลื่อนไปจุดเริ่ม {start} mm")
            status[0] = f"moving to start {start} mm"
            transit.update(on=True, hits=0, xg=xg)
            belt.move(start)
            transit["on"] = False
            moved += start
            # ผู้ใช้ (2026-10-29): ระหว่างวิ่งไป start กล้องเคยเห็นเข็มสายพาน = เข็มเลยไปแล้ว -> ถอยหา
            # ไม่เคยเห็น = เข็มยังมาไม่ถึง -> หาข้างหน้า (เดิมถอยหาก่อนเสมอ start 300 ถอยถึง Home 14 ก้าวเปล่า)
            passed = transit["hits"] >= int(pcfg["transit_min_hits"])
            print(f"[ALIGN] ระหว่างวิ่งไป {start} mm เห็นเข็มสายพาน {transit['hits']} เฟรม -> "
                  + ("เข็มเลยไปแล้ว ถอยหา" if passed else "เข็มยังไม่ถึง หาข้างหน้า"))
            xc0, frame, pins = look(belt.position_mm)
            if xc0 is None and passed:
                xc0, frame, pins = search_back()
                if xc0 is None:
                    # เห็นผิด (ของอื่นตอนวิ่ง) -> กลับไปจุดเริ่มแล้วหาข้างหน้า
                    print(f"[ALIGN] ถอยถึง Home แล้วไม่เจอเข็ม กลับไปหาข้างหน้า {start} mm")
                    belt.goto(start)
        tried_back = passed
        while xc0 is None:
            if belt.at_end or moved >= float(pcfg["approach_max_mm"]):
                why = "ชน End" if belt.at_end else f"เลื่อนหาถึง approach_max_mm {pcfg['approach_max_mm']}"
                if tried_back or start <= 0:
                    raise SystemExit(f"{why} แล้วยังไม่เจอเข็มสายพาน (เลื่อนไป {moved:.0f} mm)")
                # ตอนวิ่งไป start อาจมองไม่ทัน (ภาพเบลอ) ทั้งที่เข็มเลยไปแล้ว -> กลับไปจุดเริ่มแล้วถอยหา
                print(f"[ALIGN] {why} ไม่เจอเข็ม กลับไป {start} mm แล้วถอยหา")
                belt.goto(start)
                belt.at_end = False
                tried_back = True
                xc0, frame, pins = search_back()
                if xc0 is None:
                    raise SystemExit("หาทั้งข้างหน้าและข้างหลังจุดเริ่มแล้วไม่เจอเข็มสายพาน")
                break
            status[0] = f"searching conveyor pin... {moved + approach:.0f} mm"
            belt.move(approach)
            moved += approach
            xc0, frame, pins = look(belt.position_mm)
        if moved:
            print(f"[ALIGN] เจอเข็มสายพานที่ {belt.position_mm:.1f} mm (x={xc0:.1f})")
        # ปลายเข็มสายพานต้องไม่ห้อยเข้ามาในกรอบเขียว (ตอนตรงกันสองเข็มต่อเป็นก้อนเดียว)
        # ปลายเข็มสายพานสูงคงที่ (ห้อยจากแผ่นใส) วัดครั้งเดียวตอนนี้ที่สองเข็มยังห่างกัน; ใกล้กันเกินไป = คงกรอบด้านล่างไว้
        if pins[0] is not None and abs(xc0 - xg) > 2 * float(pcfg["max_width_px"]):
            tip = conveyor_tip(frame, pcfg, pins[0])
            # ยังอยู่ด้านล่าง แต่ถ้าปลายเข็มสายพานยาวลงมาถึงกรอบ ดันขอบบนลงใต้ปลาย (กรอบเตี้ยลง)
            if tip + float(pcfg["roi_gap_px"]) > bottom_y0():
                set_gripper_roi_top(pcfg, tip + float(pcfg["roi_gap_px"]), frame.shape, f"ด้านล่าง ใต้ปลายเข็มสายพาน y={tip}", grip_roi_cfg[3])
            else:
                print(f"[ALIGN] ปลายเข็มสายพาน y={tip} อยู่เหนือกรอบเขียว ไม่ต้องปรับ")
        xc0, xg1, frame, pins = measure(cap, pcfg)
        if xc0 is None or xg1 is None:
            raise SystemExit("ปรับกรอบเขียวแล้ววัดเข็มไม่เจอ")
        xg = xg1

        if pcfg.get("scale_px_per_mm"):
            # ใช้สเกลคงที่ (วัดจากการเลื่อนระยะยาว) -> เลื่อนครั้งแรกเข้าใกล้ทันที ไม่ต้องวัดสเกลก่อน
            px_per_mm = float(pcfg["scale_px_per_mm"])
            xc = xc0
            print(f"[ALIGN] ใช้สเกลคงที่ {px_per_mm:+.2f} px/mm")
        else:
            # วัดสเกล: เลื่อน calib_mm แล้วดูว่าเข็มสายพานขยับกี่ px (มีเครื่องหมาย = ทิศด้วย)
            # ยังไม่รู้ทิศ จึงสั่ง M+ ก่อน (calib_mm ต้องเล็กพอไม่ให้เข็มหลุดจอ)
            calib = float(pcfg["calib_mm"])
            belt.move(calib)
            wait_live(pcfg["settle_sec"])
            xc, xg1, frame, pins = measure(cap, pcfg)
            show(frame, pins)
            if xc is None:
                raise SystemExit("เข็มสายพานหลุดจอตอนวัดสเกล ลด calib_mm")
            xg = xg1 if xg1 is not None else xg
            px_per_mm = (xc - xc0) / calib
            if abs(px_per_mm) < 1.0:
                raise SystemExit("เลื่อนแล้วเข็มแทบไม่ขยับ (สายพานไม่เดิน/จับเข็มผิดอัน)")
            print(f"[ALIGN] สเกล {px_per_mm:+.2f} px/mm (บวก = สั่ง M+ แล้วเข็มไปทางขวาของภาพ)")

        # ครั้งแรก (และทุกครั้งที่ยังห่างเกิน coarse_min_px) เลื่อนหยาบเล็งเป้าตรงๆ ให้ใกล้ที่สุดในครั้งเดียว
        # จากนั้นขยับทีละนิด (fine_px_per_mm มากกว่าสเกลจริง -> ก้าวสั้น) ทิศไหนก็ได้ตามที่เข็มอยู่
        # สายพานกลับทิศแล้วกระโดดเกินสั่ง -> ทุกครั้งที่ก้าวละเอียดกลับทิศ ทำให้ก้าวสั้นลง 1.5 เท่า กันแกว่ง
        def recover_lost(back_mm, it):
            """เข็มสายพานหลุดจอหลังเลื่อน (เลยไปไกล/สายพานกระโดด): ถอยกลับเท่าที่เพิ่งเลื่อน แล้วถอยต่อทีละครึ่งก้าวหา
            จนเห็นเข็มอีกครั้ง (ไม่เกิน lost_search_mm จากจุดที่ถอยครบ) คืน (xc, xg)"""
            sign = 1.0 if back_mm > 0 else -1.0
            limit = abs(back_mm) + float(pcfg["lost_search_mm"])
            step, moved_back, n = abs(back_mm), 0.0, 0
            while moved_back < limit:
                if (sign < 0 and (belt.position_mm or 0.0) <= 0.5) or (sign > 0 and belt.at_end):
                    break                                # ถอยต่อไม่ได้แล้ว (ถึง Home/End)
                print(f"[ALIGN]   เข็มสายพานหลุดจอ ถอยกลับ {sign * step:+.2f} mm")
                status[0] = f"pin lost: back {sign * step:+.2f} mm"
                belt.move(sign * step)
                moved_back += step
                n += 1
                wait_live(pcfg["settle_sec"])
                xc_r, xg_r, frame_r, pins_r = measure(cap, pcfg)
                cv2.imwrite(str(dbg_dir / f"align_{it:02d}_back{n}.jpg"), frame_r)
                show(frame_r, pins_r)
                # เชื่อเฉพาะเมื่อเข็มกริปเปอร์ยังอยู่ที่เดิม (กันจับของอื่นที่โผล่มาตอนถอย)
                if xc_r is not None and (xg_r is None or abs(xg_r - xg) <= 10):
                    print(f"[ALIGN]   เจอเข็มสายพานอีกครั้ง x={xc_r:.1f} (ถอยรวม {moved_back:.2f} mm)")
                    return xc_r, xg_r
                step = min(0.5 * approach, limit - moved_back)
            raise SystemExit(f"เข็มสายพานหลุดจอ ถอยกลับ {moved_back:.1f} mm แล้วยังไม่เจอ (ภาพอยู่ที่ {dbg_dir})")

        fine = abs(float(pcfg["fine_px_per_mm"])) * (1.0 if px_per_mm > 0 else -1.0)
        min_mm = float(pcfg["min_step_mm"])
        last_dir = 0.0
        for it in range(1, int(pcfg["max_iter"]) + 1):
            dx = xc - xg
            print(f"[ALIGN] รอบ {it}: dx = {dx:+.1f} px")
            if abs(dx) <= pcfg["tolerance_px"]:
                print(f"[ALIGN] ตรงแล้ว (dx {dx:+.1f} px = {dx / px_per_mm:+.3f} mm) ตำแหน่งสายพาน {belt.position_mm} mm")
                # แสดงภาพสดต่อจนกดปุ่มใดก็ได้ (หน้าต่างไม่ค้างภาพนิ่ง)
                status[0] = f"ALIGNED (measured dx={dx:+.1f}px) belt={belt.position_mm}mm  press any key"
                while True:
                    ok, live_frame = cap.read()
                    if ok:
                        c, g = find_pins(live_frame, pcfg)
                        now = f"  live dx={c['x'] - g['x']:+.1f}px" if c and g else ""
                        cv2.imshow("pin_align", draw(live_frame, c, g, pcfg, status[0] + now))
                    if cv2.waitKey(1) != -1 or cv2.getWindowProperty("pin_align", cv2.WND_PROP_VISIBLE) < 1:
                        break
                return
            if abs(dx) > float(pcfg["coarse_min_px"]):
                mode = "หยาบ"                            # ไกล: เล็งเป้าตรงๆ
                mm = -dx / px_per_mm
            else:
                mode = "ละเอียด"                         # ใกล้: ก้าวเล็ก อย่างน้อย min_step_mm
                mm = -dx / fine
                if abs(mm) < min_mm:
                    mm = min_mm if mm > 0 else -min_mm
                if last_dir and mm * last_dir < 0:
                    fine *= 1.5                          # กลับทิศ: ก้าวถัดไปสั้นลง
            last_dir = 1.0 if mm > 0 else -1.0
            mm = float(np.clip(mm, -pcfg["max_step_mm"], pcfg["max_step_mm"]))
            print(f"[ALIGN]   {mode} เลื่อน {mm:+.2f} mm")
            status[0] = f"step {it}: {'coarse' if mode == 'หยาบ' else 'fine'} {mm:+.2f} mm"
            belt.move(mm)
            wait_live(pcfg["settle_sec"])
            xc_new, xg_new, frame, pins = measure(cap, pcfg)
            cv2.imwrite(str(dbg_dir / f"align_{it:02d}.jpg"), frame)
            show(frame, pins)
            for retry in range(3):
                if xc_new is not None:
                    break
                # หาไม่เจอ: รอให้เข็มที่ห้อยอยู่หยุดแกว่งแล้ววัดใหม่
                wait_live(1.0)
                xc_new, xg_new, frame, pins = measure(cap, pcfg)
                cv2.imwrite(str(dbg_dir / f"align_{it:02d}_retry{retry + 1}.jpg"), frame)
                show(frame, pins)
            if xc_new is None:
                # ไม่หยุดโปรแกรม: ถอยกลับทางที่มาจนเห็นเข็ม แล้ววนแก้ dx ต่อ (การเลื่อนนี้ไม่ใช้วัดสเกล)
                xc_new, xg_new = recover_lost(-mm, it)
                mode = "ถอยหา"
                last_dir = -last_dir
            # ใช้สเกลจากการเลื่อนหยาบทิศเข้าเป้า (เลื่อนสั้น/กลับทิศไม่น่าเชื่อ: สายพานกระตุก/คลอน)
            if mode == "หยาบ" and abs(mm) >= 0.5:
                measured = (xc_new - xc) / mm
                if it == 1 and pcfg.get("scale_px_per_mm") and not 0.5 <= measured / px_per_mm <= 2.0:
                    # เลื่อนหาจาก Home ผ่านช่วงที่ไม่เคยทดสอบ: ถ้าสิ่งที่เจอไม่ขยับตามสายพาน = จับของอื่นเป็นเข็ม
                    raise SystemExit(f"สิ่งที่จับเป็นเข็มสายพานขยับ {measured:+.1f} px/mm (คาด {px_per_mm:+.1f}) "
                                     f"น่าจะจับผิดอัน ดูภาพที่ {dbg_dir}")
                if measured * px_per_mm > 0:
                    px_per_mm = measured
                    print(f"[ALIGN] สเกลจากการเลื่อนจริง {px_per_mm:+.2f} px/mm")
            xc = xc_new
            xg = xg_new if xg_new is not None else xg
        print(f"[ALIGN] ครบ {pcfg['max_iter']} รอบแล้วยังไม่ตรง (dx {xc - xg:+.1f} px)")
    finally:
        belt.close()
        cap.release()
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser()
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument("--image", help="ทดสอบกับภาพนิ่ง")
    g.add_argument("--live", action="store_true", help="ดูกล้องสด")
    g.add_argument("--align", action="store_true", help="สั่งสายพานจนเข็มตรงกัน")
    parser.add_argument("--source", default=None, help="index กล้องตัวที่ 2 (ค่าเริ่มต้น pin_align.source)")
    parser.add_argument("--config", default=str(ROOT / "config.yaml"))
    args = parser.parse_args()

    cfg, pcfg = load_cfg(args.config)
    if args.source is not None:
        pcfg["source"] = args.source
    if args.image:
        run_image(args.image, pcfg)
    elif args.live:
        run_live(pcfg)
    else:
        run_align(cfg, pcfg)


if __name__ == "__main__":
    main()
