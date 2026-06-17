"""
╔══════════════════════════════════════════════════════════════════════╗
║         N E O N   A I R   D R A W   v5  —  CINEMATIC EDITION       ║
╠══════════════════════════════════════════════════════════════════════╣
║  GESTURES                                                           ║
║  ☝  Index only          → DRAW                                      ║
║  ✌  Index + Middle      → COLOUR PICKER  (hover index over swatch)  ║
║  🤙  Pinky only          → UNDO  (hold for fast undo)                ║
║  ✊  Fist               → CLEAR canvas                              ║
║  🖐  All 5 up            → ERASE                                     ║
║  [ ] keys               → Brush size smaller / bigger               ║
║  S   key                → Save drawing as PNG                       ║
║  ESC key                → Quit                                       ║
╚══════════════════════════════════════════════════════════════════════╝
"""

import cv2
import mediapipe as mp
import numpy as np
import time
import math
from collections import deque
from datetime import datetime

# ══════════════════════════════════════════════════════
#  MEDIAPIPE
# ══════════════════════════════════════════════════════
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.78,
    min_tracking_confidence=0.78,
    model_complexity=1,
)
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 60)

# ══════════════════════════════════════════════════════
#  PALETTE  (BGR) — rich cinematic colours
# ══════════════════════════════════════════════════════
PALETTE = [
    ("CRIMSON",  (  45,  25, 230)),
    ("CORAL",    (  60, 110, 255)),
    ("AMBER",    (   0, 190, 255)),
    ("LIME",     (  30, 255, 100)),
    ("AQUA",     ( 230, 240,  50)),
    ("SKY",      ( 255, 180,  50)),
    ("VIOLET",   ( 230,  30, 160)),
    ("ROSE",     ( 140,  60, 255)),
    ("WHITE",    ( 230, 230, 230)),
]

SKEL_COLORS = [
    (190, 190, 190),
    (  0, 150, 255),
    (  0, 230, 255),
    ( 50, 255, 110),
    (255,  50, 200),
    (170,  40, 255),
]
SKEL_GROUPS = [
    [0],
    [1,2,3,4],
    [5,6,7,8],
    [9,10,11,12],
    [13,14,15,16],
    [17,18,19,20],
]
CONNECTIONS = list(mp_hands.HAND_CONNECTIONS)

# ══════════════════════════════════════════════════════
#  STATE
# ══════════════════════════════════════════════════════
canvas      = None
prev_pt     = None
color_idx   = 0
draw_color  = PALETTE[0][1]
eraser_mode = False

BRUSH_SIZES    = [2, 4, 6, 10, 16, 24]
brush_size_idx = 2
brush_size     = BRUSH_SIZES[brush_size_idx]

stroke_stack: deque = deque(maxlen=50)
in_stroke    = False

smooth_x, smooth_y = 0, 0
ALPHA   = 0.45
MIN_MOV = 3

last_gesture       = 'idle'
gesture_start_time = 0.0
GESTURE_HOLD       = 0.07

undo_count  = 0
undo_last_t = 0.0
UNDO_FIRST  = 0.45
UNDO_REPEAT = 0.10

fps_times: deque = deque(maxlen=30)
mode_label = "NO HAND"

SWATCH_R      = 22
SWATCH_GAP    = 10
PICKER_Y_FRAC = 0.18
DWELL_NEEDED  = 0.55
picker_hovered     = -1
picker_dwell_start = 0.0

_vignette = None

# ══════════════════════════════════════════════════════
#  GESTURE HELPERS
# ══════════════════════════════════════════════════════
def finger_up(lm, tip, pip, mcp):
    return lm[tip].y < lm[pip].y and lm[pip].y < lm[mcp].y + 0.025

def thumb_up(lm):
    palm_cx = (lm[0].x + lm[9].x) / 2
    return abs(lm[4].x - palm_cx) > abs(lm[3].x - palm_cx) + 0.01

def get_states(lm):
    t = thumb_up(lm)
    i = finger_up(lm, 8,  6,  5)
    m = finger_up(lm, 12, 10, 9)
    r = finger_up(lm, 16, 14, 13)
    p = finger_up(lm, 20, 18, 17)
    return t, i, m, r, p

def classify(lm):
    t, i, m, r, p = get_states(lm)
    up = sum([t, i, m, r, p])
    if up == 5:                              return 'erase'
    if i and not m and not r and not p:      return 'draw'
    if i and m and not r and not p:          return 'pick'
    if p and not i and not m and not r:      return 'undo'
    if not i and not m and not r and not p:  return 'clear'
    return 'idle'

# ══════════════════════════════════════════════════════
#  VISUAL HELPERS
# ══════════════════════════════════════════════════════
def neon_line(dst, p1, p2, color, core):
    if p1 == p2: return
    b, g, r = color
    layers = [
        (core+18, (b//14, g//14, r//14)),
        (core+11, (b//7,  g//7,  r//7 )),
        (core+6,  (b//3,  g//3,  r//3 )),
        (core+2,  (b//2,  g//2,  r//2 )),
        (core,     color),
        (max(1,core-2), (min(255,b+100), min(255,g+100), min(255,r+100))),
        (max(1,core-4), (255, 255, 255)),
    ]
    for th, col in layers:
        cv2.line(dst, p1, p2, col, th, cv2.LINE_AA)

def eraser_stroke(dst, p1, p2, size):
    cv2.line(dst, p1, p2, (0,0,0), size*2+6, cv2.LINE_AA)

def draw_skeleton(img, lm, H, W):
    pts = [(int(l.x*W), int(l.y*H)) for l in lm]
    for a, b in CONNECTIONS:
        fi = 0
        for gi, grp in enumerate(SKEL_GROUPS):
            if a in grp or b in grp: fi = gi; break
        col = SKEL_COLORS[fi]
        d = tuple(max(0, c//7) for c in col)
        m = tuple(max(0, c//2) for c in col)
        cv2.line(img, pts[a], pts[b], d, 9, cv2.LINE_AA)
        cv2.line(img, pts[a], pts[b], m, 4, cv2.LINE_AA)
        cv2.line(img, pts[a], pts[b], col, 1, cv2.LINE_AA)
    for fi, grp in enumerate(SKEL_GROUPS):
        col = SKEL_COLORS[fi]
        d = tuple(max(0, c//7) for c in col)
        m = tuple(max(0, c//2) for c in col)
        for idx in grp:
            tip = idx in [4,8,12,16,20]
            R = 9 if tip else 5
            cv2.circle(img, pts[idx], R+5, d,   -1, cv2.LINE_AA)
            cv2.circle(img, pts[idx], R+2, m,   -1, cv2.LINE_AA)
            cv2.circle(img, pts[idx], R,   col, -1, cv2.LINE_AA)
            cv2.circle(img, pts[idx], max(1,R-3), (255,255,255), -1, cv2.LINE_AA)

def draw_cursor(img, cx, cy, color, core, eraser):
    if eraser:
        hs = 28
        cv2.rectangle(img, (cx-hs,cy-hs), (cx+hs,cy+hs), (80,80,80), 2, cv2.LINE_AA)
        cv2.line(img, (cx-hs,cy-hs), (cx+hs,cy+hs), (80,80,80), 1, cv2.LINE_AA)
        cv2.line(img, (cx+hs,cy-hs), (cx-hs,cy+hs), (80,80,80), 1, cv2.LINE_AA)
    else:
        b, g, r = color
        cv2.circle(img, (cx,cy), core+10, tuple(max(0,c//4) for c in color), 2, cv2.LINE_AA)
        cv2.circle(img, (cx,cy), core+5,  tuple(max(0,c//2) for c in color), 1, cv2.LINE_AA)
        cv2.circle(img, (cx,cy), 4, color, -1, cv2.LINE_AA)
        cv2.circle(img, (cx,cy), 2, (255,255,255), -1, cv2.LINE_AA)

# ══════════════════════════════════════════════════════
#  PICKER UI — dwell-to-select swatches
# ══════════════════════════════════════════════════════
def picker_positions(W, H):
    n     = len(PALETTE)
    total = n*(SWATCH_R*2+SWATCH_GAP) - SWATCH_GAP
    sx    = (W-total)//2
    sy    = int(H*PICKER_Y_FRAC)
    return [(sx + i*(SWATCH_R*2+SWATCH_GAP)+SWATCH_R, sy) for i in range(n)]

def draw_picker(img, cx, cy, active_idx, hovered_idx, dwell_frac, H, W):
    positions = picker_positions(W, H)
    pad = 18
    px0 = positions[0][0]-SWATCH_R-pad
    px1 = positions[-1][0]+SWATCH_R+pad
    py0 = positions[0][1]-SWATCH_R-pad
    py1 = positions[0][1]+SWATCH_R+pad+22

    ov = img.copy()
    cv2.rectangle(ov, (px0,py0), (px1,py1), (5,5,12), -1)
    cv2.addWeighted(ov, 0.78, img, 0.22, 0, img)
    cv2.rectangle(img, (px0,py0), (px1,py1), (40,40,60), 1, cv2.LINE_AA)

    for ci, (sx, sy) in enumerate(positions):
        name, col = PALETTE[ci]
        dim  = tuple(max(0,c//5) for c in col)
        half = tuple(max(0,c//2) for c in col)
        cv2.circle(img, (sx,sy), SWATCH_R+8, dim, -1, cv2.LINE_AA)
        cv2.circle(img, (sx,sy), SWATCH_R,   col, -1, cv2.LINE_AA)
        if ci == active_idx:
            cv2.circle(img, (sx,sy), SWATCH_R+4, (255,255,255), 2, cv2.LINE_AA)
        if ci == hovered_idx and dwell_frac > 0:
            ang = int(360*dwell_frac)
            cv2.ellipse(img, (sx,sy), (SWATCH_R+7,SWATCH_R+7),
                        -90, 0, ang, (255,255,255), 2, cv2.LINE_AA)
        cv2.putText(img, name,
                    (sx-len(name)*4, sy+SWATCH_R+15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, half, 1, cv2.LINE_AA)

    cv2.putText(img, "Point at colour and hold to select",
                (px0, py0-8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (130,130,160), 1, cv2.LINE_AA)
    draw_cursor(img, cx, cy, PALETTE[active_idx][1], brush_size, False)

# ══════════════════════════════════════════════════════
#  HUD
# ══════════════════════════════════════════════════════
def draw_hud(frame, mode, color, fps_val, undo_cnt, b_size, eraser):
    H, W = frame.shape[:2]
    BAR  = 56
    ov   = frame.copy()
    cv2.rectangle(ov, (0,H-BAR), (W,H), (4,4,10), -1)
    cv2.addWeighted(ov, 0.75, frame, 0.25, 0, frame)
    bar_col = (80,80,80) if eraser else color
    cv2.line(frame, (0,H-BAR), (W,H-BAR), bar_col, 1)
    cy_ = H-BAR//2

    if eraser:
        cv2.rectangle(frame, (14,cy_-13),(38,cy_+13),(55,55,55),-1)
        cv2.putText(frame,"E",(21,cy_+6),cv2.FONT_HERSHEY_DUPLEX,0.65,(180,180,180),1,cv2.LINE_AA)
    else:
        cv2.circle(frame,(26,cy_),18,tuple(max(0,c//5) for c in color),-1,cv2.LINE_AA)
        cv2.circle(frame,(26,cy_),13,color,-1,cv2.LINE_AA)
        cv2.circle(frame,(26,cy_),13,(255,255,255),1,cv2.LINE_AA)

    cv2.putText(frame,mode,(52,cy_+7),cv2.FONT_HERSHEY_DUPLEX,0.62,bar_col,1,cv2.LINE_AA)

    # Brush size dots
    bdx = W//2
    for bi, bs in enumerate(BRUSH_SIZES):
        bx = bdx+(bi-len(BRUSH_SIZES)//2)*18
        active = (bs==b_size)
        cv2.circle(frame,(bx,cy_), max(3,bs//3) if active else 3,
                   bar_col if active else (40,40,55), -1, cv2.LINE_AA)

    fps_col = (40,200,40) if fps_val>=25 else (40,40,220)
    cv2.putText(frame,f"{fps_val}",(W-60,cy_+6),cv2.FONT_HERSHEY_DUPLEX,0.5,fps_col,1,cv2.LINE_AA)
    cv2.putText(frame,"fps",(W-32,cy_+6),cv2.FONT_HERSHEY_SIMPLEX,0.32,(60,60,80),1,cv2.LINE_AA)

    if undo_cnt:
        cv2.putText(frame,f"↩{undo_cnt}",(W-130,cy_+6),cv2.FONT_HERSHEY_DUPLEX,0.45,(100,190,255),1,cv2.LINE_AA)

    items = [
        ("☝ DRAW",(100,180,255)),("✌ COLOUR",(100,255,160)),
        ("🖐 ERASE",(200,200,200)),("[ ] BRUSH",(255,200,80)),
        ("🤙 UNDO",(255,130,70)),("✊ CLEAR",(100,70,255)),("S SAVE",(70,70,100)),
    ]
    x = 10
    for txt, col in items:
        (tw,_),_ = cv2.getTextSize(txt,cv2.FONT_HERSHEY_SIMPLEX,0.36,1)
        cv2.putText(frame,txt,(x,22),cv2.FONT_HERSHEY_SIMPLEX,0.36,col,1,cv2.LINE_AA)
        x += tw+16
        if x>W-20: break

# ══════════════════════════════════════════════════════
#  VIGNETTE
# ══════════════════════════════════════════════════════
def get_vignette(H, W):
    global _vignette
    if _vignette is not None and _vignette.shape[:2]==(H,W):
        return _vignette
    cx,cy = W//2, H//2
    Y,X   = np.ogrid[:H,:W]
    dist  = np.sqrt(((X-cx)/cx)**2 + ((Y-cy)/cy)**2)
    mask  = np.clip(1.0-dist*0.55,0,1).astype(np.float32)
    _vignette = np.stack([mask]*3,axis=-1)
    return _vignette

# ══════════════════════════════════════════════════════
#  MAIN LOOP
# ══════════════════════════════════════════════════════
while True:
    ok, img = cap.read()
    if not ok:
        continue

    img = cv2.flip(img, 1)
    H, W = img.shape[:2]

    if canvas is None:
        canvas = np.zeros((H,W,3), dtype=np.uint8)

    rgb    = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)
    now    = time.time()

    if result.multi_hand_landmarks:
        for handLms in result.multi_hand_landmarks:
            lm      = handLms.landmark
            gesture = classify(lm)

            rx = int(lm[8].x*W); ry = int(lm[8].y*H)
            if smooth_x==0 and smooth_y==0: smooth_x,smooth_y = rx,ry
            smooth_x = int(ALPHA*rx+(1-ALPHA)*smooth_x)
            smooth_y = int(ALPHA*ry+(1-ALPHA)*smooth_y)
            cx,cy   = smooth_x, smooth_y

            if gesture != last_gesture:
                last_gesture       = gesture
                gesture_start_time = now
                prev_pt            = None
                if in_stroke: in_stroke = False

            stable = (now-gesture_start_time) >= GESTURE_HOLD

            # ── DRAW ──────────────────────────────────
            if gesture=='draw':
                eraser_mode = False
                mode_label  = f"DRAW  {PALETTE[color_idx][0]}"
                if stable:
                    if prev_pt is None:
                        prev_pt=((cx,cy)); stroke_stack.append(canvas.copy()); in_stroke=True
                    d = math.hypot(cx-prev_pt[0], cy-prev_pt[1])
                    if d>MIN_MOV:
                        neon_line(canvas, prev_pt, (cx,cy), draw_color, brush_size)
                        prev_pt=(cx,cy)
                draw_cursor(img,cx,cy,draw_color,brush_size,False)

            # ── PICK ──────────────────────────────────
            elif gesture=='pick':
                prev_pt=None; in_stroke=False; eraser_mode=False
                mode_label="COLOUR PICKER"
                positions = picker_positions(W,H)
                hov=-1
                for ci,(sx,sy) in enumerate(positions):
                    if math.hypot(cx-sx,cy-sy) < SWATCH_R+20: hov=ci; break
                if hov!=picker_hovered:
                    picker_hovered=hov; picker_dwell_start=now
                dwell_frac=0.0
                if picker_hovered>=0:
                    dwell_frac=min(1.0,(now-picker_dwell_start)/DWELL_NEEDED)
                    if dwell_frac>=1.0:
                        color_idx=picker_hovered
                        draw_color=PALETTE[color_idx][1]
                        picker_dwell_start=now
                draw_picker(img,cx,cy,color_idx,picker_hovered,dwell_frac,H,W)

            # ── ERASE ─────────────────────────────────
            elif gesture=='erase':
                eraser_mode=True; mode_label="ERASE"
                if stable:
                    if prev_pt is None:
                        prev_pt=(cx,cy); stroke_stack.append(canvas.copy()); in_stroke=True
                    d=math.hypot(cx-prev_pt[0],cy-prev_pt[1])
                    if d>MIN_MOV:
                        eraser_stroke(canvas,prev_pt,(cx,cy),28); prev_pt=(cx,cy)
                draw_cursor(img,cx,cy,draw_color,brush_size,True)

            # ── UNDO ──────────────────────────────────
            elif gesture=='undo':
                prev_pt=None; in_stroke=False; eraser_mode=False
                do_undo=False
                if stable:
                    if undo_count==0: do_undo=True; undo_count=1; undo_last_t=now
                    else:
                        wait=UNDO_FIRST if undo_count==1 else UNDO_REPEAT
                        if now-undo_last_t>=wait: do_undo=True; undo_count+=1; undo_last_t=now
                if do_undo and stroke_stack: canvas=stroke_stack.pop()
                mode_label=f"↩ UNDO {'×'+str(undo_count) if undo_count>1 else ''}"

            # ── CLEAR ─────────────────────────────────
            elif gesture=='clear':
                eraser_mode=False
                if stable and not in_stroke:
                    stroke_stack.append(canvas.copy())
                    canvas=np.zeros((H,W,3),dtype=np.uint8)
                    prev_pt=None; in_stroke=True; smooth_x=smooth_y=0
                mode_label="✊ CLEAR"

            # ── IDLE ──────────────────────────────────
            else:
                prev_pt=None; in_stroke=False; eraser_mode=False; mode_label="IDLE"

            if gesture!='undo': undo_count=0
            if gesture!='pick': picker_hovered=-1

            draw_skeleton(img,lm,H,W)

    else:
        prev_pt=None; in_stroke=False; smooth_x=smooth_y=0
        last_gesture='idle'; gesture_start_time=0.0
        undo_count=0; picker_hovered=-1; mode_label="NO HAND"

    # ── COMPOSITE ─────────────────────────────────────
    if np.any(canvas):
        bw=cv2.GaussianBlur(canvas,(35,35),0)
        bt=cv2.GaussianBlur(canvas,(9,9),0)
        glow=cv2.addWeighted(bw,0.40,bt,0.95,0)
        combined=cv2.add(img,glow)
    else:
        combined=img.copy()

    # Cinematic vignette
    combined=(combined.astype(np.float32)*get_vignette(H,W)).astype(np.uint8)

    # ── FPS & HUD ─────────────────────────────────────
    fps_times.append(now)
    fps_val=int((len(fps_times)-1)/max(fps_times[-1]-fps_times[0],1e-9)) if len(fps_times)>1 else 0
    draw_hud(combined,mode_label,draw_color,fps_val,len(stroke_stack),brush_size,eraser_mode)

    cv2.imshow("✦ Neon Air Draw v5",combined)

    key=cv2.waitKey(1)&0xFF
    if key==27: break
    elif key in (ord('s'),ord('S')):
        ts=datetime.now().strftime("%Y%m%d_%H%M%S"); fname=f"neon_{ts}.png"
        cv2.imwrite(fname,canvas)
        cv2.putText(combined,f"✓ Saved  {fname}",(W//2-180,H//2),
                    cv2.FONT_HERSHEY_DUPLEX,0.9,(60,255,130),2,cv2.LINE_AA)
        cv2.imshow("✦ Neon Air Draw v5",combined); cv2.waitKey(700)
        print(f"[SAVED] {fname}")
    elif key in (ord(']'),ord('=')):
        brush_size_idx=min(len(BRUSH_SIZES)-1,brush_size_idx+1); brush_size=BRUSH_SIZES[brush_size_idx]
    elif key in (ord('['),ord('-')):
        brush_size_idx=max(0,brush_size_idx-1); brush_size=BRUSH_SIZES[brush_size_idx]

cap.release()
cv2.destroyAllWindows()
print("Bye!")