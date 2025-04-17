from pathlib import Path

# 업데이트된 Streamlit 코드: 객체가 좌표 바뀌었을 때 3초마다 Webex 알림 전송

import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image, ImageDraw
import time
from datetime import datetime
import os
from webex_utils import send_webex_message
import winsound

st.set_page_config(page_title="🦜 버드 스트라이크 감지 시스템 🦌", layout="centered")
st.title("🦜 헤헷 : 동물 충돌 사고 예방 시스템 🦌")
st.markdown("For people, For Animals")

FRAME_WINDOW = st.image([])
log_box = st.empty()

model = YOLO("best_hehev1.pt")
##모델 
# best_2. pt : 최초 개발 모델
# hehebest.pt : 두번째 개발 모델 __ 조류 데이터 추가 학습
# best_hehev1.pt : 최종 개발 모델 _ 조류, 멧돼지, 고라니 데이터 학습


cap = cv2.VideoCapture(0)
if not cap.isOpened():
    st.error("❌ 웹캠을 열 수 없습니다.")
    st.stop()

class_names = {0: "bird", 1: "wildboar", 2: "deer"}
target_class_ids = list(class_names.keys())
os.makedirs("detections", exist_ok=True)
tracked_objects = {}
global_object_id_counter = 1
log_texts = []

while True:
    ret, frame = cap.read()
    if not ret:
        st.error("❌ 프레임을 가져올 수 없습니다.")
        break

    results = model.track(source=frame, conf=0.5, imgsz=640, show=False, verbose=False)
    current_time = time.time()

    frame_height, frame_width = frame.shape[:2]
    danger_zone = (0, 0, frame_width // 3, frame_height // 3)
    danger_color = (0, 0, 255)

    pil_im = Image.fromarray(frame)
    draw = ImageDraw.Draw(pil_im)

    for result in results:
        if result.boxes is not None and len(result.boxes) > 0:
            for box in result.boxes:
                class_id = int(box.cls)
                conf = float(box.conf)
                track_id = int(box.id) if hasattr(box, "id") else None
                if track_id is None or class_id not in target_class_ids:
                    continue

                label = class_names[class_id]
                color = (0, 255, 0) if class_id == 0 else (255, 0, 0) if class_id == 1 else (0, 0, 255)

                xyxy = box.xyxy.cpu().numpy().astype(int).flatten()
                if len(xyxy) != 4:
                    continue
                x1, y1, x2, y2 = xyxy
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2

                draw.rectangle([(x1, y1), (x2, y2)], outline=color, width=2)
                draw.text((x1, y1 - 25), f"{label} {conf:.2f} ID:{track_id}", fill=color)

                # 위험구역 침입 체크
                if (danger_zone[0] <= center_x <= danger_zone[2] and
                        danger_zone[1] <= center_y <= danger_zone[3]):
                    danger_log = f"🚨 [위험] {label} (ID:{track_id}) 위험구역 접근!\n 좌표: ({center_x}, {center_y})"+"\n"
                    log_texts.append(danger_log)
                    draw.text((x1, y2 + 10), "[WARNING] Danger zone!", fill=danger_color)
                    send_webex_message(danger_log)
                    winsound.Beep(1000, 500)

                # 새로운 객체 감지
                if track_id not in tracked_objects:
                    log = f"🟢 [알림] {label} 감지됨 (ID {track_id})\n📍 좌표: ({center_x}, {center_y})"+"\n"
                    log_texts.append(log)
                    send_webex_message(log)

                    image_filename = f"detections/{label}_{int(current_time)}.jpg"
                    detected_region = frame[y1:y2, x1:x2]
                    cv2.imwrite(image_filename, detected_region)

                    tracked_objects[track_id] = {
                        "object_id": global_object_id_counter,
                        "last_move_call": current_time,
                        "last_seen": current_time,
                        "last_reported_location": (center_x, center_y),
                        "move_paths": []
                    }
                    global_object_id_counter += 1

                    tracked_objects[track_id]["move_paths"].append({
                        "CAPTURED_TIME": datetime.now().isoformat(),
                        "LOCATION_X": float(center_x),
                        "LOCATION_Y": float(center_y)
                    })
                    tracked_objects[track_id]["last_move_call"] = current_time

                else:
                    tracked = tracked_objects[track_id]
                    tracked["last_seen"] = current_time
                    tracked["move_paths"].append({
                        "CAPTURED_TIME": datetime.now().isoformat(),
                        "LOCATION_X": float(center_x),
                        "LOCATION_Y": float(center_y)
                    })

                    # 위치가 바뀌었고, 3초 지났을 때 Webex에 위치 보고
                    last_x, last_y = tracked["last_reported_location"]
                    moved = (center_x != last_x or center_y != last_y)
                    if moved and (current_time - tracked["last_move_call"] >= 3.0):
                        moved_log = f"🔄 [이동] {label} (ID {track_id}) 위치 이동\n📍 현재 좌표: ({center_x}, {center_y})"+"\n"
                        log_texts.append(moved_log)
                        send_webex_message(moved_log)

                        tracked["last_move_call"] = current_time
                        tracked["last_reported_location"] = (center_x, center_y)
                        tracked["move_paths"] = []

    frame = np.array(pil_im)
    cv2.rectangle(frame, (danger_zone[0], danger_zone[1]), (danger_zone[2], danger_zone[3]), danger_color, 2)
    cv2.putText(frame, "DANGER ZONE", (danger_zone[0] + 5, danger_zone[1] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, danger_color, 2)

    FRAME_WINDOW.image(frame[:, :, ::-1])

    lost_ids = [tid for tid, info in tracked_objects.items() if current_time - info["last_seen"] > 1]
    for tid in lost_ids:
        log_texts.append(f"⚪ [종료] track id {tid} 추적 종료\n\n\n")
        del tracked_objects[tid]

    if len(log_texts) > 20:
        log_texts = log_texts[-20:]

    log_box.markdown("#### 📋 감지 로그\n" + "\n".join(log_texts))

    time.sleep(0.1)


