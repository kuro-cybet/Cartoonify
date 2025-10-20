import cv2
import numpy as np

print("🎨 Ghibli Cartoon Camera Started")
print("➡️ Press 'c' to toggle Ghibli mode ON/OFF.")
print("➡️ Press 's' to save a snapshot.")
print("➡️ Press 'q' to quit.")

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Webcam not detected!")
    exit()

ghibli_mode = True

def ghibli_style(frame):
    # 1. Edge-preserving smoothing (soft painterly look)
    smooth = cv2.edgePreservingFilter(frame, flags=1, sigma_s=80, sigma_r=0.5)

    # 2. Mild color quantization for painterly effect
    Z = smooth.reshape((-1,3))
    Z = np.float32(Z)
    K = 16  # number of colors
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    center = np.uint8(center)
    res = center[label.flatten()]
    quantized = res.reshape((smooth.shape))

    # 3. Boost saturation and brightness slightly
    hsv = cv2.cvtColor(quantized, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    s = cv2.add(s, 30)
    v = cv2.add(v, 15)
    hsv = cv2.merge([h, s, v])
    colored = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # 4. Optional light edge detection for outlines
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.medianBlur(gray, 7)
    edges = cv2.adaptiveThreshold(blur, 255,
                                  cv2.ADAPTIVE_THRESH_MEAN_C,
                                  cv2.THRESH_BINARY, 9, 5)
    edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    edges_colored = cv2.bitwise_not(edges_colored)
    edges_colored = cv2.cvtColor(edges_colored, cv2.COLOR_BGR2GRAY)
    edges_colored = cv2.threshold(edges_colored, 100, 255, cv2.THRESH_BINARY)[1]
    edges_colored = cv2.cvtColor(edges_colored, cv2.COLOR_GRAY2BGR)
    edges_colored = cv2.bitwise_not(edges_colored)
    # Blend softly
    ghibli = cv2.addWeighted(colored, 0.9, edges_colored, 0.1, 0)

    return ghibli

saved_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    display = frame.copy()

    if ghibli_mode:
        display = ghibli_style(frame)
        cv2.putText(display, "🎨 Ghibli Mode (Press 'c' to toggle)",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    else:
        cv2.putText(display, "📷 Normal Mode (Press 'c' to toggle)",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow("Ghibli Cartoon Camera", display)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('c'):
        ghibli_mode = not ghibli_mode
        mode = "Ghibli" if ghibli_mode else "Normal"
        print(f"🔁 Switched to {mode} mode")
    elif key == ord('s') and ghibli_mode:
        saved_count += 1
        filename = f"ghibli_snapshot_{saved_count}.jpg"
        cv2.imwrite(filename, display)
        print(f"✅ Saved snapshot as '{filename}'")

cap.release()
cv2.destroyAllWindows()
