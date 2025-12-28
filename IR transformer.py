import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import os
import time

# ================= 全局配置 =================
BRUSH_MAX_TEMP = 200  # 统一最高温度

paint_state = {
    'layer': None,
    'drawing': False,
    'brush_size': 20,
    'brush_temp': 0.5,
    'action': 'none',
    'app_mode': 'paint',
    'crop_start': None,
    'crop_curr': None
}


def mouse_callback(event, x, y, flags, param):
    global paint_state
    h, w = paint_state['layer'].shape

    # === 绘画模式 ===
    if paint_state['app_mode'] == 'paint':
        if event == cv2.EVENT_LBUTTONDOWN:
            paint_state['drawing'] = True
            paint_state['action'] = 'paint'
        elif event == cv2.EVENT_RBUTTONDOWN:
            paint_state['drawing'] = True
            paint_state['action'] = 'erase'
        elif event == cv2.EVENT_LBUTTONUP or event == cv2.EVENT_RBUTTONUP:
            paint_state['drawing'] = False
            paint_state['action'] = 'none'

        if event == cv2.EVENT_MOUSEMOVE and paint_state['drawing']:
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.circle(mask, (x, y), paint_state['brush_size'], 255, -1)
            if paint_state['action'] == 'paint':
                paint_state['layer'][mask == 255] = paint_state['brush_temp']
            elif paint_state['action'] == 'erase':
                paint_state['layer'][mask == 255] = -999.0

    # === 截图模式 ===
    elif paint_state['app_mode'] == 'crop':
        if event == cv2.EVENT_LBUTTONDOWN:
            paint_state['drawing'] = True
            paint_state['crop_start'] = (x, y)
            paint_state['crop_curr'] = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and paint_state['drawing']:
            paint_state['crop_curr'] = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            paint_state['drawing'] = False
            paint_state['crop_curr'] = (x, y)


def generate_temperature_map(gray_image, mode='intensity'):
    h, w = gray_image.shape
    norm_gray = gray_image / 255.0
    if mode == 'intensity':
        return norm_gray
    elif mode == 'inverse_intensity':
        return 1.0 - norm_gray
    elif mode == 'radial':
        Y, X = np.ogrid[:h, :w]
        center_y, center_x = h / 2, w / 2
        dist = np.sqrt((X - center_x) ** 2 + (Y - center_y) ** 2)
        max_dist = np.sqrt((h / 2) ** 2 + (w / 2) ** 2)
        return np.clip(1.0 - (dist / max_dist), 0, 1)
    elif mode == 'linear_gradient':
        return np.tile(np.linspace(1, 0, h).reshape(-1, 1), (1, w))
    return norm_gray


def add_colorbar_sidebar(image, lut_bgr, min_temp, max_temp):
    """生成侧边栏，自动适应截图的高度"""
    h, w = image.shape[:2]

    sidebar_width = 400
    if h < 300: sidebar_width = 350

    gradient_width = 100
    margin_top = 100
    margin_bottom = 60
    margin_left = 20

    font_scale = 3
    if h < 400: font_scale = 2
    thickness = 6 if h > 400 else 4

    if h < (margin_top + margin_bottom + 50):
        margin_top = 10
        margin_bottom = 10

    grad_height = h - margin_top - margin_bottom
    if grad_height < 20: grad_height = 20

    sidebar = np.full((h, sidebar_width, 3), 255, dtype=np.uint8)

    gradient = np.linspace(255, 0, grad_height).astype(np.uint8)
    gradient = np.tile(gradient.reshape(-1, 1), (1, gradient_width))
    gradient_bgr = cv2.cvtColor(gradient, cv2.COLOR_GRAY2BGR)
    gradient_colored = cv2.LUT(gradient_bgr, lut_bgr)

    end_y = min(margin_top + grad_height, h)
    sidebar[margin_top:end_y, margin_left:margin_left + gradient_width] = gradient_colored[0:end_y - margin_top, :]

    cv2.rectangle(sidebar, (margin_left, margin_top),
                  (margin_left + gradient_width, end_y), (0, 0, 0), 2)

    color = (0, 0, 0)
    line_start_x = margin_left + gradient_width
    line_end_x = line_start_x + 15
    text_x = line_end_x + 10
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Top
    cv2.line(sidebar, (line_start_x, margin_top), (line_end_x, margin_top), color, thickness)
    cv2.putText(sidebar, f"{max_temp}C", (text_x, margin_top + 20), font, font_scale, color, thickness)

    # Bottom
    cv2.line(sidebar, (line_start_x, end_y), (line_end_x, end_y), color, thickness)
    cv2.putText(sidebar, f"{min_temp}C", (text_x, end_y), font, font_scale, color, thickness)

    return cv2.hconcat([image, sidebar])


def interactive_thermal_painter(image_path):
    global paint_state

    rgb_image = cv2.imread(image_path)
    if rgb_image is None:
        print("Error: Image not found")
        return

    gray = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)
    texture_detail_map = (gray / 255.0) - 0.5
    h, w = gray.shape
    paint_state['layer'] = np.full((h, w), -999.0, dtype=np.float32)

    colors = ['black', 'darkred', 'red', 'yellow', 'white']
    mpl_cmap = LinearSegmentedColormap.from_list('thermal', colors, N=256)
    lut_data = (mpl_cmap(np.linspace(0, 1, 256))[:, :3] * 255).astype(np.uint8)
    lut_bgr = np.zeros((256, 1, 3), dtype=np.uint8)
    lut_bgr[:, 0, 0], lut_bgr[:, 0, 1], lut_bgr[:, 0, 2] = lut_data[:, 2], lut_data[:, 1], lut_data[:, 0]

    window_name = "Thermal Studio (High Contrast UI)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 900, 600)
    cv2.setMouseCallback(window_name, mouse_callback)

    def nothing(x):
        pass

    # === 控制面板 ===
    cv2.createTrackbar('Mode', window_name, 0, 3, nothing)
    cv2.createTrackbar('Brush Temp', window_name, 40, BRUSH_MAX_TEMP, nothing)
    cv2.createTrackbar('Brush Size', window_name, 30, 150, nothing)
    cv2.createTrackbar('Texture %', window_name, 30, 100, nothing)
    cv2.createTrackbar('Range Min', window_name, 20, BRUSH_MAX_TEMP - 10, nothing)

    # === 新增：显示标尺开关 (0=Off, 1=On) ===
    cv2.createTrackbar('Show Scale', window_name, 1, 1, nothing)

    msg_timer = 0
    msg_text = ""

    print(f"=== 系统就绪 ===")
    print("1. 绘画模式: 左键涂抹，右键擦除，'S' 保存全图")
    print("2. 截图模式: 按 'C' 进入/退出。拖拽选框，按 'Enter' 保存选中区域")
    print("3. 标尺控制: 调节 'Show Scale' 选择保存时是否附带温度色标")

    while True:
        # 获取参数
        mode_idx = cv2.getTrackbarPos('Mode', window_name)
        brush_temp_c = cv2.getTrackbarPos('Brush Temp', window_name)
        brush_size_val = cv2.getTrackbarPos('Brush Size', window_name)
        texture_strength = cv2.getTrackbarPos('Texture %', window_name) / 100.0
        min_t = cv2.getTrackbarPos('Range Min', window_name)
        show_scale = cv2.getTrackbarPos('Show Scale', window_name)  # 获取开关状态

        max_t = BRUSH_MAX_TEMP
        if min_t >= max_t: min_t = max_t - 10

        denom = max_t - min_t
        brush_norm_val = (brush_temp_c - min_t) / denom
        paint_state['brush_temp'] = brush_norm_val
        paint_state['brush_size'] = max(1, brush_size_val)

        # 图像计算
        modes = ['intensity', 'inverse_intensity', 'radial', 'linear_gradient']
        base_map = generate_temperature_map(gray, modes[mode_idx])
        detail_layer = texture_detail_map * texture_strength * (100.0 / denom)
        painted_region_map = paint_state['layer'] + detail_layer
        final_map = np.where(paint_state['layer'] != -999.0, painted_region_map, base_map)
        idx_map_bgr = cv2.cvtColor((np.clip(final_map, 0, 1) * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        clean_thermal_view = cv2.LUT(idx_map_bgr, lut_bgr)

        ui_view = clean_thermal_view.copy()

        # UI 绘制
        if paint_state['app_mode'] == 'paint':
            display_color_idx = int(np.clip(brush_norm_val, 0, 1) * 255)
            brush_color = lut_bgr[display_color_idx, 0].tolist()
            cv2.circle(ui_view, (30, 90), 15, [int(c) for c in brush_color], -1)
            cv2.circle(ui_view, (30, 90), 17, (255, 255, 255), 1)
            cv2.putText(ui_view, f"Brush: {brush_temp_c}C", (55, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            scale_status = "ON" if show_scale == 1 else "OFF"
            info_str = f"Scale: {min_t}-{max_t}C | Bar: {scale_status} | [C] Crop"

        elif paint_state['app_mode'] == 'crop':
            scale_status = "ON" if show_scale == 1 else "OFF"
            info_str = f"MODE: CROP | Bar: {scale_status} | Drag -> ENTER to Save"

            if paint_state['crop_start'] and paint_state['crop_curr']:
                p1 = paint_state['crop_start']
                p2 = paint_state['crop_curr']
                # 双层高对比度选框
                cv2.rectangle(ui_view, p1, p2, (0, 0, 0), 4)
                cv2.rectangle(ui_view, p1, p2, (255, 255, 0), 2)

        cv2.rectangle(ui_view, (0, 0), (600, 50), (0, 0, 0), -1)
        cv2.putText(ui_view, info_str, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

        if msg_timer > 0:
            cv2.putText(ui_view, msg_text, (w // 2 - 200, h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
            msg_timer -= 1

        cv2.imshow(window_name, ui_view)

        key = cv2.waitKey(20) & 0xFF
        if key == ord('q') or key == 27:
            break
        elif key == ord('r'):
            paint_state['layer'][:] = -999.0
            msg_text = "Canvas Reset"
            msg_timer = 20
        elif key == ord('c'):
            if paint_state['app_mode'] == 'paint':
                paint_state['app_mode'] = 'crop'
                paint_state['crop_start'] = None
                paint_state['crop_curr'] = None
            else:
                paint_state['app_mode'] = 'paint'

        # === 保存逻辑 (根据开关决定是否加 Scale) ===
        elif key == ord('s') and paint_state['app_mode'] == 'paint':
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"thermal_full_{timestamp}.jpg"

            # 根据 Show Scale 决定保存内容
            if show_scale == 1:
                save_img = add_colorbar_sidebar(clean_thermal_view, lut_bgr, min_t, max_t)
            else:
                save_img = clean_thermal_view  # 仅保存纯净热图

            cv2.imwrite(filename, save_img)
            msg_text = "Full Image Saved!"
            msg_timer = 30
            print(f"保存: {filename} (Scale: {show_scale})")

        elif (key == 13 or key == 32) and paint_state['app_mode'] == 'crop':
            if paint_state['crop_start'] and paint_state['crop_curr']:
                x1, y1 = paint_state['crop_start']
                x2, y2 = paint_state['crop_curr']
                xmin, xmax = min(x1, x2), max(x1, x2)
                ymin, ymax = min(y1, y2), max(y1, y2)
                if (xmax - xmin > 10) and (ymax - ymin > 10):
                    crop_img = clean_thermal_view[ymin:ymax, xmin:xmax]
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    filename = f"thermal_crop_{timestamp}.jpg"

                    # 根据 Show Scale 决定保存内容
                    if show_scale == 1:
                        save_img = add_colorbar_sidebar(crop_img, lut_bgr, min_t, max_t)
                    else:
                        save_img = crop_img  # 仅保存纯净截图

                    cv2.imwrite(filename, save_img)
                    msg_text = "Crop Saved!"
                    msg_timer = 30
                    print(f"截图保存: {filename} (Scale: {show_scale})")
                    paint_state['crop_start'] = None

    cv2.destroyAllWindows()


if __name__ == "__main__":
    img_path = "/home/lab-server/Projects/JMS2026/Battery cells.png"
    if os.path.exists(img_path):
        interactive_thermal_painter(img_path)
    else:
        print("路径错误")