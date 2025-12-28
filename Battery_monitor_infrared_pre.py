import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

try:
    from scipy.ndimage import maximum_filter

    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False


def extract_local_max_temps(image_path, min_temp, max_temp, neighborhood=3, top_n=5, roi=None, min_eval_temp=None):
    """
    从红外图像中提取温度并返回局部最高点。
    参数:
    - image_path: 红外图像路径（支持灰度或RGB图像，强度映射到温度需要线性校准）
    - min_temp, max_temp: 图像强度到温度的线性映射区间
    - neighborhood: 用于局部极大值检测的邻域大小（正方形大小）
    - top_n: 要返回的局部最高点数量
    - roi: 可选的感兴趣区域，形式为 (x, y, w, h)；仅在 ROI 内搜索
    - min_eval_temp: 最小温度阈值，低于此温度的点不被视为局部极大值
    返回: list of dicts: [{"x": int, "y": int, "temperature": float}, ...]，按温度从高到低排序
    """
    # 1) 读取并转换为灰度数组
    img = Image.open(image_path)
    if img.mode != "L":
        img_gray = img.convert("L")
    else:
        img_gray = img
    arr = np.asarray(img_gray, dtype=np.float32)  # 0-255

    # 2) 应用 ROI（如果给定）
    if roi is not None:
        x, y, w, h = roi
        x2 = int(x + w)
        y2 = int(y + h)
        arr = arr[y:y2, x:x2]

    # 3) 将像素强度映射到实际温度（线性映射：0 -> min_temp, 255 -> max_temp）
    temp_map = min_temp + (arr / 255.0) * (max_temp - min_temp)

    # 4) 找局部极大值
    # 4a) 需要 scipy 的 maximum_filter
    if SCIPY_AVAILABLE:
        footprint = np.ones((neighborhood, neighborhood), dtype=bool)
        local_max_mask = (temp_map == maximum_filter(temp_map, footprint=footprint))
    else:
        # 简单的兜底实现：逐像素比较周围像素（性能较差，仅用于无 scipy 环境）
        pad = neighborhood // 2
        padded = np.pad(temp_map, pad, mode='edge')
        local_max_mask = np.zeros_like(temp_map, dtype=bool)
        H, W = temp_map.shape
        for i in range(H):
            for j in range(W):
                neighborhood_vals = padded[i:i + neighborhood, j:j + neighborhood]
                if temp_map[i, j] >= neighborhood_vals.max():
                    local_max_mask[i, j] = True

    # 4b) 应用阈值（如给定 min_eval_temp）
    if min_eval_temp is not None:
        local_max_mask = local_max_mask & (temp_map >= min_eval_temp)

    # 5) 提取坐标和温度
    coords = np.argwhere(local_max_mask)  # 每个元素为 (row, col) = (y, x)
    temps = temp_map[local_max_mask]

    # 6) 组合成结果并排序
    results = []
    for idx, (y, x) in enumerate(coords):
        t = float(temps[idx])
        # 如果 ROI 被应用，外部坐标需要回推到原始图像坐标系
        if roi is not None:
            x0, y0, _, _ = roi
            x += x0
            y += y0
        results.append({"x": int(x), "y": int(y), "temperature": t})

    # 按温度降序排序，返回前 top_n 个
    results.sort(key=lambda r: r["temperature"], reverse=True)
    return results[:int(top_n)]


def compute_temp_map(image_path, min_temp, max_temp, roi=None):
    """
    将 IR 图像映射到温度矩阵（基于线性映射：0-255 -> min_temp~max_temp）。
    如果提供 roi=(x,y,w,h)，则仅返回 ROI 内的温度矩阵及偏移信息。
    返回：temp_map, (x0, y0) 的原图偏移
    """
    img = Image.open(image_path)
    if img.mode != "L":
        img_gray = img.convert("L")
    else:
        img_gray = img
    arr = np.asarray(img_gray, dtype=np.float32)  # 0-255

    x_offset, y_offset = 0, 0
    if roi is not None:
        x, y, w, h = roi
        x2 = int(x + w)
        y2 = int(y + h)
        arr = arr[y:y2, x:x2]
        x_offset, y_offset = int(x), int(y)

    temp_map = min_temp + (arr / 255.0) * (max_temp - min_temp)
    return temp_map, (x_offset, y_offset)


def extract_local_max_temps_multizone(image_path, min_temp, max_temp, grid_size=(3, 3),
                                      neighborhood=3, top_n_per_zone=3, min_eval_temp=None):
    """
    将图像平均拆分成多个区域，然后对每个区域进行局部高温点分析。
    参数:
    - image_path: 红外图像路径
    - min_temp, max_temp: 温度映射参数
    - grid_size: 网格划分 (rows, cols)，如 (3,3) 表示 3x3 网格
    - neighborhood: 局部极大值检测的邻域大小
    - top_n_per_zone: 每个区域返回的最高点数量
    - min_eval_temp: 最小温度阈值
    返回: list of dicts: 所有区域的局部最高点，按温度从高到低排序
    """
    # 读取图像获取尺寸
    img = Image.open(image_path)
    if img.mode != "L":
        img_gray = img.convert("L")
    else:
        img_gray = img
    arr = np.asarray(img_gray, dtype=np.float32)
    height, width = arr.shape

    rows, cols = grid_size
    zone_height = height // rows
    zone_width = width // cols

    all_points = []

    print(f"[INFO] 将图像划分为 {rows}x{cols} 网格，每个区域尺寸: {zone_width}x{zone_height}")

    for i in range(rows):
        for j in range(cols):
            # 计算当前区域的 ROI
            x = j * zone_width
            y = i * zone_height
            w = zone_width if j < cols - 1 else width - x  # 最后一个区域处理边界
            h = zone_height if i < rows - 1 else height - y  # 最后一个区域处理边界

            roi = (x, y, w, h)

            # 对当前区域提取局部最高点
            zone_points = extract_local_max_temps(
                image_path=image_path,
                min_temp=min_temp,
                max_temp=max_temp,
                neighborhood=neighborhood,
                top_n=top_n_per_zone,
                roi=roi,
                min_eval_temp=min_eval_temp
            )

            # 为每个点添加区域信息
            for point in zone_points:
                point["zone"] = f"({i},{j})"
                point["zone_coords"] = (i, j)

            all_points.extend(zone_points)
            print(f"[INFO] 区域 ({i},{j}) 找到 {len(zone_points)} 个高温点")

    # 按温度降序排序所有点
    all_points.sort(key=lambda r: r["temperature"], reverse=True)
    return all_points


def visualize_local_max_temps_multizone(image_path, min_temp, max_temp, top_points,
                                        grid_size=(3, 3), save_path=None, show=True):
    """
    可视化多区域分析结果
    """
    # 读取图像
    img = Image.open(image_path)
    if img.mode != "L":
        img_gray = img.convert("L")
    else:
        img_gray = img
    arr = np.asarray(img_gray, dtype=np.float32)
    height, width = arr.shape

    rows, cols = grid_size
    zone_height = height // rows
    zone_width = width // cols

    # 创建可视化
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    # 子图1：温度热力图
    temp_map, _ = compute_temp_map(image_path, min_temp, max_temp)
    im1 = ax1.imshow(temp_map, cmap='hot', origin='upper')
    plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    ax1.set_title("Temperature Map with Multi-Zone Analysis", fontsize=20)

    # 子图2：原始图像与标记
    ax2.imshow(img_gray, cmap='gray', origin='upper')
    ax2.set_title("Infrared Image with Zone Boundaries", fontsize=20)

    # 绘制网格边界
    for i in range(1, rows):
        y = i * zone_height
        ax1.axhline(y=y, color='cyan', linestyle='--', alpha=0.7, linewidth=1)
        ax2.axhline(y=y, color='cyan', linestyle='--', alpha=0.7, linewidth=1)

    for j in range(1, cols):
        x = j * zone_width
        ax1.axvline(x=x, color='cyan', linestyle='--', alpha=0.7, linewidth=1)
        ax2.axvline(x=x, color='cyan', linestyle='--', alpha=0.7, linewidth=1)

    # 标记高温点
    colors = plt.cm.Set3(np.linspace(0, 1, rows * cols))

    for point in top_points:
        x, y = point["x"], point["y"]
        temp = point["temperature"]
        zone_i, zone_j = point["zone_coords"]
        color_idx = zone_i * cols + zone_j

        # 在热力图上标记
        ax1.plot(x, y, 'o', markersize=8, color=colors[color_idx],
                 markeredgecolor='white', markeredgewidth=1)

        # 在原始图像上标记
        ax2.plot(x, y, 'o', markersize=8, color=colors[color_idx],
                 markeredgecolor='white', markeredgewidth=1)
        ax2.annotate(f"{temp:.1f}°C", (x, y), color='white', fontsize=12,
                     textcoords="offset points", xytext=(0, -15), ha='center',
                     bbox=dict(boxstyle="round,pad=0.2", facecolor=colors[color_idx], alpha=0.8))

    # 添加图例
    from matplotlib.patches import Patch
    legend_elements = []
    for i in range(rows):
        for j in range(cols):
            color_idx = i * cols + j
            legend_elements.append(
                Patch(facecolor=colors[color_idx], label=f'Zone ({i},{j})')
            )

    ax2.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.4, 1.0), prop={'size': 12})
    ax1.axis('off')
    ax2.axis('off')
    #plt.legend(handles=legend_elements, prop={'size': 10})
    plt.tight_layout()

    # 保存或显示
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"[INFO] 多区域分析可视化已保存至: {save_path}")
    if show:
        plt.show()
    plt.close()

    return top_points


def print_zone_statistics(top_points, grid_size):
    """
    打印每个区域的统计信息
    """
    rows, cols = grid_size
    zone_stats = {}

    for i in range(rows):
        for j in range(cols):
            zone_key = (i, j)
            zone_points = [p for p in top_points if p["zone_coords"] == zone_key]

            if zone_points:
                max_temp = max(p["temperature"] for p in zone_points)
                avg_temp = np.mean([p["temperature"] for p in zone_points])
                count = len(zone_points)
            else:
                max_temp = avg_temp = 0
                count = 0

            zone_stats[zone_key] = {
                "count": count,
                "max_temp": max_temp,
                "avg_temp": avg_temp
            }

    print("\n" + "=" * 50)
    print("各区域统计信息:")
    print("=" * 50)
    for (i, j), stats in zone_stats.items():
        print(
            f"区域 ({i},{j}): {stats['count']}个点, 最高温: {stats['max_temp']:.2f}°C, 平均温: {stats['avg_temp']:.2f}°C")


# 使用示例
if __name__ == "__main__":
    image_path = "thermal_crop_0_0.jpg"
    min_temp = 20  # 根据你的标定设定
    max_temp = 140

    # 多区域分析
    grid_size = (5, 5)  # 3x3 网格
    top_points = extract_local_max_temps_multizone(
        image_path=image_path,
        min_temp=min_temp,
        max_temp=max_temp,
        grid_size=grid_size,
        neighborhood=5,
        top_n_per_zone=3,  # 每个区域找3个最高点
        min_eval_temp=None
    )

    # 打印结果
    print("\n所有区域的高温点（按温度排序）:")
    print("=" * 60)
    for i, p in enumerate(top_points):
        print(f"{i + 1:2d}. 区域{p['zone']}: x={p['x']:3d}, y={p['y']:3d}, temp={p['temperature']:.2f}°C")

    # 打印区域统计
    print_zone_statistics(top_points, grid_size)

    # 可视化
    visualize_local_max_temps_multizone(
        image_path=image_path,
        min_temp=min_temp,
        max_temp=max_temp,
        top_points=top_points,
        grid_size=grid_size,
        save_path="/home/lab-server/Projects/JMS2026/infrared_multizone_analysis.png",
        show=True
    )