import matplotlib.pyplot as plt
from shapely.geometry import box, MultiPolygon
from shapely.ops import unary_union


def is_overlapping(rect1, rect2):
    """
    检查两个矩形框是否重叠
    """
    return rect1.intersects(rect2)


def merge_overlapping_rectangles(rectangles):
    """
    合并所有重叠的矩形框
    """
    merged = unary_union(rectangles)
    return [merged]


def visualize_rectangles(rectangles, title):
    """
    可视化矩形框
    """
    fig, ax = plt.subplots()
    for rect in rectangles:
        if isinstance(rect, MultiPolygon):
            for poly in rect.geoms:
                x, y = poly.exterior.xy
                ax.fill(x, y, alpha=0.5, fc='orange')
        else:
            x, y = rect.exterior.xy
            ax.fill(x, y, alpha=0.5, fc='orange')
    # ax.set_xlim(-1, 8)
    # ax.set_ylim(-1, 8)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.show()


def get_vertices(rectangles):
    """
    从 Polygon 或 MultiPolygon 对象中获取所有顶点的坐标
    """
    vertices = []
    for rect in rectangles:
        if isinstance(rect, MultiPolygon):
            for poly in rect.geoms:
                # 获取每个 Polygon 的顶点坐标
                vertices.append([list(i) for i in list(poly.exterior.coords)])
        else:
            # 获取当前 Polygon 的顶点坐标
            vertices.append(list(rect.exterior.coords))
    return vertices


def main():
    # 定义矩形框的坐标列表
    rectangles = [
        box(0, 0, 2, 2),  # 矩形1: [x1, y1, x2, y2] -> [0, 0, 2, 2]
        box(1, 1, 3, 3),  # 矩形2: [1, 1, 3, 3]
        box(4, 4, 6, 6),  # 矩形3: [4, 4, 6, 6]
        box(5, 5, 7, 7)   # 矩形4: [5, 5, 7, 7]
    ]

    # 可视化原始矩形框
    visualize_rectangles(rectangles, "Original Rectangles")

    # 合并所有重叠的矩形框
    merged_rectangles = merge_overlapping_rectangles(rectangles)

    # 可视化合并后的矩形框
    visualize_rectangles(merged_rectangles, "Merged Rectangles")

    # 获取合并后的矩形框的顶点坐标
    vertices = get_vertices(merged_rectangles)

    # 打印顶点坐标
    print("顶点坐标:")
    for vertex in vertices:
        print(vertex)


if __name__ == "__main__":
    main()
