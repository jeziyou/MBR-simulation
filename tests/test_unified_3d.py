"""
统一3D场景测试 - 验证气泡与膜组件集成效果
"""

import sys
sys.path.insert(0, '/workspace/mbr_simulation')

from visualization_v2 import MBRVisualizerV2
import plotly.graph_objects as go

def test_unified_scene():
    """测试统一3D场景"""
    print("=" * 60)
    print("统一3D场景测试")
    print("=" * 60)
    
    viz = MBRVisualizerV2()
    
    # 测试1: 基本场景
    print("\n[1] 创建基本统一场景...")
    fig = viz.create_unified_3d_scene(
        sheet_count=3,
        sheet_width=1.25,
        sheet_length=2.0,
        sheet_spacing=0.08,
        fiber_diameter=1.65,
        fiber_slack=1.5,
        show_fibers=True,
        fiber_density=15,
        bubble_count=50,
        aeration_intensity=100,
        water_depth=4.0,
        show_bubbles=True,
        show_sludge=False
    )
    
    print(f"✅ 场景创建成功")
    print(f"   轨迹数量: {len(fig.data)}")
    print(f"   标题: {fig.layout.title.text}")
    
    # 检查是否有膜组件元素
    membrane_traces = [t for t in fig.data if t.name in ['膜丝', '导轨', '集水管 top', '集水管 bottom']]
    bubble_traces = [t for t in fig.data if '气泡' in str(t.name)]
    
    print(f"\n[2] 元素检查...")
    print(f"   膜组件轨迹: {len(membrane_traces)}")
    print(f"   气泡轨迹: {len(bubble_traces)}")
    
    # 测试2: 带污泥的场景
    print("\n[3] 创建带污泥的统一场景...")
    fig2 = viz.create_unified_3d_scene(
        sheet_count=3,
        show_bubbles=True,
        show_sludge=True,
        sludge_count=50
    )
    
    sludge_traces = [t for t in fig2.data if '污泥' in str(t.name)]
    print(f"✅ 带污泥场景创建成功")
    print(f"   污泥轨迹: {len(sludge_traces)}")
    
    # 测试3: 保存HTML
    print("\n[4] 保存HTML文件...")
    fig.write_html('/workspace/mbr_simulation/tests/unified_3d_test.html')
    print("✅ HTML已保存: tests/unified_3d_test.html")
    
    # 测试4: 验证坐标范围
    print("\n[5] 坐标范围验证...")
    x_range = fig.layout.scene.xaxis.range
    y_range = fig.layout.scene.yaxis.range
    z_range = fig.layout.scene.zaxis.range
    
    print(f"   X范围: {x_range}")
    print(f"   Y范围: {y_range}")
    print(f"   Z范围: {z_range}")
    
    # 验证气泡在合理位置
    for trace in bubble_traces:
        if hasattr(trace, 'x') and len(trace.x) > 0:
            x_vals = trace.x
            y_vals = trace.y
            z_vals = trace.z
            
            print(f"\n[6] 气泡位置验证...")
            print(f"   X: [{min(x_vals):.2f}, {max(x_vals):.2f}]")
            print(f"   Y: [{min(y_vals):.2f}, {max(y_vals):.2f}]")
            print(f"   Z: [{min(z_vals):.2f}, {max(z_vals):.2f}]")
            
            # 验证气泡在膜组件附近
            assert min(x_vals) >= -1.5 and max(x_vals) <= 1.5, "X坐标超出范围"
            assert min(y_vals) >= -3 and max(y_vals) <= 3, "Y坐标超出范围"
            print("✅ 气泡位置在合理范围内")
            break
    
    print("\n" + "=" * 60)
    print("所有测试通过! ✅")
    print("=" * 60)
    
    return True

if __name__ == "__main__":
    test_unified_scene()
