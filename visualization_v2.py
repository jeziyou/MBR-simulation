"""
MBR仿真系统 - 增强版3D可视化组件 V2.0
MBR Simulation System - Enhanced 3D Visualization V2.0

增强功能：
- 真实材质和光照效果
- 粒子系统（气泡、污泥）
- 动态动画
- CFD结果可视化
- 热力图
- 交互式控制

作者: MBR Engineering Team
版本: 2.0
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from typing import List, Dict, Tuple, Optional
import colorsys


class MBRVisualizerV2:
    """增强版MBR系统3D可视化器"""
    
    def __init__(self, config=None):
        self.config = config
        self.colors = {
            'primary': '#00f2ff',
            'secondary': '#0088ff',
            'success': '#00ff88',
            'warn': '#ff9900',
            'danger': '#ff4444',
            'sludge': '#d4a84b',
            'sludge_dark': '#8b6914',
            'fiber': '#e0e8f0',
            'fiber_glow': '#ffffff',
            'bubble': '#aaddff',
            'bubble_glow': '#ffffff',
            'water': '#0a1a2e',
            'water_surface': 'rgba(10, 26, 46, 0.3)',
            'pipe': '#445566',
            'pipe_highlight': '#667788',
            'header': '#F5DEB3',
            'tank_frame': 'rgba(30, 74, 133, 0.5)',
            'background': '#01050a'
        }
    
    def create_membrane_structure_enhanced(self,
                                           sheet_count: int = 5,
                                           sheet_width: float = 1.25,
                                           sheet_length: float = 2.0,
                                           sheet_spacing: float = 0.08,
                                           fiber_diameter: float = 1.65,
                                           fiber_slack: float = 1.5,
                                           show_fibers: bool = True,
                                           fiber_density: int = 25,
                                           lighting: bool = True) -> go.Figure:
        """
        创建增强版MBR膜组件3D结构
        
        包含真实材质、光照、阴影效果
        """
        fig = go.Figure()
        
        half_length = sheet_length / 2
        half_width = sheet_width / 2
        
        # 光照设置
        if lighting:
            lighting_settings = dict(
                ambient=0.6,
                diffuse=0.8,
                roughness=0.4,
                specular=0.5,
                fresnel=0.2
            )
        else:
            lighting_settings = dict()
        
        # 绘制膜片框架和膜丝
        for i in range(sheet_count):
            z = (i - (sheet_count - 1) / 2) * sheet_spacing
            
            # 上下集水管 - 使用圆柱体效果
            self._add_enhanced_header(fig, sheet_width, z, half_length, 'top', lighting_settings)
            self._add_enhanced_header(fig, sheet_width, z, -half_length, 'bottom', lighting_settings)
            
            # 导轨
            self._add_enhanced_rails(fig, sheet_length, sheet_width, z)
            
            # 膜丝 - 增强版
            if show_fibers:
                self._add_enhanced_fibers(fig, sheet_width, sheet_length, z, 
                                         fiber_density, fiber_slack)
        
        # 曝气管 - 增强版
        self._add_enhanced_aeration_pipes(fig, sheet_count, sheet_spacing, 
                                         sheet_width, half_length)
        
        # 水箱框架
        self._add_enhanced_tank_frame(fig, sheet_width, sheet_length, 
                                     sheet_count, sheet_spacing)
        
        # 水面效果
        self._add_enhanced_water_surface(fig, sheet_width, sheet_length,
                                        sheet_count, sheet_spacing)
        
        # 设置布局
        fig.update_layout(
            scene=dict(
                xaxis=dict(
                    title='X (m)', 
                    range=[-1.5, 1.5],
                    backgroundcolor=self.colors['background'],
                    gridcolor='rgba(0, 242, 255, 0.08)',
                    showbackground=True,
                    zerolinecolor='rgba(0, 242, 255, 0.2)'
                ),
                yaxis=dict(
                    title='Y (高度 m)', 
                    range=[-3, 3],
                    backgroundcolor=self.colors['background'],
                    gridcolor='rgba(0, 242, 255, 0.08)',
                    showbackground=True,
                    zerolinecolor='rgba(0, 242, 255, 0.2)'
                ),
                zaxis=dict(
                    title='Z (m)', 
                    range=[-0.5, 0.5],
                    backgroundcolor=self.colors['background'],
                    gridcolor='rgba(0, 242, 255, 0.08)',
                    showbackground=True,
                    zerolinecolor='rgba(0, 242, 255, 0.2)'
                ),
                aspectmode='manual',
                aspectratio=dict(x=1.5, y=3, z=0.5),
                camera=dict(
                    eye=dict(x=2.5, y=1.5, z=1.5),
                    center=dict(x=0, y=0, z=0),
                    up=dict(x=0, y=0, z=1)
                ),
                bgcolor=self.colors['background']
            ),
            title=dict(
                text='<b>MBR膜组件3D结构</b><br><sup>增强版可视化</sup>',
                font=dict(size=18, color=self.colors['primary']),
                x=0.5
            ),
            paper_bgcolor=self.colors['background'],
            plot_bgcolor=self.colors['background'],
            showlegend=False,
            margin=dict(l=0, r=0, t=60, b=0),
            hovermode='closest'
        )
        
        return fig
    
    def _add_enhanced_header(self, fig: go.Figure, width: float, z: float, 
                            y: float, position: str, lighting: dict):
        """添加增强版集水管"""
        half_width = width / 2
        
        # 使用3D表面创建圆柱体效果
        theta = np.linspace(0, 2*np.pi, 20)
        x_cyl = half_width * np.cos(theta)
        z_cyl = 0.015 * np.sin(theta) + z
        
        # 主体
        fig.add_trace(go.Scatter3d(
            x=[-half_width, half_width, half_width, -half_width, -half_width],
            y=[y + 0.06, y + 0.06, y - 0.06, y - 0.06, y + 0.06],
            z=[z + 0.015, z + 0.015, z - 0.015, z - 0.015, z + 0.015],
            mode='lines',
            line=dict(color=self.colors['header'], width=6),
            name=f'集水管 {position}',
            showlegend=False
        ))
        
        # 高光效果
        fig.add_trace(go.Scatter3d(
            x=[-half_width * 0.8, half_width * 0.8],
            y=[y + 0.04, y + 0.04],
            z=[z + 0.02, z + 0.02],
            mode='lines',
            line=dict(color='rgba(255, 255, 255, 0.3)', width=2),
            showlegend=False
        ))
    
    def _add_enhanced_rails(self, fig: go.Figure, length: float, width: float, z: float):
        """添加增强版导轨"""
        half_width = width / 2 - 0.05
        half_length = length / 2
        
        for x in [-half_width, half_width]:
            y = np.linspace(-half_length, half_length, 30)
            x_coords = [x] * len(y)
            z_coords = [z] * len(y)
            
            # 主导轨
            fig.add_trace(go.Scatter3d(
                x=x_coords, y=y, z=z_coords,
                mode='lines',
                line=dict(color=self.colors['fiber'], width=3, dash='dot'),
                name='导轨',
                showlegend=False
            ))
            
            # 导轨端点标记
            fig.add_trace(go.Scatter3d(
                x=[x, x], y=[-half_length, half_length], z=[z, z],
                mode='markers',
                marker=dict(size=4, color=self.colors['fiber_glow']),
                showlegend=False
            ))
    
    def _add_enhanced_fibers(self, fig: go.Figure, width: float, length: float,
                            z: float, density: int, slack: float):
        """添加增强版膜丝 - 带材质效果"""
        half_width = width / 2 - 0.05
        half_length = length / 2
        
        # 为每根膜丝创建渐变颜色
        for i in range(density):
            x = (i / (density - 1)) * width - half_width if density > 1 else 0
            
            # 膜丝位置
            y = np.linspace(-half_length, half_length, 60)
            
            # 松弛度引起的弯曲
            slack_m = slack / 100 * length
            envelope = np.sin(np.pi * (y + half_length) / length)
            x_fiber = x + slack_m * envelope * 0.3
            z_fiber = z + slack_m * envelope * 0.15
            
            # 颜色渐变 - 基于位置
            color_intensity = np.sin(np.pi * (i / density))
            r = int(224 + 31 * color_intensity)
            g = int(232 + 23 * color_intensity)
            b = int(240 + 15 * color_intensity)
            fiber_color = f'rgb({r}, {g}, {b})'
            
            fig.add_trace(go.Scatter3d(
                x=x_fiber, y=y, z=z_fiber,
                mode='lines',
                line=dict(color=fiber_color, width=1.8),
                opacity=0.75,
                name='膜丝',
                showlegend=False
            ))
    
    def _add_enhanced_aeration_pipes(self, fig: go.Figure, sheet_count: int,
                                    sheet_spacing: float, sheet_width: float,
                                    half_length: float):
        """添加增强版曝气管"""
        depth = sheet_count * sheet_spacing + 0.4
        pipe_count = max(3, int(depth / 0.08))
        pipe_spacing = depth / pipe_count
        
        pipe_y = -half_length - 0.45
        
        for i in range(pipe_count):
            z = -depth / 2 + i * pipe_spacing
            
            # 主管 - 带渐变效果
            x = np.linspace(-sheet_width/2 - 0.15, sheet_width/2 + 0.15, 30)
            y_coords = [pipe_y] * len(x)
            z_coords = [z] * len(x)
            
            # 管道主体
            fig.add_trace(go.Scatter3d(
                x=x, y=y_coords, z=z_coords,
                mode='lines',
                line=dict(color=self.colors['pipe'], width=8),
                name='曝气管',
                showlegend=False
            ))
            
            # 管道高光
            fig.add_trace(go.Scatter3d(
                x=x, y=[pipe_y + 0.01] * len(x), z=[z + 0.005] * len(x),
                mode='lines',
                line=dict(color=self.colors['pipe_highlight'], width=3),
                showlegend=False
            ))
            
            # 曝气孔 - 带发光效果
            for j in range(6):
                x_orifice = -sheet_width/2 + j * sheet_width / 5
                
                # 主孔
                fig.add_trace(go.Scatter3d(
                    x=[x_orifice], y=[pipe_y], z=[z],
                    mode='markers',
                    marker=dict(
                        size=5, 
                        color=self.colors['primary'],
                        symbol='circle'
                    ),
                    name='曝气孔',
                    showlegend=False
                ))
                
                # 发光效果
                fig.add_trace(go.Scatter3d(
                    x=[x_orifice], y=[pipe_y], z=[z],
                    mode='markers',
                    marker=dict(
                        size=12, 
                        color='rgba(0, 242, 255, 0.2)',
                        symbol='circle'
                    ),
                    showlegend=False
                ))
    
    def _add_enhanced_tank_frame(self, fig: go.Figure, width: float, length: float,
                                sheet_count: int, sheet_spacing: float):
        """添加增强版水箱框架"""
        tank_width = width + 0.4
        tank_length = length + 1.0
        tank_depth = sheet_count * sheet_spacing + 0.4
        
        half_w = tank_width / 2
        half_l = tank_length / 2
        half_d = tank_depth / 2
        
        # 框架边线 - 带发光效果
        edges = [
            # 底面
            ([-half_w, half_w], [-half_l, -half_l], [-half_d, -half_d]),
            ([half_w, half_w], [-half_l, half_l], [-half_d, -half_d]),
            ([half_w, -half_w], [half_l, half_l], [-half_d, -half_d]),
            ([-half_w, -half_w], [half_l, -half_l], [-half_d, -half_d]),
            # 顶面
            ([-half_w, half_w], [-half_l, -half_l], [half_d, half_d]),
            ([half_w, half_w], [-half_l, half_l], [half_d, half_d]),
            ([half_w, -half_w], [half_l, half_l], [half_d, half_d]),
            ([-half_w, -half_w], [half_l, -half_l], [half_d, half_d]),
            # 垂直边
            ([-half_w, -half_w], [-half_l, -half_l], [-half_d, half_d]),
            ([half_w, half_w], [-half_l, -half_l], [-half_d, half_d]),
            ([half_w, half_w], [half_l, half_l], [-half_d, half_d]),
            ([-half_w, -half_w], [half_l, half_l], [-half_d, half_d]),
        ]
        
        for edge in edges:
            fig.add_trace(go.Scatter3d(
                x=edge[0], y=edge[1], z=edge[2],
                mode='lines',
                line=dict(color=self.colors['tank_frame'], width=2),
                name='水箱',
                showlegend=False
            ))
    
    def _add_enhanced_water_surface(self, fig: go.Figure, width: float, length: float,
                                   sheet_count: int, sheet_spacing: float):
        """添加增强版水面效果"""
        half_w = (width + 0.4) / 2
        half_l = (length + 1.0) / 2
        y_surface = length / 2 + 0.3
        
        # 创建波浪效果
        x = np.linspace(-half_w, half_w, 30)
        z = np.linspace(-(sheet_count * sheet_spacing + 0.4)/2, 
                       (sheet_count * sheet_spacing + 0.4)/2, 30)
        X, Z = np.meshgrid(x, z)
        
        # 添加波浪
        Y = y_surface + 0.02 * np.sin(X * 3) * np.cos(Z * 2)
        
        fig.add_trace(go.Surface(
            x=X, y=Y, z=Z,
            colorscale=[[0, 'rgba(10, 40, 80, 0.4)'], 
                       [0.5, 'rgba(10, 60, 100, 0.3)'],
                       [1, 'rgba(10, 40, 80, 0.4)']],
            showscale=False,
            name='水面',
            hoverinfo='skip',
            lighting=dict(
                ambient=0.8,
                diffuse=0.5,
                specular=0.9,
                roughness=0.1
            )
        ))
    
    def create_unified_3d_scene(self,
                                sheet_count: int = 5,
                                sheet_width: float = 1.25,
                                sheet_length: float = 2.0,
                                sheet_spacing: float = 0.08,
                                fiber_diameter: float = 1.65,
                                fiber_slack: float = 1.5,
                                show_fibers: bool = True,
                                fiber_density: int = 20,
                                bubble_count: int = 150,
                                aeration_intensity: float = 100,
                                water_depth: float = 4.0,
                                show_bubbles: bool = True,
                                show_sludge: bool = False,
                                sludge_count: int = 200) -> go.Figure:
        """
        创建统一的3D场景 - 膜组件 + 气泡 + 污泥
        
        将气泡粒子系统集成到膜组件结构中，实现整体可视化效果
        """
        fig = go.Figure()
        
        half_length = sheet_length / 2
        half_width = sheet_width / 2
        tank_depth = sheet_count * sheet_spacing + 0.4
        
        # ========== 1. 绘制膜组件结构 ==========
        for i in range(sheet_count):
            z = (i - (sheet_count - 1) / 2) * sheet_spacing
            
            # 上下集水管
            self._add_enhanced_header(fig, sheet_width, z, half_length, 'top', {})
            self._add_enhanced_header(fig, sheet_width, z, -half_length, 'bottom', {})
            
            # 导轨
            self._add_enhanced_rails(fig, sheet_length, sheet_width, z)
            
            # 膜丝
            if show_fibers:
                self._add_enhanced_fibers(fig, sheet_width, sheet_length, z, 
                                         fiber_density, fiber_slack)
        
        # 曝气管
        self._add_enhanced_aeration_pipes(fig, sheet_count, sheet_spacing, 
                                         sheet_width, half_length)
        
        # 水箱框架
        self._add_enhanced_tank_frame(fig, sheet_width, sheet_length, 
                                     sheet_count, sheet_spacing)
        
        # 水面
        self._add_enhanced_water_surface(fig, sheet_width, sheet_length,
                                        sheet_count, sheet_spacing)
        
        # ========== 2. 添加气泡粒子 ==========
        if show_bubbles:
            self._add_bubbles_to_scene(fig, sheet_width, sheet_length, tank_depth,
                                      bubble_count, aeration_intensity, half_length)
        
        # ========== 3. 添加污泥粒子 ==========
        if show_sludge:
            self._add_sludge_to_scene(fig, sheet_width, sheet_length, tank_depth,
                                     sludge_count, half_length)
        
        # ========== 4. 设置布局和相机 ==========
        fig.update_layout(
            scene=dict(
                xaxis=dict(
                    title='X (m)', 
                    range=[-1.5, 1.5],
                    backgroundcolor=self.colors['background'],
                    gridcolor='rgba(0, 242, 255, 0.08)',
                    showbackground=True,
                    zerolinecolor='rgba(0, 242, 255, 0.2)'
                ),
                yaxis=dict(
                    title='Y (高度 m)', 
                    range=[-3, 3],
                    backgroundcolor=self.colors['background'],
                    gridcolor='rgba(0, 242, 255, 0.08)',
                    showbackground=True,
                    zerolinecolor='rgba(0, 242, 255, 0.2)'
                ),
                zaxis=dict(
                    title='Z (m)', 
                    range=[-0.6, 0.6],
                    backgroundcolor=self.colors['background'],
                    gridcolor='rgba(0, 242, 255, 0.08)',
                    showbackground=True,
                    zerolinecolor='rgba(0, 242, 255, 0.2)'
                ),
                aspectmode='manual',
                aspectratio=dict(x=1.5, y=3, z=0.6),
                camera=dict(
                    eye=dict(x=2.8, y=2.0, z=1.8),
                    center=dict(x=0, y=0, z=0),
                    up=dict(x=0, y=0, z=1)
                ),
                bgcolor=self.colors['background']
            ),
            title=dict(
                text='<b>MBR膜组件统一3D场景</b><br><sup>膜组件 + 气泡 + 污泥</sup>',
                font=dict(size=18, color=self.colors['primary']),
                x=0.5
            ),
            paper_bgcolor=self.colors['background'],
            plot_bgcolor=self.colors['background'],
            showlegend=True,
            legend=dict(
                font=dict(color='white', size=10),
                bgcolor='rgba(0,0,0,0.3)',
                x=0.02,
                y=0.98
            ),
            margin=dict(l=0, r=0, t=80, b=0),
            hovermode='closest'
        )
        
        return fig
    
    def _add_bubbles_to_scene(self, fig: go.Figure, sheet_width: float, 
                             sheet_length: float, tank_depth: float,
                             count: int, intensity: float, half_length: float):
        """在3D场景中添加气泡"""
        np.random.seed(42)
        
        # 气泡分布范围 - 在膜组件周围
        half_width = sheet_width / 2 + 0.1
        
        # 气泡位置 - 从曝气管位置开始
        bubble_x = (np.random.random(count) - 0.5) * 2 * half_width
        bubble_z = (np.random.random(count) - 0.5) * tank_depth * 0.8
        
        # Y位置 - 从底部开始，向上分布
        # 曝气管在 y = -half_length - 0.45
        pipe_y = -half_length - 0.45
        bubble_y = pipe_y + np.random.random(count) * (sheet_length + 0.8)
        
        # 气泡大小 - 基于曝气强度
        base_size = 2.0 + intensity / 50
        bubble_size = np.random.exponential(1.0, count) + base_size * 0.5
        bubble_size = np.clip(bubble_size, 1.0, 6.0)
        
        # 气泡颜色 - 基于大小渐变
        colors = []
        for size in bubble_size:
            if size < 2:
                colors.append('rgba(200, 240, 255, 0.7)')
            elif size < 3.5:
                colors.append('rgba(150, 220, 255, 0.6)')
            else:
                colors.append('rgba(100, 200, 255, 0.5)')
        
        # 添加气泡轨迹 - 上升曲线
        for i in range(min(count, 80)):  # 限制轨迹数量避免性能问题
            # 创建上升轨迹
            y_start = bubble_y[i]
            y_end = min(y_start + 0.5 + np.random.random() * 0.5, half_length + 0.5)
            
            y_traj = np.linspace(y_start, y_end, 15)
            # 添加摆动
            x_traj = bubble_x[i] + 0.02 * np.sin(y_traj * 5) * (y_traj - y_start)
            z_traj = [bubble_z[i]] * len(y_traj)
            
            # 轨迹透明度渐变
            opacity_traj = np.linspace(0.3, 0.1, len(y_traj))
            
            fig.add_trace(go.Scatter3d(
                x=x_traj, y=y_traj, z=z_traj,
                mode='lines',
                line=dict(
                    color='rgba(170, 221, 255, 0.3)',
                    width=1
                ),
                opacity=0.3,
                showlegend=False,
                hoverinfo='skip'
            ))
        
        # 添加气泡主体
        fig.add_trace(go.Scatter3d(
            x=bubble_x, y=bubble_y, z=bubble_z,
            mode='markers',
            marker=dict(
                size=bubble_size * 2,
                color=colors,
                opacity=0.7,
                symbol='circle',
                line=dict(color='rgba(255, 255, 255, 0.4)', width=1)
            ),
            name='气泡',
            hovertemplate='气泡<br>大小: %{marker.size:.1f}<br>位置: (%{x:.2f}, %{y:.2f}, %{z:.2f})<extra></extra>'
        ))
        
        # 添加气泡发光效果
        fig.add_trace(go.Scatter3d(
            x=bubble_x, y=bubble_y, z=bubble_z,
            mode='markers',
            marker=dict(
                size=bubble_size * 3.5,
                color='rgba(200, 240, 255, 0.15)',
                opacity=0.3,
                symbol='circle'
            ),
            name='气泡光晕',
            showlegend=False,
            hoverinfo='skip'
        ))
    
    def _add_sludge_to_scene(self, fig: go.Figure, sheet_width: float,
                            sheet_length: float, tank_depth: float,
                            count: int, half_length: float):
        """在3D场景中添加污泥粒子"""
        np.random.seed(123)
        
        half_width = sheet_width / 2 + 0.15
        
        # 污泥位置
        sludge_x = (np.random.random(count) - 0.5) * 2 * half_width
        sludge_z = (np.random.random(count) - 0.5) * tank_depth * 0.9
        
        # 垂直分布 - 底部浓度高
        y_raw = np.random.random(count)
        # 使用幂函数使污泥集中在底部
        sludge_y = -half_length - 0.3 + y_raw**1.5 * (sheet_length + 0.6)
        
        # 污泥大小
        sludge_size = np.random.random(count) * 2.5 + 1.5
        
        # 根据高度确定颜色 - 底部深色，上部浅色
        colors = []
        for y in sludge_y:
            y_norm = (y - (-half_length - 0.3)) / (sheet_length + 0.6)
            if y_norm < 0.3:
                colors.append(self.colors['sludge_dark'])
            else:
                colors.append(self.colors['sludge'])
        
        fig.add_trace(go.Scatter3d(
            x=sludge_x, y=sludge_y, z=sludge_z,
            mode='markers',
            marker=dict(
                size=sludge_size,
                color=colors,
                opacity=0.55,
                symbol='circle'
            ),
            name='污泥粒子',
            hovertemplate='污泥<br>位置: (%{x:.2f}, %{y:.2f}, %{z:.2f})<extra></extra>'
        ))
    
    def create_particle_system(self,
                               particle_type: str = 'bubble',
                               count: int = 200,
                               intensity: float = 100,
                               water_depth: float = 4.0) -> go.Figure:
        """
        创建粒子系统可视化
        
        Args:
            particle_type: 'bubble' 或 'sludge'
            count: 粒子数量
            intensity: 曝气强度
            water_depth: 水深
        """
        fig = go.Figure()
        
        np.random.seed(42)
        
        if particle_type == 'bubble':
            # 气泡粒子
            half_width = 0.6
            
            bubble_x = (np.random.random(count) - 0.5) * 2 * half_width
            bubble_z = (np.random.random(count) - 0.5) * 0.3
            bubble_y = np.random.random(count) * (-water_depth) - 0.5
            
            # 气泡大小分布
            bubble_size = np.random.exponential(1.5, count) + 0.5
            bubble_size = np.clip(bubble_size, 0.5, 4.0)
            
            # 颜色基于大小
            colors = [self._bubble_color(s) for s in bubble_size]
            
            # 气泡
            fig.add_trace(go.Scatter3d(
                x=bubble_x, y=bubble_y, z=bubble_z,
                mode='markers',
                marker=dict(
                    size=bubble_size * 4,
                    color=colors,
                    opacity=0.7,
                    symbol='circle',
                    line=dict(color='rgba(255, 255, 255, 0.3)', width=1)
                ),
                name='气泡'
            ))
            
            title = '气泡粒子系统'
            
        else:  # sludge
            # 污泥粒子
            half_width = 0.8
            
            sludge_x = (np.random.random(count) - 0.5) * 2 * half_width
            sludge_z = (np.random.random(count) - 0.5) * 0.4
            
            # 垂直分布 - 底部浓度高
            y_raw = np.random.random(count)
            sludge_y = -1.5 + y_raw**2 * 3
            
            # 大小
            sludge_size = np.random.random(count) * 3 + 1
            
            #  settled vs suspended
            settled = sludge_y < -0.5
            colors = [self.colors['sludge_dark'] if s else self.colors['sludge'] 
                     for s in settled]
            
            fig.add_trace(go.Scatter3d(
                x=sludge_x, y=sludge_y, z=sludge_z,
                mode='markers',
                marker=dict(
                    size=sludge_size * 2,
                    color=colors,
                    opacity=0.6,
                    symbol='circle'
                ),
                name='污泥粒子'
            ))
            
            title = '污泥粒子系统'
        
        fig.update_layout(
            scene=dict(
                xaxis=dict(title='X (m)', range=[-1, 1]),
                yaxis=dict(title='Y (高度 m)', range=[-3, 2]),
                zaxis=dict(title='Z (m)', range=[-0.5, 0.5]),
                aspectmode='manual',
                aspectratio=dict(x=1, y=2.5, z=0.5),
                bgcolor=self.colors['background']
            ),
            paper_bgcolor=self.colors['background'],
            title=dict(
                text=f'<b>{title}</b>',
                font=dict(size=16, color=self.colors['primary'])
            )
        )
        
        return fig
    
    def _bubble_color(self, size: float) -> str:
        """根据气泡大小获取颜色"""
        if size < 1.0:
            return 'rgba(170, 221, 255, 0.8)'
        elif size < 2.0:
            return 'rgba(140, 200, 255, 0.7)'
        else:
            return 'rgba(100, 180, 255, 0.6)'
    
    def create_cfd_visualization(self, cfd_results) -> go.Figure:
        """
        CFD结果可视化

        使用2D热力图展示速度场、剪切应力场、湍动能分布和气泡分布
        """
        fig = make_subplots(
            rows=2, cols=2,
            specs=[[{'type': 'heatmap'}, {'type': 'heatmap'}],
                   [{'type': 'heatmap'}, {'type': 'heatmap'}]],
            subplot_titles=('速度场', '剪切应力场', '湍动能分布', '气泡分布'),
            vertical_spacing=0.1
        )

        if hasattr(cfd_results, 'velocity_field') and cfd_results.velocity_field.size > 0:
            velocity = cfd_results.velocity_field

            # 速度场 - 取中间切片
            mid_z = velocity.shape[2] // 2
            vel_slice = velocity[:, :, mid_z].T
            fig.add_trace(go.Heatmap(
                z=vel_slice,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title='速度 (m/s)')
            ), row=1, col=1)

            # 剪切应力场
            if hasattr(cfd_results, 'shear_stress_field'):
                shear = cfd_results.shear_stress_field
                shear_slice = shear[:, :, mid_z].T
                fig.add_trace(go.Heatmap(
                    z=shear_slice,
                    colorscale='Plasma',
                    showscale=True,
                    colorbar=dict(title='剪切应力 (Pa)')
                ), row=1, col=2)
            else:
                fig.add_trace(go.Heatmap(
                    z=np.zeros_like(vel_slice),
                    colorscale='Plasma',
                    showscale=False
                ), row=1, col=2)

            # 湍动能分布 - 取中间切片
            if hasattr(cfd_results, 'turbulence_field'):
                turb = cfd_results.turbulence_field
                turb_slice = turb[:, :, mid_z].T
            else:
                turb_slice = np.abs(velocity[:, :, mid_z].T - np.mean(velocity[:, :, mid_z].T))
            fig.add_trace(go.Heatmap(
                z=turb_slice,
                colorscale='Inferno',
                showscale=True,
                colorbar=dict(title='湍动能 (m²/s²)')
            ), row=2, col=1)

            # 气泡分布 - 基于速度场模拟
            bubble_dist = np.exp(-velocity[:, :, mid_z].T / (np.max(velocity[:, :, mid_z].T) + 1e-10))
            fig.add_trace(go.Heatmap(
                z=bubble_dist,
                colorscale='Blues',
                showscale=True,
                colorbar=dict(title='气泡浓度')
            ), row=2, col=2)
        else:
            # 无数据时显示空白热力图
            for row, col in [(1, 1), (1, 2), (2, 1), (2, 2)]:
                fig.add_trace(go.Heatmap(
                    z=np.zeros((10, 10)),
                    colorscale='Viridis',
                    showscale=False
                ), row=row, col=col)

        fig.update_layout(
            title=dict(
                text='<b>CFD计算结果可视化</b>',
                font=dict(size=16, color=self.colors['primary'])
            ),
            paper_bgcolor=self.colors['background'],
            plot_bgcolor=self.colors['background'],
            font=dict(color='white'),
            height=800
        )

        return fig
    
    def create_heatmap_3d(self, data: np.ndarray, 
                         x_label: str = 'X',
                         y_label: str = 'Y',
                         title: str = '3D热力图') -> go.Figure:
        """
        创建3D热力图
        """
        fig = go.Figure()
        
        x = np.arange(data.shape[0])
        y = np.arange(data.shape[1])
        X, Y = np.meshgrid(x, y)
        
        fig.add_trace(go.Surface(
            x=X, y=Y, z=data.T,
            colorscale='Jet',
            showscale=True,
            colorbar=dict(
                title='值',
                titleside='right'
            )
        ))
        
        fig.update_layout(
            title=dict(
                text=f'<b>{title}</b>',
                font=dict(size=16, color=self.colors['primary'])
            ),
            scene=dict(
                xaxis_title=x_label,
                yaxis_title=y_label,
                zaxis_title='值',
                bgcolor=self.colors['background']
            ),
            paper_bgcolor=self.colors['background'],
            height=600
        )
        
        return fig
    
    def create_animation_frames(self,
                                frame_count: int = 30,
                                particle_type: str = 'bubble') -> go.Figure:
        """
        创建动画帧序列
        
        用于展示动态效果
        """
        fig = go.Figure()
        
        np.random.seed(42)
        count = 150
        
        # 初始化位置
        x_base = (np.random.random(count) - 0.5) * 1.2
        z_base = (np.random.random(count) - 0.5) * 0.3
        y_base = np.random.random(count) * (-4) - 0.5
        sizes = np.random.random(count) * 2 + 0.5
        
        frames = []
        
        for frame in range(frame_count):
            t = frame * 0.15
            
            # 更新位置
            y_new = y_base + 0.5 * t
            y_new = np.where(y_new > 0, y_base, y_new)
            
            # 添加摆动
            x_new = x_base + 0.03 * np.sin(t * 2 + z_base * 10)
            
            # 大小变化
            size_new = sizes * (1 + 0.1 * np.sin(t))
            
            frame_data = [go.Scatter3d(
                x=x_new, y=y_new, z=z_base,
                mode='markers',
                marker=dict(
                    size=size_new * 3,
                    color=[self._bubble_color(s) for s in size_new],
                    opacity=0.6
                ),
                name='粒子'
            )]
            
            frames.append(go.Frame(data=frame_data, name=f'frame{frame}'))
        
        # 初始帧
        fig.add_trace(go.Scatter3d(
            x=x_base, y=y_base, z=z_base,
            mode='markers',
            marker=dict(
                size=sizes * 3,
                color=[self._bubble_color(s) for s in sizes],
                opacity=0.6
            ),
            name='粒子'
        ))
        
        fig.frames = frames
        
        # 动画控制
        fig.update_layout(
            updatemenus=[{
                'type': 'buttons',
                'showactive': False,
                'buttons': [
                    {
                        'label': '▶ 播放',
                        'method': 'animate',
                        'args': [None, {
                            'frame': {'duration': 80, 'redraw': True},
                            'fromcurrent': True,
                            'transition': {'duration': 40}
                        }]
                    },
                    {
                        'label': '⏸ 暂停',
                        'method': 'animate',
                        'args': [[None], {
                            'frame': {'duration': 0, 'redraw': False},
                            'mode': 'immediate',
                            'transition': {'duration': 0}
                        }]
                    },
                    {
                        'label': '🔄 循环',
                        'method': 'animate',
                        'args': [None, {
                            'frame': {'duration': 80, 'redraw': True},
                            'mode': 'immediate',
                            'fromcurrent': True,
                            'transition': {'duration': 40}
                        }]
                    }
                ]
            }],
            scene=dict(
                xaxis=dict(title='X (m)', range=[-1, 1]),
                yaxis=dict(title='Y (高度 m)', range=[-5, 3]),
                zaxis=dict(title='Z (m)', range=[-0.5, 0.5]),
                aspectmode='manual',
                aspectratio=dict(x=1, y=2, z=0.3),
                bgcolor=self.colors['background']
            ),
            paper_bgcolor=self.colors['background'],
            title=dict(
                text='<b>动态粒子动画</b>',
                font=dict(size=16, color=self.colors['primary'])
            )
        )
        
        return fig
    
    def create_performance_radar(self, result) -> go.Figure:
        """
        创建性能雷达图
        """
        categories = ['能效', '剪切控制', '污染控制', '经济性', '处理效率', '可持续性']
        
        # 计算各项得分
        energy_score = max(0, 100 - result.sec_kwh_m3 * 100)
        shear_score = 100 - abs(result.avg_shear_pa - 1.5) * 30
        shear_score = max(0, min(100, shear_score))
        
        fouling_scores = {
            'very_low': 100, 'low': 90, 'medium': 70, 
            'high': 50, 'critical': 30
        }
        fouling_score = fouling_scores.get(result.fouling_risk.value, 50)
        
        cost_score = max(0, 100 - result.total_cost_rmb_m3 * 50)
        eff_score = (result.cod_removal_efficiency + result.bod_removal_efficiency) / 2
        sustain_score = result.sustainability_score
        
        values = [energy_score, shear_score, fouling_score, 
                 cost_score, eff_score, sustain_score]
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=values + [values[0]],  # 闭合
            theta=categories + [categories[0]],
            fill='toself',
            fillcolor='rgba(0, 242, 255, 0.2)',
            line=dict(color=self.colors['primary'], width=2),
            name='当前配置'
        ))
        
        # 添加理想值参考
        fig.add_trace(go.Scatterpolar(
            r=[90, 90, 90, 90, 95, 90, 90],
            theta=categories + [categories[0]],
            fill='toself',
            fillcolor='rgba(0, 255, 136, 0.1)',
            line=dict(color=self.colors['success'], width=1, dash='dash'),
            name='理想值'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100],
                    gridcolor='rgba(0, 242, 255, 0.1)'
                ),
                bgcolor=self.colors['background']
            ),
            paper_bgcolor=self.colors['background'],
            font=dict(color='white'),
            title=dict(
                text='<b>性能雷达图</b>',
                font=dict(size=16, color=self.colors['primary'])
            ),
            showlegend=True,
            legend=dict(
                font=dict(color='white'),
                bgcolor='rgba(0, 0, 0, 0.5)'
            )
        )
        
        return fig
    
    def create_sensitivity_chart(self, sensitivities: Dict[str, float]) -> go.Figure:
        """
        创建敏感性分析图
        """
        # 分离SEC和评分敏感性
        sec_items = {k: v for k, v in sensitivities.items() if '_sec' in k}
        score_items = {k: v for k, v in sensitivities.items() if '_score' in k}
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('对SEC的影响', '对运行评分的影响'),
            specs=[[{'type': 'bar'}, {'type': 'bar'}]]
        )
        
        # SEC敏感性
        params = [k.replace('_sec', '') for k in sec_items.keys()]
        values = list(sec_items.values())
        
        fig.add_trace(go.Bar(
            x=params,
            y=values,
            marker_color=[self.colors['primary']] * len(params),
            text=[f'{v:.3f}' for v in values],
            textposition='outside',
            name='SEC敏感性'
        ), row=1, col=1)
        
        # 评分敏感性
        params_score = [k.replace('_score', '') for k in score_items.keys()]
        values_score = list(score_items.values())
        
        fig.add_trace(go.Bar(
            x=params_score,
            y=values_score,
            marker_color=[self.colors['success']] * len(params_score),
            text=[f'{v:.1f}' for v in values_score],
            textposition='outside',
            name='评分敏感性'
        ), row=1, col=2)
        
        fig.update_layout(
            title=dict(
                text='<b>参数敏感性分析</b>',
                font=dict(size=16, color=self.colors['primary'])
            ),
            paper_bgcolor=self.colors['background'],
            plot_bgcolor=self.colors['background'],
            font=dict(color='white'),
            showlegend=False,
            height=400
        )
        
        return fig


def create_visualizer_v2(config=None) -> MBRVisualizerV2:
    """工厂函数"""
    return MBRVisualizerV2(config)


if __name__ == "__main__":
    viz = MBRVisualizerV2()
    
    # 测试增强版膜结构
    fig = viz.create_membrane_structure_enhanced()
    fig.show()
