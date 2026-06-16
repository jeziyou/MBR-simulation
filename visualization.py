"""
MBR仿真系统 - 3D可视化组件
MBR Simulation System - 3D Visualization Components

使用Plotly实现交互式3D可视化，包括：
- MBR膜组件3D结构
- 气泡上升动画
- 膜丝振动模拟
- 污泥分布
- 剪切力分布热力图

作者: MBR Engineering Team
版本: 2.0
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from typing import List, Dict, Tuple, Optional
import colorsys


class MBRVisualizer:
    """MBR系统3D可视化器"""
    
    def __init__(self, config=None):
        """初始化可视化器"""
        self.config = config
        self.color_schemes = {
            'primary': '#00f2ff',
            'success': '#00ff88',
            'warn': '#ff9900',
            'danger': '#ff4444',
            'sludge': '#d4a84b',
            'fiber': '#ffffff',
            'bubble': '#aaddff',
            'water': '#0a1a2e',
            'pipe': '#445566',
            'header': '#F5DEB3'
        }
    
    def create_membrane_structure(self, 
                                   sheet_count: int = 5,
                                   sheet_width: float = 1.25,
                                   sheet_length: float = 2.0,
                                   sheet_spacing: float = 0.08,
                                   fiber_diameter: float = 1.65,
                                   fiber_slack: float = 1.5,
                                   show_fibers: bool = True,
                                   fiber_density: int = 20) -> go.Figure:
        """
        创建MBR膜组件3D结构
        
        Args:
            sheet_count: 膜片数量
            sheet_width: 膜片宽度 (m)
            sheet_length: 膜片长度 (m)
            sheet_spacing: 膜片间距 (m)
            fiber_diameter: 膜丝直径 (mm)
            fiber_slack: 膜丝松弛度 (%)
            show_fibers: 是否显示膜丝
            fiber_density: 膜丝密度 (每片显示数量)
            
        Returns:
            Plotly Figure对象
        """
        fig = go.Figure()
        
        half_length = sheet_length / 2
        half_width = sheet_width / 2
        
        # 绘制膜片框架和膜丝
        for i in range(sheet_count):
            z = (i - (sheet_count - 1) / 2) * sheet_spacing
            
            # 上下集水管
            self._add_header(fig, sheet_width, z, half_length, 'top')
            self._add_header(fig, sheet_width, z, -half_length, 'bottom')
            
            # 导轨
            self._add_rails(fig, sheet_length, sheet_width, z)
            
            # 膜丝
            if show_fibers:
                self._add_fibers(fig, sheet_width, sheet_length, z, 
                                fiber_density, fiber_slack)
        
        # 绘制曝气管
        self._add_aeration_pipes(fig, sheet_count, sheet_spacing, 
                                 sheet_width, half_length)
        
        # 绘制水箱框架
        self._add_tank_frame(fig, sheet_width, sheet_length, 
                            sheet_count, sheet_spacing)
        
        # 绘制水面
        self._add_water_surface(fig, sheet_width, sheet_length,
                               sheet_count, sheet_spacing)
        
        # 设置布局
        fig.update_layout(
            scene=dict(
                xaxis=dict(title='X (m)', range=[-1.5, 1.5], 
                          backgroundcolor='rgba(1, 5, 10, 0.9)',
                          gridcolor='rgba(0, 242, 255, 0.1)'),
                yaxis=dict(title='Y (高度 m)', range=[-3, 3],
                          backgroundcolor='rgba(1, 5, 10, 0.9)',
                          gridcolor='rgba(0, 242, 255, 0.1)'),
                zaxis=dict(title='Z (m)', range=[-0.5, 0.5],
                          backgroundcolor='rgba(1, 5, 10, 0.9)',
                          gridcolor='rgba(0, 242, 255, 0.1)'),
                aspectmode='manual',
                aspectratio=dict(x=1.5, y=3, z=0.5),
                camera=dict(eye=dict(x=2, y=1.5, z=1.5)),
                bgcolor='rgba(1, 5, 10, 0.95)'
            ),
            title=dict(
                text='MBR膜组件3D结构',
                font=dict(size=16, color=self.color_schemes['primary'])
            ),
            paper_bgcolor='rgba(1, 5, 10, 0.95)',
            plot_bgcolor='rgba(1, 5, 10, 0.95)',
            showlegend=False,
            margin=dict(l=0, r=0, t=40, b=0)
        )
        
        return fig
    
    def _add_header(self, fig: go.Figure, width: float, z: float, 
                   y: float, position: str):
        """添加集水管"""
        half_width = width / 2
        
        # 集水管主体
        x = [-half_width, half_width, half_width, -half_width, -half_width]
        y_coords = [y + 0.06, y + 0.06, y - 0.06, y - 0.06, y + 0.06]
        z_coords = [z + 0.015, z + 0.015, z - 0.015, z - 0.015, z + 0.015]
        
        fig.add_trace(go.Scatter3d(
            x=x, y=y_coords, z=z_coords,
            mode='lines',
            line=dict(color=self.color_schemes['header'], width=4),
            name=f'集水管 {position}',
            showlegend=False
        ))
    
    def _add_rails(self, fig: go.Figure, length: float, width: float, z: float):
        """添加导轨"""
        half_width = width / 2 - 0.05
        half_length = length / 2
        
        for x in [-half_width, half_width]:
            y = np.linspace(-half_length, half_length, 20)
            x_coords = [x] * len(y)
            z_coords = [z] * len(y)
            
            fig.add_trace(go.Scatter3d(
                x=x_coords, y=y, z=z_coords,
                mode='lines',
                line=dict(color=self.color_schemes['fiber'], width=2, dash='dot'),
                name='导轨',
                showlegend=False
            ))
    
    def _add_fibers(self, fig: go.Figure, width: float, length: float,
                   z: float, density: int, slack: float):
        """添加膜丝"""
        half_width = width / 2 - 0.05
        half_length = length / 2
        
        for i in range(density):
            x = (i / (density - 1)) * width - half_width if density > 1 else 0
            
            # 膜丝位置 - 添加松弛度造成的弯曲
            y = np.linspace(-half_length, half_length, 50)
            
            # 松弛度引起的弯曲 (正弦曲线)
            slack_m = slack / 100 * length
            envelope = np.sin(np.pi * (y + half_length) / length)
            x_fiber = x + slack_m * envelope * 0.3
            
            z_fiber = z + slack_m * envelope * 0.1
            
            fig.add_trace(go.Scatter3d(
                x=x_fiber, y=y, z=z_fiber,
                mode='lines',
                line=dict(color=self.color_schemes['fiber'], width=1.5),
                opacity=0.7,
                name='膜丝',
                showlegend=False
            ))
    
    def _add_aeration_pipes(self, fig: go.Figure, sheet_count: int,
                           sheet_spacing: float, sheet_width: float,
                           half_length: float):
        """添加曝气管"""
        depth = sheet_count * sheet_spacing + 0.4
        pipe_count = max(3, int(depth / 0.1))
        pipe_spacing = depth / pipe_count
        
        pipe_y = -half_length - 0.45
        
        for i in range(pipe_count):
            z = -depth / 2 + i * pipe_spacing
            
            # 主管
            x = np.linspace(-sheet_width/2 - 0.15, sheet_width/2 + 0.15, 20)
            y_coords = [pipe_y] * len(x)
            z_coords = [z] * len(x)
            
            fig.add_trace(go.Scatter3d(
                x=x, y=y_coords, z=z_coords,
                mode='lines',
                line=dict(color=self.color_schemes['pipe'], width=6),
                name='曝气管',
                showlegend=False
            ))
            
            # 曝气孔
            for j in range(5):
                x_orifice = -sheet_width/2 + j * sheet_width / 4
                fig.add_trace(go.Scatter3d(
                    x=[x_orifice], y=[pipe_y], z=[z],
                    mode='markers',
                    marker=dict(size=3, color=self.color_schemes['primary']),
                    name='曝气孔',
                    showlegend=False
                ))
    
    def _add_tank_frame(self, fig: go.Figure, width: float, length: float,
                       sheet_count: int, sheet_spacing: float):
        """添加水箱框架"""
        tank_width = width + 0.4
        tank_length = length + 1.0
        tank_depth = sheet_count * sheet_spacing + 0.4
        
        half_w = tank_width / 2
        half_l = tank_length / 2
        half_d = tank_depth / 2
        
        # 框架边线
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
                line=dict(color='rgba(30, 74, 133, 0.5)', width=2),
                name='水箱',
                showlegend=False
            ))
    
    def _add_water_surface(self, fig: go.Figure, width: float, length: float,
                          sheet_count: int, sheet_spacing: float):
        """添加水面"""
        half_w = (width + 0.4) / 2
        half_l = (length + 1.0) / 2
        y_surface = length / 2 + 0.3
        
        x = np.linspace(-half_w, half_w, 20)
        z = np.linspace(-(sheet_count * sheet_spacing + 0.4)/2, 
                       (sheet_count * sheet_spacing + 0.4)/2, 20)
        X, Z = np.meshgrid(x, z)
        Y = np.full_like(X, y_surface)
        
        fig.add_trace(go.Surface(
            x=X, y=Y, z=Z,
            colorscale=[[0, 'rgba(10, 26, 46, 0.3)'], [1, 'rgba(10, 26, 46, 0.3)']],
            showscale=False,
            name='水面',
            hoverinfo='skip'
        ))
    
    def create_bubble_animation(self,
                                 bubble_count: int = 200,
                                 intensity: float = 100,
                                 water_depth: float = 4.0,
                                 duration_frames: int = 50) -> go.Figure:
        """
        创建气泡上升动画帧
        
        Args:
            bubble_count: 气泡数量
            intensity: 曝气强度
            water_depth: 水深
            duration_frames: 动画帧数
            
        Returns:
            Plotly Figure对象 (含动画帧)
        """
        fig = go.Figure()
        
        # 初始化气泡位置
        np.random.seed(42)
        half_width = 0.6
        
        bubble_x = (np.random.random(bubble_count) - 0.5) * 2 * half_width
        bubble_z = (np.random.random(bubble_count) - 0.5) * 0.3
        bubble_y = np.random.random(bubble_count) * (-water_depth) - 0.5
        bubble_size = 0.5 + np.random.random(bubble_count) * 1.5
        
        # 气泡速度 (与曝气强度相关)
        velocity = 0.2 + (intensity / 100) * 0.3  # m/s
        
        # 创建动画帧
        frames = []
        for frame in range(duration_frames):
            t = frame * 0.1
            
            # 更新气泡位置
            y_new = bubble_y + velocity * t
            
            # 循环气泡
            y_new = np.where(y_new > water_depth/2, 
                           -water_depth/2 - 0.5, y_new)
            
            # 添加摆动
            x_new = bubble_x + 0.02 * np.sin(t * 2 + bubble_z * 10)
            
            # 气泡大小变化 (上升时膨胀)
            depth_factor = 1 + (water_depth/2 - y_new) / water_depth * 0.3
            size_new = bubble_size * depth_factor
            
            frame_data = [go.Scatter3d(
                x=x_new, y=y_new, z=bubble_z,
                mode='markers',
                marker=dict(
                    size=size_new * 3,
                    color=self.color_schemes['bubble'],
                    opacity=0.6
                ),
                name='气泡'
            )]
            
            frames.append(go.Frame(data=frame_data, name=f'frame{frame}'))
        
        # 初始帧
        fig.add_trace(go.Scatter3d(
            x=bubble_x, y=bubble_y, z=bubble_z,
            mode='markers',
            marker=dict(
                size=bubble_size * 3,
                color=self.color_schemes['bubble'],
                opacity=0.6
            ),
            name='气泡'
        ))
        
        fig.frames = frames
        
        # 添加动画控制
        fig.update_layout(
            updatemenus=[{
                'type': 'buttons',
                'showactive': False,
                'buttons': [
                    {
                        'label': '▶ 播放',
                        'method': 'animate',
                        'args': [None, {
                            'frame': {'duration': 100, 'redraw': True},
                            'fromcurrent': True,
                            'transition': {'duration': 50}
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
                    }
                ]
            }],
            scene=dict(
                xaxis=dict(title='X (m)', range=[-1, 1]),
                yaxis=dict(title='Y (高度 m)', range=[-3, 2]),
                zaxis=dict(title='Z (m)', range=[-0.5, 0.5]),
                aspectmode='manual',
                aspectratio=dict(x=1, y=2.5, z=0.5),
                bgcolor='rgba(1, 5, 10, 0.95)'
            ),
            paper_bgcolor='rgba(1, 5, 10, 0.95)',
            title=dict(
                text='气泡上升模拟',
                font=dict(size=16, color=self.color_schemes['primary'])
            )
        )
        
        return fig
    
    def create_shear_distribution(self,
                                   shear_profile: np.ndarray,
                                   position: np.ndarray,
                                   fiber_length: float = 2.0) -> go.Figure:
        """
        创建剪切力分布图
        
        Args:
            shear_profile: 剪切力分布数组 (Pa)
            position: 位置数组 (归一化 0-1)
            fiber_length: 膜丝长度 (m)
            
        Returns:
            Plotly Figure对象
        """
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('剪切力沿膜丝分布', '剪切力分布直方图'),
            row_heights=[0.6, 0.4],
            vertical_spacing=0.15
        )
        
        # 转换为实际位置
        actual_position = position * fiber_length
        
        # 颜色映射 (根据剪切力大小)
        colors = self._get_shear_colors(shear_profile)
        
        # 剪切力分布曲线
        fig.add_trace(go.Scatter(
            x=actual_position,
            y=shear_profile,
            mode='lines',
            line=dict(color=self.color_schemes['primary'], width=2),
            fill='tozeroy',
            fillcolor='rgba(0, 242, 255, 0.2)',
            name='剪切力'
        ), row=1, col=1)
        
        # 添加参考线
        fig.add_hline(y=0.5, line_dash="dash", line_color=self.color_schemes['success'],
                     annotation_text="最小有效剪切", row=1, col=1)
        fig.add_hline(y=1.5, line_dash="dash", line_color=self.color_schemes['warn'],
                     annotation_text="理想上限", row=1, col=1)
        fig.add_hline(y=2.5, line_dash="dash", line_color=self.color_schemes['danger'],
                     annotation_text="过高剪切", row=1, col=1)
        
        # 直方图
        fig.add_trace(go.Histogram(
            x=shear_profile,
            nbinsx=30,
            marker_color=self.color_schemes['primary'],
            opacity=0.7,
            name='分布'
        ), row=2, col=1)
        
        fig.update_layout(
            title=dict(
                text='膜丝剪切力分析',
                font=dict(size=16, color=self.color_schemes['primary'])
            ),
            xaxis_title='膜丝位置 (m)',
            yaxis_title='剪切力 (Pa)',
            xaxis2_title='剪切力 (Pa)',
            yaxis2_title='频数',
            paper_bgcolor='rgba(1, 5, 10, 0.95)',
            plot_bgcolor='rgba(1, 5, 10, 0.95)',
            font=dict(color='white'),
            showlegend=False
        )
        
        return fig
    
    def _get_shear_colors(self, shear_values: np.ndarray) -> List[str]:
        """根据剪切力值获取颜色"""
        colors = []
        for val in shear_values:
            if val < 0.5:
                colors.append(self.color_schemes['danger'])
            elif val < 1.5:
                colors.append(self.color_schemes['success'])
            elif val < 2.5:
                colors.append(self.color_schemes['warn'])
            else:
                colors.append(self.color_schemes['danger'])
        return colors
    
    def create_performance_dashboard(self, result) -> go.Figure:
        """
        创建性能仪表盘
        
        Args:
            result: CalculationResult对象
            
        Returns:
            Plotly Figure对象
        """
        fig = make_subplots(
            rows=2, cols=3,
            specs=[
                [{'type': 'indicator'}, {'type': 'indicator'}, {'type': 'indicator'}],
                [{'type': 'bar'}, {'type': 'bar'}, {'type': 'bar'}]
            ],
            subplot_titles=('运行评分', '优化评分', '能耗等级',
                          '关键参数对比', '污染指标', '经济性分析'),
            vertical_spacing=0.2
        )
        
        # 运行评分仪表盘
        fig.add_trace(go.Indicator(
            mode="gauge+number+delta",
            value=result.operation_score,
            domain={'row': 0, 'column': 0},
            title={'text': "运行评分"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': self._score_color(result.operation_score)},
                'steps': [
                    {'range': [0, 50], 'color': 'rgba(255, 68, 68, 0.3)'},
                    {'range': [50, 75], 'color': 'rgba(255, 153, 0, 0.3)'},
                    {'range': [75, 100], 'color': 'rgba(0, 255, 136, 0.3)'}
                ],
                'threshold': {
                    'line': {'color': "white", 'width': 4},
                    'thickness': 0.75,
                    'value': 75
                }
            }
        ), row=1, col=1)
        
        # 优化评分仪表盘
        fig.add_trace(go.Indicator(
            mode="gauge+number",
            value=result.optimization_score,
            domain={'row': 0, 'column': 1},
            title={'text': "优化评分"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': self._score_color(result.optimization_score)},
                'steps': [
                    {'range': [0, 50], 'color': 'rgba(255, 68, 68, 0.3)'},
                    {'range': [50, 75], 'color': 'rgba(255, 153, 0, 0.3)'},
                    {'range': [75, 100], 'color': 'rgba(0, 255, 136, 0.3)'}
                ]
            }
        ), row=1, col=2)
        
        # 能耗等级仪表盘
        sec = result.sec_kwh_m3
        sec_normalized = min(100, sec / 1.2 * 100)  # 1.2 kWh/m³为最大值
        fig.add_trace(go.Indicator(
            mode="gauge+number",
            value=100 - sec_normalized,  # 反转：低能耗=高分
            domain={'row': 0, 'column': 2},
            title={'text': "能效评分"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': self._score_color(100 - sec_normalized)},
                'steps': [
                    {'range': [0, 50], 'color': 'rgba(255, 68, 68, 0.3)'},
                    {'range': [50, 75], 'color': 'rgba(255, 153, 0, 0.3)'},
                    {'range': [75, 100], 'color': 'rgba(0, 255, 136, 0.3)'}
                ]
            }
        ), row=1, col=3)
        
        # 关键参数对比
        fig.add_trace(go.Bar(
            x=['实际通量', '临界通量', '平均剪切', '最大剪切'],
            y=[result.target_flux_lmh if hasattr(result, 'target_flux_lmh') else 20,
               result.critical_flux_lmh,
               result.avg_shear_pa,
               result.max_shear_pa],
            marker_color=[self.color_schemes['primary'], 
                         self.color_schemes['success'],
                         self.color_schemes['warn'],
                         self.color_schemes['danger']],
            text=[f'{v:.1f}' for v in [
                result.target_flux_lmh if hasattr(result, 'target_flux_lmh') else 20,
                result.critical_flux_lmh,
                result.avg_shear_pa,
                result.max_shear_pa
            ]],
            textposition='outside'
        ), row=2, col=1)
        
        # 污染指标
        fouling_data = {
            '污染率 (mbar/h)': result.fouling_rate_gap_h,
            '清洗周期 (天)': result.cleaning_frequency_days,
            '膜寿命 (年)': result.membrane_lifetime_years
        }
        fig.add_trace(go.Bar(
            x=list(fouling_data.keys()),
            y=list(fouling_data.values()),
            marker_color=[self.color_schemes['danger'],
                         self.color_schemes['warn'],
                         self.color_schemes['success']],
            text=[f'{v:.1f}' for v in fouling_data.values()],
            textposition='outside'
        ), row=2, col=2)
        
        # 经济性分析
        cost_data = {
            '年能耗 (MWh)': result.annual_energy_kwh / 1000,
            '年费用 (万元)': result.energy_cost_annual_rmb / 10000,
            'SEC (kWh/m³)': result.sec_kwh_m3
        }
        fig.add_trace(go.Bar(
            x=list(cost_data.keys()),
            y=list(cost_data.values()),
            marker_color=[self.color_schemes['primary'],
                         self.color_schemes['warn'],
                         self.color_schemes['success']],
            text=[f'{v:.2f}' for v in cost_data.values()],
            textposition='outside'
        ), row=2, col=3)
        
        fig.update_layout(
            title=dict(
                text='MBR系统性能仪表盘',
                font=dict(size=18, color=self.color_schemes['primary'])
            ),
            paper_bgcolor='rgba(1, 5, 10, 0.95)',
            plot_bgcolor='rgba(1, 5, 10, 0.95)',
            font=dict(color='white'),
            showlegend=False,
            height=700
        )
        
        return fig
    
    def _score_color(self, score: float) -> str:
        """根据评分获取颜色"""
        if score >= 75:
            return self.color_schemes['success']
        elif score >= 50:
            return self.color_schemes['warn']
        else:
            return self.color_schemes['danger']
    
    def create_sludge_distribution(self,
                                    mlss: float = 8000,
                                    sludge_level: float = 0.3,
                                    tank_width: float = 1.65,
                                    tank_length: float = 3.0,
                                    particle_count: int = 500) -> go.Figure:
        """
        创建污泥分布3D图
        
        Args:
            mlss: MLSS浓度 (mg/L)
            sludge_level: 污泥层高度 (m)
            tank_width: 水箱宽度 (m)
            tank_length: 水箱长度 (m)
            particle_count: 粒子数量
            
        Returns:
            Plotly Figure对象
        """
        fig = go.Figure()
        
        np.random.seed(42)
        
        # 生成污泥粒子位置
        half_width = tank_width / 2
        half_length = tank_length / 2
        bottom = -1.5
        
        # 活性污泥粒子
        active_ratio = 0.4 + (mlss / 15000) * 0.5
        active_count = int(particle_count * active_ratio)
        
        x = (np.random.random(active_count) - 0.5) * 2 * half_width
        z = (np.random.random(active_count) - 0.5) * 0.3
        
        # 垂直分布 (底部浓度高)
        y_raw = np.random.random(active_count)
        y = bottom + y_raw * y_raw * (half_length * 2)
        
        # 粒子大小 (与浓度相关)
        sizes = 2 + np.random.random(active_count) * 3
        
        # 颜色 ( settled vs suspended)
        settled = y < (bottom + sludge_level)
        colors = [self.color_schemes['sludge'] if s else '#c8a050' 
                 for s in settled]
        
        fig.add_trace(go.Scatter3d(
            x=x, y=y, z=z,
            mode='markers',
            marker=dict(
                size=sizes,
                color=colors,
                opacity=0.6
            ),
            name='污泥粒子'
        ))
        
        # 污泥层
        if sludge_level > 0.05:
            x_layer = np.linspace(-half_width, half_width, 10)
            z_layer = np.linspace(-0.2, 0.2, 10)
            X, Z = np.meshgrid(x_layer, z_layer)
            Y = np.full_like(X, bottom + sludge_level)
            
            fig.add_trace(go.Surface(
                x=X, y=Y, z=Z,
                colorscale=[[0, 'rgba(139, 90, 43, 0.4)'], 
                           [1, 'rgba(139, 90, 43, 0.4)']],
                showscale=False,
                name='污泥层',
                hoverinfo='skip'
            ))
        
        fig.update_layout(
            scene=dict(
                xaxis=dict(title='X (m)', range=[-1, 1]),
                yaxis=dict(title='Y (高度 m)', range=[-2, 2]),
                zaxis=dict(title='Z (m)', range=[-0.5, 0.5]),
                aspectmode='manual',
                aspectratio=dict(x=1, y=2, z=0.3),
                bgcolor='rgba(1, 5, 10, 0.95)'
            ),
            paper_bgcolor='rgba(1, 5, 10, 0.95)',
            title=dict(
                text=f'污泥分布 (MLSS: {mlss} mg/L)',
                font=dict(size=16, color=self.color_schemes['primary'])
            ),
            showlegend=False
        )
        
        return fig
    
    def create_parameter_comparison(self,
                                     scenarios: List[Dict]) -> go.Figure:
        """
        创建多场景参数对比图
        
        Args:
            scenarios: 场景列表，每个场景为字典
            
        Returns:
            Plotly Figure对象
        """
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('能耗对比', '剪切力对比', 
                          '污染风险对比', '膜寿命对比'),
            specs=[[{'type': 'bar'}, {'type': 'bar'}],
                   [{'type': 'bar'}, {'type': 'bar'}]]
        )
        
        names = [s['name'] for s in scenarios]
        colors = [self.color_schemes['primary'], 
                 self.color_schemes['success'],
                 self.color_schemes['warn']]
        
        # 能耗
        secs = [s['sec'] for s in scenarios]
        fig.add_trace(go.Bar(
            x=names, y=secs,
            marker_color=colors[:len(names)],
            text=[f'{v:.3f}' for v in secs],
            textposition='outside',
            name='SEC'
        ), row=1, col=1)
        
        # 剪切力
        shears = [s['avg_shear'] for s in scenarios]
        fig.add_trace(go.Bar(
            x=names, y=shears,
            marker_color=colors[:len(names)],
            text=[f'{v:.2f}' for v in shears],
            textposition='outside',
            name='剪切力'
        ), row=1, col=2)
        
        # 污染风险 (数值化)
        risk_map = {'low': 1, 'medium': 2, 'high': 3, 'critical': 4}
        risks = [risk_map.get(s['risk'], 2) for s in scenarios]
        fig.add_trace(go.Bar(
            x=names, y=risks,
            marker_color=colors[:len(names)],
            text=[s['risk'] for s in scenarios],
            textposition='outside',
            name='风险等级'
        ), row=2, col=1)
        
        # 膜寿命
        lifetimes = [s['lifetime'] for s in scenarios]
        fig.add_trace(go.Bar(
            x=names, y=lifetimes,
            marker_color=colors[:len(names)],
            text=[f'{v:.0f}' for v in lifetimes],
            textposition='outside',
            name='膜寿命(年)'
        ), row=2, col=2)
        
        fig.update_layout(
            title=dict(
                text='多场景参数对比分析',
                font=dict(size=18, color=self.color_schemes['primary'])
            ),
            paper_bgcolor='rgba(1, 5, 10, 0.95)',
            plot_bgcolor='rgba(1, 5, 10, 0.95)',
            font=dict(color='white'),
            showlegend=False,
            height=600
        )
        
        return fig


def create_default_visualizer(config=None) -> MBRVisualizer:
    """工厂函数：创建默认可视化器"""
    return MBRVisualizer(config)


if __name__ == "__main__":
    # 示例使用
    viz = MBRVisualizer()
    
    # 创建膜结构
    fig = viz.create_membrane_structure()
    fig.show()
    
    # 创建剪切力分布
    shear = np.random.random(100) * 2
    pos = np.linspace(0, 1, 100)
    fig2 = viz.create_shear_distribution(shear, pos)
    fig2.show()
