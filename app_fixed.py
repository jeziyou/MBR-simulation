"""
MBR工业仿真系统 - 单文件可视化版本
MBR Simulation System - Single File Visualization Version

功能：
- 工程级MBR计算引擎（CFD、传质、膜污染动力学）
- 统一3D可视化（膜组件 + 气泡 + 污泥）
- 交互式参数控制
- 性能评估与优化建议

依赖：pip install numpy plotly

作者: MBR Engineering Team
版本: 3.0
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum
import json
import warnings
import os

warnings.filterwarnings('ignore')

# 创建输出目录
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# 配置与常量
# ============================================================

class AerationMode(Enum):
    CONTINUOUS = "continuous"
    PULSE = "pulse"
    INTERMITTENT = "intermittent"

class FoulingRisk(Enum):
    VERY_LOW = "very_low"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class SimulationConfig:
    """仿真配置"""
    aeration_intensity: float = 100.0
    aeration_mode: AerationMode = AerationMode.CONTINUOUS
    orifice_diameter_mm: float = 3.0
    pipe_spacing_mm: float = 100.0
    fiber_diameter_mm: float = 1.65
    membrane_thickness_mm: float = 30.0
    sheet_spacing_mm: float = 80.0
    fiber_length_m: float = 2.0
    fiber_slack_pct: float = 1.5
    mlss_mg_l: float = 8000.0
    srt_days: float = 15.0
    settling_rate_m_h: float = 2.5
    return_ratio_pct: float = 100.0
    water_depth_m: float = 4.0
    temperature_c: float = 20.0
    target_flux_lmh: float = 20.0
    operating_tmp_pa: float = 15000
    ph: float = 7.0
    cod_influent_mg_l: float = 300.0
    electricity_price_rmb_kwh: float = 0.6
    membrane_replacement_cost_rmb_m2: float = 80.0

@dataclass
class CalculationResult:
    """计算结果"""
    sec_kwh_m3: float = 0.0
    avg_shear_pa: float = 0.0
    max_shear_pa: float = 0.0
    min_shear_pa: float = 0.0
    shear_uniformity_index: float = 0.0
    bubble_diameter_mm: float = 0.0
    bubble_velocity_ms: float = 0.0
    gas_holdup: float = 0.0
    kla_actual: float = 0.0
    oxygen_transfer_rate: float = 0.0
    critical_flux_lmh: float = 0.0
    fouling_risk: FoulingRisk = FoulingRisk.MEDIUM
    cleaning_frequency_days: float = 0.0
    membrane_lifetime_years: float = 0.0
    cod_removal_efficiency: float = 0.0
    bod_removal_efficiency: float = 0.0
    total_cost_rmb_m3: float = 0.0
    operation_score: float = 0.0
    optimization_score: float = 0.0
    sustainability_score: float = 0.0
    warnings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)

# ============================================================
# 计算引擎
# ============================================================

class MBREngineeringCalculator:
    """MBR工程级计算引擎"""
    
    def __init__(self, config: Optional[SimulationConfig] = None):
        self.config = config or SimulationConfig()
        self._update_constants()
    
    def _update_constants(self):
        T = self.config.temperature_c
        self.water_density = 1000 - 0.0178 * (T - 4)**2
        mu_ref = 1.002e-3
        self.water_viscosity = mu_ref * np.exp(-0.027 * (T - 20))
        self.kinematic_viscosity = self.water_viscosity / self.water_density
        self.surface_tension = 0.0728 * (1 - 0.002 * (T - 20))
        self.oxygen_diffusivity = 2.1e-9 * (1 + 0.02 * (T - 20))
        self.oxygen_saturation = 9.07 * (1 - 0.002 * (T - 20))
    
    def calculate_shear_stress(self) -> Dict[str, float]:
        """剪切力计算"""
        intensity = self.config.aeration_intensity
        n_segments = 50
        y_positions = np.linspace(0, self.config.fiber_length_m, n_segments)
        
        shear_profile = np.zeros(n_segments)
        for i, y in enumerate(y_positions):
            y_norm = y / self.config.fiber_length_m
            mode1 = np.sin(np.pi * y_norm)
            mode2 = np.sin(2 * np.pi * y_norm) * 0.3
            mode3 = np.sin(3 * np.pi * y_norm) * 0.1
            vibration_envelope = mode1 + mode2 + mode3
            intensity_factor = (intensity / 100)**1.2
            slack_factor = 1 + self.config.fiber_slack_pct / 100
            base_shear = 0.5
            shear_profile[i] = base_shear * intensity_factor * vibration_envelope * slack_factor
        
        turbulence = np.random.normal(0, 0.1 * np.mean(shear_profile), n_segments)
        shear_profile = np.abs(shear_profile + turbulence)
        
        tau_avg = np.mean(shear_profile)
        tau_std = np.std(shear_profile)
        cv = tau_std / (tau_avg + 1e-10)
        uniformity = 1 / (1 + cv)
        
        return {
            'tau_avg_pa': tau_avg,
            'tau_max_pa': np.max(shear_profile),
            'tau_min_pa': np.min(shear_profile),
            'uniformity_index': uniformity,
            'profile': shear_profile,
            'positions': y_positions
        }
    
    def calculate_bubble_params(self) -> Dict[str, float]:
        """气泡参数计算"""
        gas_velocity = self.config.aeration_intensity / 3600 / 1000
        orifice_dia = self.config.orifice_diameter_mm / 1000
        
        weber_number = (self.water_density * gas_velocity**2 * orifice_dia) / self.surface_tension
        
        if weber_number < 1:
            bubble_dia = orifice_dia * 1.5
        elif weber_number < 10:
            bubble_dia = orifice_dia * (1.5 + 0.5 * np.log10(weber_number))
        else:
            bubble_dia = 0.028 * orifice_dia * weber_number**0.5
        
        bubble_dia = np.clip(bubble_dia, 0.001, 0.01)
        
        re_bubble = (bubble_dia * 9.81 * (self.water_density - 1.205) * bubble_dia**2 / 
                     (18 * self.water_viscosity))
        
        if re_bubble < 1:
            v_bubble = (9.81 * (self.water_density - 1.205) * bubble_dia**2) / (18 * self.water_viscosity)
        elif re_bubble < 1000:
            v_bubble = 0.2 * np.sqrt(9.81 * bubble_dia * (self.water_density - 1.205) / self.water_density)
        else:
            v_bubble = np.sqrt(3 * 9.81 * bubble_dia * (self.water_density - 1.205) / self.water_density)
        
        gas_holdup = gas_velocity / v_bubble
        gas_holdup = np.clip(gas_holdup, 0, 0.3)
        
        return {
            'bubble_diameter_mm': bubble_dia * 1000,
            'bubble_velocity_ms': v_bubble,
            'gas_holdup': gas_holdup,
            'weber_number': weber_number
        }
    
    def calculate_mass_transfer(self, bubble_params: Dict) -> Dict[str, float]:
        """传质计算"""
        d32 = bubble_params['bubble_diameter_mm'] / 1000
        gas_holdup = bubble_params['gas_holdup']
        
        a_interface = 6 * gas_holdup / d32
        bubble_velocity = 0.25
        contact_time = d32 / bubble_velocity
        kl = 2 * np.sqrt(self.oxygen_diffusivity / (np.pi * contact_time))
        
        kla = kl * a_interface * 3600
        kla_base = np.clip(kla, 5, 30)
        
        temp_factor = 1.024 ** (self.config.temperature_c - 20)
        kla_temp_corrected = kla_base * temp_factor
        
        kla_corrected = kla_temp_corrected * 0.65 * 0.95 * 1.0
        kla_corrected = np.clip(kla_corrected, 2, 50)
        
        do_operating = 2.0
        otr = kla_corrected * (self.oxygen_saturation - do_operating)
        
        return {
            'kla_actual_h1': kla_corrected,
            'otr_mgl_h': otr,
            'oxygen_saturation_mgl': self.oxygen_saturation
        }
    
    def calculate_energy(self) -> Dict[str, float]:
        """能耗计算"""
        water_depth = self.config.water_depth_m
        static_head = self.water_density * 9.81 * water_depth
        
        total_pressure = static_head + 4000
        airflow_rate = self.config.aeration_intensity / 3600 / 1000 * 40
        
        gamma = 1.4
        p1 = 101325
        p2 = p1 + total_pressure
        compressor_work = (gamma / (gamma - 1)) * p1 * airflow_rate * ((p2 / p1)**((gamma - 1) / gamma) - 1)
        
        total_efficiency = 0.65 * 0.92
        blower_power = compressor_work / total_efficiency
        
        permeate_flow = self.config.target_flux_lmh / 1000 / 3600 * 40
        pumping_power = permeate_flow * self.config.operating_tmp_pa / 0.6
        mixing_power = 0.01 * 40
        
        total_power = blower_power + pumping_power + mixing_power
        sec = total_power / 1000 / (permeate_flow * 3600) if permeate_flow > 0 else 0
        
        return {
            'sec_kwh_m3': sec,
            'blower_power_kw': blower_power / 1000,
            'pumping_power_kw': pumping_power / 1000,
            'total_power_kw': total_power / 1000
        }
    
    def calculate_fouling(self, shear: Dict) -> Dict[str, any]:
        """膜污染计算"""
        flux = self.config.target_flux_lmh / 1000 / 3600
        mlss = self.config.mlss_mg_l / 1000
        tau_avg = shear['tau_avg_pa']
        
        # 临界通量
        k_critical = 15.0
        shear_factor = (tau_avg / 1.0)**0.3 if tau_avg > 0 else 0.5
        mlss_factor = (mlss / 8.0)**(-0.15) if mlss > 0 else 1.0
        critical_flux_lmh = k_critical * shear_factor * mlss_factor
        critical_flux_lmh = np.clip(critical_flux_lmh, 10, 50)
        
        # 污染风险
        if mlss < 4000:
            mlss_risk = 0.5
        elif mlss < 8000:
            mlss_risk = 1.0
        elif mlss < 12000:
            mlss_risk = 1.5
        else:
            mlss_risk = 2.0
        
        flux_risk = 0.5 if self.config.target_flux_lmh < 15 else (1.0 if self.config.target_flux_lmh < 25 else 1.5)
        risk_score = mlss_risk * flux_risk
        
        if risk_score < 0.75:
            fouling_risk = FoulingRisk.VERY_LOW
        elif risk_score < 1.25:
            fouling_risk = FoulingRisk.LOW
        elif risk_score < 2.0:
            fouling_risk = FoulingRisk.MEDIUM
        elif risk_score < 3.0:
            fouling_risk = FoulingRisk.HIGH
        else:
            fouling_risk = FoulingRisk.CRITICAL
        
        # 清洗周期
        risk_adjustments = {
            FoulingRisk.VERY_LOW: 60,
            FoulingRisk.LOW: 30,
            FoulingRisk.MEDIUM: 14,
            FoulingRisk.HIGH: 7,
            FoulingRisk.CRITICAL: 3
        }
        cleaning_period = risk_adjustments.get(fouling_risk, 14)
        
        # 膜寿命
        chemical_cleaning_count = 365 / max(cleaning_period, 1)
        if chemical_cleaning_count < 10:
            membrane_life = 8
        elif chemical_cleaning_count < 20:
            membrane_life = 5
        else:
            membrane_life = 3
        
        return {
            'critical_flux_lmh': critical_flux_lmh,
            'fouling_risk': fouling_risk,
            'cleaning_frequency_days': cleaning_period,
            'membrane_lifetime_years': membrane_life
        }
    
    def calculate_treatment_efficiency(self) -> Dict[str, float]:
        """处理效率"""
        hrt = 8
        cod_eff = 1 - np.exp(-0.1 * hrt * (1 + self.config.srt_days / 30))
        bod_eff = 1 - np.exp(-0.15 * hrt * (1 + self.config.srt_days / 20))
        return {
            'cod_efficiency': cod_eff * 100,
            'bod_efficiency': bod_eff * 100
        }
    
    def calculate_economics(self, energy: Dict) -> Dict[str, float]:
        """经济性"""
        design_flow = self.config.target_flux_lmh / 1000 * 40 * 24
        energy_cost = energy['sec_kwh_m3'] * design_flow * 365 * self.config.electricity_price_rmb_kwh
        opex = energy_cost / (design_flow * 365) if design_flow > 0 else 0
        total_cost = opex + 0.1
        return {'total_cost_per_m3': np.clip(total_cost, 0.3, 5.0)}
    
    def _calculate_scores(self, result: CalculationResult) -> None:
        """计算评分"""
        sec = result.sec_kwh_m3
        if sec < 0.3:
            energy_score = 100
        elif sec < 0.5:
            energy_score = 90
        elif sec < 0.8:
            energy_score = 75
        else:
            energy_score = 60
        
        tau = result.avg_shear_pa
        if 0.5 <= tau <= 2.0:
            shear_score = 100
        elif 0.3 <= tau < 0.5 or 2.0 < tau <= 3.0:
            shear_score = 80
        else:
            shear_score = 60
        
        risk_scores = {
            'very_low': 100, 'low': 90, 'medium': 70, 'high': 50, 'critical': 30
        }
        fouling_score = risk_scores.get(result.fouling_risk.value, 50)
        
        result.operation_score = (energy_score * 0.2 + shear_score * 0.2 + 
                                  result.shear_uniformity_index * 100 * 0.15 +
                                  fouling_score * 0.2 + 70 * 0.15 + 85 * 0.1)
        result.optimization_score = max(0, 100 - abs(sec - 0.3) / 0.3 * 50)
        result.sustainability_score = max(0, 100 - sec * 100) * 0.4 + result.membrane_lifetime_years * 10 * 0.3 + 70 * 0.3
    
    def _generate_recommendations(self, result: CalculationResult) -> None:
        """生成建议"""
        recs = []
        if result.sec_kwh_m3 > 0.8:
            recs.append(f"⚡ 能耗偏高 ({result.sec_kwh_m3:.2f} kWh/m³)，建议采用脉冲曝气")
        if result.avg_shear_pa < 0.3:
            recs.append(f"⚠️ 剪切力偏低，建议提高曝气强度")
        if result.fouling_risk in [FoulingRisk.HIGH, FoulingRisk.CRITICAL]:
            recs.append(f"🚨 污染风险高，建议降低通量或提高曝气")
        if not recs:
            recs.append("✅ 当前运行参数良好")
        result.recommendations = recs
    
    def run_full_calculation(self) -> CalculationResult:
        """执行完整计算"""
        result = CalculationResult()
        
        shear = self.calculate_shear_stress()
        result.avg_shear_pa = shear['tau_avg_pa']
        result.max_shear_pa = shear['tau_max_pa']
        result.min_shear_pa = shear['tau_min_pa']
        result.shear_uniformity_index = shear['uniformity_index']
        
        bubble = self.calculate_bubble_params()
        result.bubble_diameter_mm = bubble['bubble_diameter_mm']
        result.bubble_velocity_ms = bubble['bubble_velocity_ms']
        result.gas_holdup = bubble['gas_holdup']
        
        mass_transfer = self.calculate_mass_transfer(bubble)
        result.kla_actual = mass_transfer['kla_actual_h1']
        result.oxygen_transfer_rate = mass_transfer['otr_mgl_h']
        
        fouling = self.calculate_fouling(shear)
        result.critical_flux_lmh = fouling['critical_flux_lmh']
        result.fouling_risk = fouling['fouling_risk']
        result.cleaning_frequency_days = fouling['cleaning_frequency_days']
        result.membrane_lifetime_years = fouling['membrane_lifetime_years']
        
        energy = self.calculate_energy()
        result.sec_kwh_m3 = energy['sec_kwh_m3']
        
        efficiency = self.calculate_treatment_efficiency()
        result.cod_removal_efficiency = efficiency['cod_efficiency']
        result.bod_removal_efficiency = efficiency['bod_efficiency']
        
        economics = self.calculate_economics(energy)
        result.total_cost_rmb_m3 = economics['total_cost_per_m3']
        
        self._calculate_scores(result)
        self._generate_recommendations(result)
        
        return result

# ============================================================
# 3D可视化
# ============================================================

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("警告: plotly未安装，3D可视化功能不可用")
    print("安装命令: pip install plotly")

class MBRVisualizer:
    """MBR 3D可视化器"""
    
    def __init__(self):
        self.colors = {
            'primary': '#00f2ff',
            'secondary': '#0088ff',
            'success': '#00ff88',
            'warn': '#ff9900',
            'danger': '#ff4444',
            'sludge': '#d4a84b',
            'sludge_dark': '#8b6914',
            'fiber': '#e0e8f0',
            'bubble': '#aaddff',
            'water': '#0a1a2e',
            'pipe': '#445566',
            'header': '#F5DEB3',
            'tank_frame': 'rgba(30, 74, 133, 0.5)',
            'background': '#01050a'
        }
    
    def create_unified_3d_scene(self, config: SimulationConfig, result: CalculationResult,
                                 sheet_count: int = 5,
                                 show_fibers: bool = True,
                                 fiber_density: int = 20,
                                 bubble_count: int = 150,
                                 show_bubbles: bool = True,
                                 show_sludge: bool = False,
                                 sludge_count: int = 200):
        """创建统一3D场景"""
        if not PLOTLY_AVAILABLE:
            return None
        
        fig = go.Figure()
        
        sheet_width = 1.25
        sheet_length = config.fiber_length_m
        sheet_spacing = config.sheet_spacing_mm / 1000
        half_length = sheet_length / 2
        tank_depth = sheet_count * sheet_spacing + 0.4
        
        # 1. 膜组件结构
        for i in range(sheet_count):
            z = (i - (sheet_count - 1) / 2) * sheet_spacing
            
            # 集水管
            for y_pos, name in [(half_length, 'top'), (-half_length, 'bottom')]:
                fig.add_trace(go.Scatter3d(
                    x=[-sheet_width/2, sheet_width/2, sheet_width/2, -sheet_width/2, -sheet_width/2],
                    y=[y_pos + 0.06, y_pos + 0.06, y_pos - 0.06, y_pos - 0.06, y_pos + 0.06],
                    z=[z + 0.015, z + 0.015, z - 0.015, z - 0.015, z + 0.015],
                    mode='lines',
                    line=dict(color=self.colors['header'], width=6),
                    name=f'集水管 {name}',
                    showlegend=False
                ))
            
            # 导轨
            for x in [-sheet_width/2 + 0.05, sheet_width/2 - 0.05]:
                y = np.linspace(-half_length, half_length, 30)
                fig.add_trace(go.Scatter3d(
                    x=[x] * len(y), y=y, z=[z] * len(y),
                    mode='lines',
                    line=dict(color=self.colors['fiber'], width=2, dash='dot'),
                    showlegend=False
                ))
            
            # 膜丝
            if show_fibers:
                for j in range(fiber_density):
                    x = (j / (fiber_density - 1)) * sheet_width - sheet_width/2 if fiber_density > 1 else 0
                    y = np.linspace(-half_length, half_length, 60)
                    slack_m = config.fiber_slack_pct / 100 * sheet_length
                    envelope = np.sin(np.pi * (y + half_length) / sheet_length)
                    x_fiber = x + slack_m * envelope * 0.3
                    z_fiber = z + slack_m * envelope * 0.15
                    
                    color_intensity = np.sin(np.pi * (j / fiber_density))
                    r = int(224 + 31 * color_intensity)
                    g = int(232 + 23 * color_intensity)
                    b = int(240 + 15 * color_intensity)
                    
                    fig.add_trace(go.Scatter3d(
                        x=x_fiber, y=y, z=z_fiber,
                        mode='lines',
                        line=dict(color=f'rgb({r}, {g}, {b})', width=1.5),
                        opacity=0.7,
                        showlegend=False
                    ))
        
        # 2. 曝气管
        pipe_count = max(3, int(tank_depth / 0.08))
        pipe_spacing = tank_depth / pipe_count
        pipe_y = -half_length - 0.45
        
        for i in range(pipe_count):
            z = -tank_depth / 2 + i * pipe_spacing
            x = np.linspace(-sheet_width/2 - 0.15, sheet_width/2 + 0.15, 30)
            fig.add_trace(go.Scatter3d(
                x=x, y=[pipe_y] * len(x), z=[z] * len(x),
                mode='lines',
                line=dict(color=self.colors['pipe'], width=8),
                showlegend=False
            ))
            
            for j in range(6):
                x_orifice = -sheet_width/2 + j * sheet_width / 5
                fig.add_trace(go.Scatter3d(
                    x=[x_orifice], y=[pipe_y], z=[z],
                    mode='markers',
                    marker=dict(size=5, color=self.colors['primary']),
                    showlegend=False
                ))
        
        # 3. 水箱框架
        tank_width = sheet_width + 0.4
        tank_length = sheet_length + 1.0
        half_w = tank_width / 2
        half_l = tank_length / 2
        half_d = tank_depth / 2
        
        edges = [
            ([-half_w, half_w], [-half_l, -half_l], [-half_d, -half_d]),
            ([half_w, half_w], [-half_l, half_l], [-half_d, -half_d]),
            ([half_w, -half_w], [half_l, half_l], [-half_d, -half_d]),
            ([-half_w, -half_w], [half_l, -half_l], [-half_d, -half_d]),
            ([-half_w, half_w], [-half_l, -half_l], [half_d, half_d]),
            ([half_w, half_w], [-half_l, half_l], [half_d, half_d]),
            ([half_w, -half_w], [half_l, half_l], [half_d, half_d]),
            ([-half_w, -half_w], [half_l, -half_l], [half_d, half_d]),
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
                showlegend=False
            ))
        
        # 4. 气泡
        if show_bubbles:
            np.random.seed(42)
            half_w_bubble = sheet_width / 2 + 0.1
            
            bubble_x = (np.random.random(bubble_count) - 0.5) * 2 * half_w_bubble
            bubble_z = (np.random.random(bubble_count) - 0.5) * tank_depth * 0.8
            pipe_y_bubble = -half_length - 0.45
            bubble_y = pipe_y_bubble + np.random.random(bubble_count) * (sheet_length + 0.8)
            
            base_size = 2.0 + config.aeration_intensity / 50
            bubble_size = np.random.exponential(1.0, bubble_count) + base_size * 0.5
            bubble_size = np.clip(bubble_size, 1.0, 6.0)
            
            colors_bubble = []
            for size in bubble_size:
                if size < 2:
                    colors_bubble.append('rgba(200, 240, 255, 0.7)')
                elif size < 3.5:
                    colors_bubble.append('rgba(150, 220, 255, 0.6)')
                else:
                    colors_bubble.append('rgba(100, 200, 255, 0.5)')
            
            # 气泡轨迹
            for i in range(min(bubble_count, 80)):
                y_start = bubble_y[i]
                y_end = min(y_start + 0.5 + np.random.random() * 0.5, half_length + 0.5)
                y_traj = np.linspace(y_start, y_end, 15)
                x_traj = bubble_x[i] + 0.02 * np.sin(y_traj * 5) * (y_traj - y_start)
                z_traj = [bubble_z[i]] * len(y_traj)
                
                fig.add_trace(go.Scatter3d(
                    x=x_traj, y=y_traj, z=z_traj,
                    mode='lines',
                    line=dict(color='rgba(170, 221, 255, 0.3)', width=1),
                    opacity=0.3,
                    showlegend=False,
                    hoverinfo='skip'
                ))
            
            # 气泡主体
            fig.add_trace(go.Scatter3d(
                x=bubble_x, y=bubble_y, z=bubble_z,
                mode='markers',
                marker=dict(size=bubble_size * 2, color=colors_bubble, opacity=0.7,
                           symbol='circle', line=dict(color='rgba(255, 255, 255, 0.4)', width=1)),
                name='气泡'
            ))
            
            # 气泡光晕
            fig.add_trace(go.Scatter3d(
                x=bubble_x, y=bubble_y, z=bubble_z,
                mode='markers',
                marker=dict(size=bubble_size * 3.5, color='rgba(200, 240, 255, 0.15)', opacity=0.3),
                name='气泡光晕',
                showlegend=False,
                hoverinfo='skip'
            ))
        
        # 5. 污泥
        if show_sludge:
            np.random.seed(123)
            half_w_sludge = sheet_width / 2 + 0.15
            
            sludge_x = (np.random.random(sludge_count) - 0.5) * 2 * half_w_sludge
            sludge_z = (np.random.random(sludge_count) - 0.5) * tank_depth * 0.9
            y_raw = np.random.random(sludge_count)
            sludge_y = -half_length - 0.3 + y_raw**1.5 * (sheet_length + 0.6)
            sludge_size = np.random.random(sludge_count) * 2.5 + 1.5
            
            colors_sludge = []
            for y in sludge_y:
                y_norm = (y - (-half_length - 0.3)) / (sheet_length + 0.6)
                colors_sludge.append(self.colors['sludge_dark'] if y_norm < 0.3 else self.colors['sludge'])
            
            fig.add_trace(go.Scatter3d(
                x=sludge_x, y=sludge_y, z=sludge_z,
                mode='markers',
                marker=dict(size=sludge_size, color=colors_sludge, opacity=0.55),
                name='污泥粒子'
            ))
        
        # 布局
        fig.update_layout(
            scene=dict(
                xaxis=dict(title='X (m)', range=[-1.5, 1.5],
                          backgroundcolor=self.colors['background'],
                          gridcolor='rgba(0, 242, 255, 0.08)'),
                yaxis=dict(title='Y (高度 m)', range=[-3, 3],
                          backgroundcolor=self.colors['background'],
                          gridcolor='rgba(0, 242, 255, 0.08)'),
                zaxis=dict(title='Z (m)', range=[-0.6, 0.6],
                          backgroundcolor=self.colors['background'],
                          gridcolor='rgba(0, 242, 255, 0.08)'),
                aspectmode='manual',
                aspectratio=dict(x=1.5, y=3, z=0.6),
                camera=dict(eye=dict(x=2.8, y=2.0, z=1.8)),
                bgcolor=self.colors['background']
            ),
            title=dict(
                text='<b>MBR膜组件统一3D场景</b><br><sup>膜组件 + 气泡 + 污泥</sup>',
                font=dict(size=18, color=self.colors['primary'])
            ),
            paper_bgcolor=self.colors['background'],
            showlegend=True,
            legend=dict(font=dict(color='white', size=10), bgcolor='rgba(0,0,0,0.3)'),
            margin=dict(l=0, r=0, t=80, b=0)
        )
        
        return fig
    
    def create_shear_profile_chart(self, shear: Dict):
        """剪切力分布图"""
        if not PLOTLY_AVAILABLE:
            return None
        
        fig = make_subplots(rows=1, cols=2, subplot_titles=('沿膜丝分布', '分布直方图'))
        
        fig.add_trace(go.Scatter(
            x=shear['positions'], y=shear['profile'],
            mode='lines',
            line=dict(color='#00f2ff', width=2),
            fill='tozeroy',
            fillcolor='rgba(0, 242, 255, 0.1)',
            name='剪切应力'
        ), row=1, col=1)
        
        fig.add_trace(go.Histogram(
            x=shear['profile'],
            nbinsx=20,
            marker_color='rgba(0, 242, 255, 0.6)',
            name='分布'
        ), row=1, col=2)
        
        fig.update_layout(
            title='剪切力分析',
            paper_bgcolor='#01050a',
            plot_bgcolor='#01050a',
            font=dict(color='white')
        )
        
        return fig
    
    def create_performance_gauge(self, result: CalculationResult):
        """性能仪表盘"""
        if not PLOTLY_AVAILABLE:
            return None
        
        fig = make_subplots(
            rows=2, cols=3,
            specs=[[{'type': 'indicator'}, {'type': 'indicator'}, {'type': 'indicator'}],
                   [{'type': 'indicator'}, {'type': 'indicator'}, {'type': 'indicator'}]],
            subplot_titles=('运行评分', '优化评分', '可持续评分', 'SEC', '剪切力', '膜寿命')
        )
        
        indicators = [
            (result.operation_score, 100, '运行评分', '#00f2ff', 1, 1),
            (result.optimization_score, 100, '优化评分', '#00ff88', 1, 2),
            (result.sustainability_score, 100, '可持续评分', '#0088ff', 1, 3),
            (result.sec_kwh_m3, 2, 'SEC kWh/m³', '#ff9900', 2, 1),
            (result.avg_shear_pa, 5, '剪切力 Pa', '#ff4444', 2, 2),
            (result.membrane_lifetime_years, 10, '膜寿命 年', '#d4a84b', 2, 3)
        ]
        
        for value, max_val, title, color, row, col in indicators:
            fig.add_trace(go.Indicator(
                mode="gauge+number",
                value=value,
                title={'text': title, 'font': {'color': 'white', 'size': 14}},
                gauge={
                    'axis': {'range': [0, max_val], 'tickcolor': 'white'},
                    'bar': {'color': color},
                    'bgcolor': 'rgba(0,0,0,0.3)',
                    'bordercolor': color,
                    'steps': [
                        {'range': [0, max_val*0.3], 'color': 'rgba(255,0,0,0.1)'},
                        {'range': [max_val*0.3, max_val*0.7], 'color': 'rgba(255,255,0,0.1)'},
                        {'range': [max_val*0.7, max_val], 'color': 'rgba(0,255,0,0.1)'}
                    ]
                }
            ), row=row, col=col)
        
        fig.update_layout(
            paper_bgcolor='#01050a',
            font=dict(color='white'),
            height=500
        )
        
        return fig

# ============================================================
# 主程序 - 独立运行
# ============================================================

def main():
    """主函数 - 执行计算并生成可视化"""
    print("=" * 60)
    print("MBR工业仿真系统 V3.0 - 单文件可视化版本")
    print("=" * 60)
    
    # 1. 创建配置
    config = SimulationConfig()
    print(f"\n📋 当前配置:")
    print(f"   曝气强度: {config.aeration_intensity} Nm³/m²/h")
    print(f"   MLSS: {config.mlss_mg_l} mg/L")
    print(f"   目标通量: {config.target_flux_lmh} LMH")
    print(f"   水温: {config.temperature_c}°C")
    
    # 2. 执行计算
    print(f"\n🔬 执行工程计算...")
    calc = MBREngineeringCalculator(config)
    result = calc.run_full_calculation()
    
    # 3. 显示结果
    print(f"\n📊 计算结果:")
    print(f"   SEC: {result.sec_kwh_m3:.3f} kWh/m³")
    print(f"   平均剪切力: {result.avg_shear_pa:.3f} Pa")
    print(f"   临界通量: {result.critical_flux_lmh:.1f} LMH")
    print(f"   污染风险: {result.fouling_risk.value}")
    print(f"   清洗周期: {result.cleaning_frequency_days:.0f} 天")
    print(f"   膜寿命: {result.membrane_lifetime_years} 年")
    print(f"   KLa: {result.kla_actual:.1f} h⁻¹")
    print(f"   总成本: ¥{result.total_cost_rmb_m3:.2f}/m³")
    print(f"\n⭐ 评分:")
    print(f"   运行评分: {result.operation_score:.0f}/100")
    print(f"   优化评分: {result.optimization_score:.0f}/100")
    print(f"   可持续评分: {result.sustainability_score:.0f}/100")
    
    print(f"\n💡 建议:")
    for rec in result.recommendations:
        print(f"   {rec}")
    
    # 4. 生成可视化
    if PLOTLY_AVAILABLE:
        print(f"\n🎨 生成3D可视化...")
        viz = MBRVisualizer()
        
        # 统一3D场景
        fig_3d = viz.create_unified_3d_scene(config, result)
        if fig_3d:
            fig_3d.write_html(os.path.join(OUTPUT_DIR, 'mbr_3d_scene.html'))
            print(f"✅ 3D场景已保存: output/mbr_3d_scene.html")
        
        # 剪切力分析
        shear = calc.calculate_shear_stress()
        fig_shear = viz.create_shear_profile_chart(shear)
        if fig_shear:
            fig_shear.write_html(os.path.join(OUTPUT_DIR, 'mbr_shear.html'))
            print(f"✅ 剪切力分析已保存: output/mbr_shear.html")
        
        # 性能仪表盘
        fig_gauge = viz.create_performance_gauge(result)
        if fig_gauge:
            fig_gauge.write_html(os.path.join(OUTPUT_DIR, 'mbr_gauge.html'))
            print(f"✅ 性能仪表盘已保存: output/mbr_gauge.html")
        
        print(f"\n📁 可视化文件:")
        print(f"   - output/mbr_3d_scene.html (统一3D场景)")
        print(f"   - output/mbr_shear.html (剪切力分析)")
        print(f"   - output/mbr_gauge.html (性能仪表盘)")
        print(f"\n💡 提示: 在浏览器中打开HTML文件查看交互式3D可视化")
    else:
        print(f"\n⚠️ plotly未安装，跳过可视化生成")
        print(f"   安装命令: pip install plotly")
    
    # 5. 导出JSON报告
    report = {
        'config': {
            'aeration_intensity': config.aeration_intensity,
            'mlss_mg_l': config.mlss_mg_l,
            'target_flux_lmh': config.target_flux_lmh,
            'temperature_c': config.temperature_c
        },
        'results': {
            'sec_kwh_m3': result.sec_kwh_m3,
            'avg_shear_pa': result.avg_shear_pa,
            'critical_flux_lmh': result.critical_flux_lmh,
            'fouling_risk': result.fouling_risk.value,
            'kla_actual': result.kla_actual,
            'operation_score': result.operation_score,
            'total_cost_rmb_m3': result.total_cost_rmb_m3
        },
        'recommendations': result.recommendations
    }
    
    with open(os.path.join(OUTPUT_DIR, 'mbr_report.json'), 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n📄 报告已保存: output/mbr_report.json")
    
    print(f"\n" + "=" * 60)
    print("计算完成!")
    print("=" * 60)
    
    return result

if __name__ == "__main__":
    main()
