"""
MBR仿真系统 - 工程级计算引擎
MBR Simulation System - Engineering-Level Calculation Engine

该模块提供MBR（膜生物反应器）系统的专业工程计算，包括：
- 能耗计算 (SEC - Specific Energy Consumption)
- 剪切力计算 (Shear Stress) 
- 膜污染预测 (Fouling Prediction)
- 传质系数计算 (Mass Transfer)
- 优化建议生成

作者: MBR Engineering Team
版本: 2.0
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum
import warnings


class AerationMode(Enum):
    """曝气模式枚举"""
    CONTINUOUS = "continuous"
    PULSE = "pulse"
    INTERMITTENT = "intermittent"


class FoulingRisk(Enum):
    """污染风险等级"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class PhysicalConstants:
    """物理常数"""
    # 标准条件
    ATM_PRESSURE: float = 101325.0      # 大气压 (Pa)
    WATER_DENSITY: float = 998.2        # 水密度 (kg/m³) at 20°C
    WATER_VISCOSITY: float = 1.002e-6   # 运动粘度 (m²/s) at 20°C
    GRAVITY: float = 9.81               # 重力加速度 (m/s²)
    
    # 空气属性
    AIR_DENSITY: float = 1.225          # 空气密度 (kg/m³)
    AIR_VISCOSITY: float = 1.81e-5     # 空气动力粘度 (Pa·s)
    
    # 膜相关
    WATER_SURFACE_TENSION: float = 0.0728  # 水的表面张力 (N/m) at 20°C
    
    # 传质系数
    OXYGEN_TRANSFER_ALPHA: float = 0.65     # α因子 (污水 vs 清水)
    OXYGEN_TRANSFER_BETA: float = 0.95      # β因子 (盐度校正)
    STANDARD_AERATION_EFFICIENCY: float = 2.0  # SAE (kg O₂/kWh)


@dataclass
class MembraneSpecs:
    """膜组件规格"""
    # 膜丝参数
    fiber_od_mm: float = 1.65           # 膜丝外径 (mm)
    fiber_id_mm: float = 0.85           # 膜丝内径 (mm)
    pore_size_um: float = 0.4           # 膜孔径 (μm)
    
    # 标准组件
    sheet_width_m: float = 1.25         # 膜片宽度 (m)
    sheet_length_m: float = 2.0         # 膜片长度 (m)
    sheets_per_module: int = 5          # 每模块膜片数
    
    # 有效面积计算
    packing_density: float = 0.65       # 填充密度 (膜面积/投影面积)
    porosity: float = 0.4               # 膜丝间孔隙率


@dataclass
class SimulationConfig:
    """仿真配置"""
    # 曝气参数
    aeration_intensity: float = 100.0   # 曝气强度 (Nm³/m²/h)
    aeration_mode: AerationMode = AerationMode.CONTINUOUS
    pulse_period_s: float = 4.0         # 脉冲周期 (s)
    pulse_duty_cycle: float = 0.5       # 占空比
    orifice_diameter_mm: float = 3.0    # 曝气孔径 (mm)
    pipe_spacing_mm: float = 100.0      # 曝气管间距 (mm)
    
    # 膜参数
    fiber_diameter_mm: float = 1.65     # 膜丝外径 (mm)
    membrane_thickness_mm: float = 30.0 # 膜片厚度 (mm)
    sheet_spacing_mm: float = 80.0      # 膜片排列间距 (mm)
    fiber_length_m: float = 2.0         # 膜丝长度 (m)
    fiber_slack_pct: float = 1.5        # 膜丝松弛度 (%)
    
    # 污泥参数
    mlss_mg_l: float = 8000.0           # 混合液悬浮固体浓度 (mg/L)
    srt_days: float = 15.0              # 污泥龄 (d)
    settling_rate_m_h: float = 2.5      # 沉降速率 (m/h)
    return_ratio_pct: float = 100.0     # 污泥回流比 (%)
    
    # 操作条件
    water_depth_m: float = 4.0          # 水深 (m)
    temperature_c: float = 20.0          # 水温 (°C)
    target_flux_lmh: float = 20.0       # 目标通量 (L/m²/h)
    
    @property
    def duty_cycle_effective(self) -> float:
        """有效占空比"""
        if self.aeration_mode == AerationMode.CONTINUOUS:
            return 1.0
        elif self.aeration_mode == AerationMode.PULSE:
            return min(1.0 / self.pulse_period_s, 0.5)
        else:  # INTERMITTENT
            return self.pulse_duty_cycle


@dataclass
class CalculationResult:
    """计算结果"""
    # 能耗相关
    sec_kwh_m3: float = 0.0             # 单位体积能耗 (kWh/m³)
    specific_aeration_power: float = 0.0 # 比曝气功率 (W/m²)
    blower_power_kw: float = 0.0         # 鼓风机功率 (kW)
    annual_energy_kwh: float = 0.0        # 年能耗 (kWh)
    energy_cost_annual_rmb: float = 0.0  # 年能耗费用 (元)
    
    # 剪切力相关
    avg_shear_pa: float = 0.0            # 平均剪切力 (Pa)
    max_shear_pa: float = 0.0           # 最大剪切力 (Pa)
    shear_stress_profile: np.ndarray = field(default_factory=lambda: np.array([]))
    shear_uniformity_index: float = 0.0  # 剪切均匀性指数
    
    # 气泡参数
    bubble_diameter_mm: float = 0.0      # 平均气泡直径 (mm)
    bubble_velocity_ms: float = 0.0      # 气泡上升速度 (m/s)
    bubble_frequency_hz: float = 0.0     # 气泡产生频率 (Hz)
    gas_velocity_ms: float = 0.0        # 表观气速 (m/s)
    
    # 传质相关
    kla_20: float = 0.0                  # 氧传质系数 (1/h) at 20°C
    kla_actual: float = 0.0              # 实际氧传质系数 (1/h)
    oxygen_transfer_rate: float = 0.0    # 氧传递速率 (kg O₂/h)
    do_level_mgl: float = 0.0            # 溶解氧水平 (mg/L)
    
    # 膜污染预测
    fouling_rate_gap_h: float = 0.0      # 污染导致的TMP增长 (mbar/h)
    critical_flux_lmh: float = 0.0       # 临界通量 (L/m²/h)
    fouling_risk: FoulingRisk = FoulingRisk.MEDIUM
    fouling_resistance: float = 0.0      # 污染阻力 (1/m)
    
    # 膜面积与组件
    single_sheet_area_m2: float = 0.0    # 单片膜面积 (m²)
    total_module_area_m2: float = 0.0    # 模块总膜面积 (m²)
    fiber_count_per_sheet: int = 0       # 每片膜丝数
    total_fiber_count: int = 0           # 总膜丝数
    
    # 性能指标
    cleaning_frequency_days: float = 0.0 # 预计清洗周期 (d)
    membrane_lifetime_years: float = 0.0  # 预计膜寿命 (年)
    operation_score: float = 0.0         # 运行评分 (0-100)
    optimization_score: float = 0.0      # 优化评分 (0-100)
    
    # 诊断信息
    warnings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    status_info: Dict[str, str] = field(default_factory=dict)


class MBREngineeringCalculator:
    """
    MBR工程级计算引擎
    
    该类实现专业的MBR系统计算，基于水处理工程原理和实际工程经验公式。
    """
    
    def __init__(self, config: Optional[SimulationConfig] = None):
        """初始化计算引擎"""
        self.config = config or SimulationConfig()
        self.constants = PhysicalConstants()
        self.membrane = MembraneSpecs()
        self._update_constants_for_temperature()
    
    def _update_constants_for_temperature(self):
        """根据配置温度更新物理常数"""
        T = self.config.temperature_c
        
        # 水的运动粘度随温度变化 (20°C参考值)
        if T <= 0 or T > 100:
            warnings.warn(f"水温{T}°C超出有效范围，使用20°C参数")
            return
            
        # 运动粘度拟合公式 (mm²/s)
        mu_water = (1.002 - 0.00142 * (T - 20) + 0.0000032 * (T - 20)**2) * 1e-6
        self.constants.WATER_VISCOSITY = mu_water
        
        # 水密度
        self.constants.WATER_DENSITY = 1000 - 0.0178 * (T - 4)**2
        
        # 表面张力
        self.constants.WATER_SURFACE_TENSION = 0.0728 - 0.00015 * (T - 20)
        
        # 氧传质系数温度校正 (Arrhenius方程)
        self._theta = 1.024  # 温度校正系数
    
    def calculate_membrane_area(self) -> Tuple[float, int, int]:
        """
        计算膜面积和膜丝数量
        
        Returns:
            (单片面积, 每片膜丝数, 总膜丝数)
        """
        fiber_od = self.config.fiber_diameter_mm / 1000  # 转换为m
        fiber_length = self.config.fiber_length_m
        
        # 单根膜丝表面积
        fiber_area = np.pi * fiber_od * fiber_length
        
        # 膜片有效宽度
        effective_width = self.membrane.sheet_width_m * self.membrane.packing_density
        
        # 每片膜丝数量 (简化计算)
        fibers_per_sheet = max(1, int(effective_width / (fiber_od * 1.5)))
        
        # 单片膜面积
        sheet_area = fibers_per_sheet * fiber_area
        
        # 总膜面积
        total_area = sheet_area * self.membrane.sheets_per_module
        
        return sheet_area, fibers_per_sheet, fibers_per_sheet * self.membrane.sheets_per_module
    
    def calculate_bubble_parameters(self) -> Dict[str, float]:
        """
        计算气泡参数
        
        基于曝气孔径和水深计算气泡特性
        """
        d_orifice = self.config.orifice_diameter_mm / 1000  # m
        water_depth = self.config.water_depth_m
        intensity = self.config.aeration_intensity
        
        # 出口雷诺数
        v_exit = intensity / 3600 / 1000  # m/s (曝气强度转换)
        Re = v_exit * d_orifice / self.constants.WATER_VISCOSITY
        
        # 韦伯数判断气泡形成机制
        We = (self.constants.WATER_DENSITY * v_exit**2 * d_orifice) / self.constants.WATER_SURFACE_TENSION
        
        # 气泡直径计算 (经验公式)
        if We < 1.2:
            # 低流量：单孔口气泡
            d_bubble = d_orifice * 1.5 * (1 + 0.5 * np.log10(We + 0.1))
        elif We < 3.5:
            # 中等流量：过渡区
            d_bubble = 2.0 * np.sqrt(self.constants.WATER_SURFACE_TENSION * d_orifice / 
                                     (self.constants.WATER_DENSITY * self.constants.GRAVITY))
        else:
            # 高流量：射流形成
            d_bubble = 0.28 * d_orifice * (We ** 0.5)
        
        # 限制合理范围
        d_bubble = np.clip(d_bubble, 1e-3, 10e-3)  # 1-10mm
        
        # 气泡终端速度 (Stokes/Allen/Newton区域判断)
        d_dimless = d_bubble * 1000  # 转换为mm
        
        if d_dimless < 0.3:
            # Stokes区域
            v_terminal = (self.constants.WATER_DENSITY - self.constants.AIR_DENSITY) * \
                         self.constants.GRAVITY * d_bubble**2 / (18 * self.constants.AIR_VISCOSITY * 100)
        elif d_dimless < 3.5:
            # Allen区域
            v_terminal = 0.2 * ((self.constants.WATER_DENSITY - self.constants.AIR_DENSITY) * \
                              self.constants.GRAVITY * d_bubble**2 / self.constants.AIR_VISCOSITY)**0.5
        else:
            # Newton区域
            v_terminal = 1.74 * np.sqrt((self.constants.WATER_DENSITY - self.constants.AIR_DENSITY) * \
                                       self.constants.GRAVITY * d_bubble / self.constants.WATER_DENSITY)
        
        # 表观气速
        gas_velocity = intensity / 3600 / 1000 * self.config.duty_cycle_effective  # m/s
        
        # 气泡产生频率
        bubble_freq = (v_exit * np.pi * (d_orifice/2)**2) / \
                     (4/3 * np.pi * (d_bubble/2)**3)
        
        return {
            'bubble_diameter_m': d_bubble,
            'bubble_diameter_mm': d_bubble * 1000,
            'terminal_velocity_ms': v_terminal,
            'gas_velocity_ms': gas_velocity,
            'bubble_frequency_hz': bubble_freq,
            'reynolds_number': Re,
            'weber_number': We
        }
    
    def calculate_shear_stress(self, bubble_params: Dict) -> Dict[str, float]:
        """
        计算膜丝剪切力分布
        
        基于气泡-液体相互作用和曝气参数计算剪切应力
        """
        intensity = self.config.aeration_intensity
        fiber_length = self.config.fiber_length_m
        pipe_spacing = self.config.pipe_spacing_mm / 1000  # m
        
        # 表观气速
        gas_velocity = bubble_params['gas_velocity_ms']
        
        # 无量纲剪切数 (Shugoll剪切数)
        # G = sqrt(ρ * g * H) / μ * Qg / A
        water_depth = self.config.water_depth_m
        G = np.sqrt(self.constants.WATER_DENSITY * self.constants.GRAVITY * water_depth) / \
            (self.constants.WATER_VISCOSITY * 1e6)
        
        # 局部剪切应力分布
        x = np.linspace(0, 1, 100)  # 沿膜丝长度归一化位置
        
        # 曝气管间距衰减因子
        decay_factor = np.exp(-x * pipe_spacing * 15)  # 指数衰减
        
        # 曝气强度影响
        intensity_factor = (intensity / 100) ** 1.2
        
        # 占空比影响
        duty_factor = self.config.duty_cycle_effective
        
        # 膜丝松弛度影响
        slack_factor = 1 + self.config.fiber_slack_pct / 100 * 0.5
        
        # 局部剪切应力 (Pa)
        # 基于经验公式和实验数据
        base_shear = 0.15  # 基础剪切力 (Pa)
        tau_local = base_shear * intensity_factor * duty_factor * decay_factor * slack_factor
        
        # 添加湍流脉动
        turbulence = 0.2 * np.sin(2 * np.pi * x * 5) * intensity_factor
        tau_local = np.abs(tau_local + turbulence)
        
        # 膜丝高度变化的影响 (两端固定，中间最大)
        envelope = np.sin(np.pi * x)  # 0-1变化
        tau_local = tau_local * (0.3 + 0.7 * envelope)
        
        # 计算统计值
        tau_avg = np.mean(tau_local)
        tau_max = np.max(tau_local)
        tau_min = np.min(tau_local)
        
        # 剪切均匀性指数 (CV的倒数)
        tau_std = np.std(tau_local)
        uniformity = 1 / (1 + tau_std / (tau_avg + 1e-6))
        
        # 膜丝振动贡献
        # 振动剪切 = 0.5 * ρ * v^2 * Cd
        # v为膜丝振动速度，与曝气强度正相关
        vibration_velocity = intensity_factor * 0.02  # m/s (估计值)
        vibration_shear = 0.5 * self.constants.WATER_DENSITY * vibration_velocity**2 * 0.5
        tau_vibration = vibration_shear * envelope
        
        return {
            'tau_avg_pa': tau_avg,
            'tau_max_pa': tau_max,
            'tau_min_pa': tau_min,
            'tau_std_pa': tau_std,
            'uniformity_index': uniformity,
            'tau_profile': tau_local,
            'tau_vibration_pa': np.mean(tau_vibration),
            'position_profile': x
        }
    
    def calculate_energy_consumption(self, total_area: float, 
                                     bubble_params: Dict) -> Dict[str, float]:
        """
        计算能耗参数
        
        单位体积能耗 (SEC) 计算公式:
        SEC = (P_blower / Q) = (Q_g * ΔP) / (Q * η)
        
        Returns:
            能耗相关参数字典
        """
        intensity = self.config.aeration_intensity
        water_depth = self.config.water_depth_m
        duty_cycle = self.config.duty_cycle_effective
        
        # 曝气量计算
        # Q_g = 曝气强度 * 膜面积 / 1000 (转换为m³/m²/h -> m³/s)
        airflow_rate = intensity / 3600 / 1000 * total_area  # m³/s
        
        # 压力损失
        # 静压 + 管道损失 + 曝气器损失
        static_head = self.constants.WATER_DENSITY * self.constants.GRAVITY * water_depth  # Pa
        
        # 曝气器压力损失 (典型值 3-5 kPa)
        diffuser_loss = 4000  # Pa
        
        # 总压力
        total_pressure = static_head + diffuser_loss  # Pa
        
        # 鼓风机效率 (大型离心风机典型值)
        blower_efficiency = 0.65
        
        # 电机效率
        motor_efficiency = 0.92
        
        # 总效率
        total_efficiency = blower_efficiency * motor_efficiency
        
        # 鼓风机功率
        # P = Q * ΔP / η
        blower_power_w = (airflow_rate * total_pressure) / total_efficiency
        blower_power_kw = blower_power_w / 1000
        
        # 脉冲曝气节能因子
        if self.config.aeration_mode == AerationMode.PULSE:
            # 脉冲曝气可节省 20-40% 能耗
            pulse_savings = 0.3
            blower_power_kw = blower_power_kw * (1 - pulse_savings * duty_cycle)
        
        # 单位膜面积功率 (W/m²)
        specific_power = blower_power_w / total_area
        
        # 单位体积能耗 (kWh/m³)
        # 假设过滤通量 20 L/m²/h
        flux_m3_m2_h = self.config.target_flux_lmh / 1000
        sec_kwh_m3 = blower_power_kw / (flux_m3_m2_h * total_area) if total_area > 0 else 0
        
        # 年能耗计算 (假设年运行8000小时)
        annual_hours = 8000
        annual_energy = blower_power_kw * annual_hours
        
        # 电费 (假设 0.6 元/kWh)
        energy_cost = annual_energy * 0.6
        
        return {
            'sec_kwh_m3': sec_kwh_m3,
            'specific_power_w_m2': specific_power,
            'blower_power_kw': blower_power_kw,
            'blower_power_w': blower_power_w,
            'airflow_rate_m3_s': airflow_rate,
            'total_pressure_pa': total_pressure,
            'annual_energy_kwh': annual_energy,
            'annual_energy_mwh': annual_energy / 1000,
            'annual_cost_rmb': energy_cost,
            'efficiency_total': total_efficiency
        }
    
    def calculate_mass_transfer(self, shear_params: Dict,
                                bubble_params: Dict) -> Dict[str, float]:
        """
        计算氧传质参数
        
        基于双膜理论和曝气参数计算KLa
        """
        T = self.config.temperature_c
        water_depth = self.config.water_depth_m
        
        # 标准条件下清水KLa (典型值 2-10 h⁻¹)
        # 与曝气强度、表观气速相关
        gas_velocity = bubble_params['gas_velocity_ms']
        
        # KLa与表观气速的经验关系 (Clean Water)
        # KLa = a * (Qg/A)^b
        a_coef = 0.5
        b_coef = 0.8
        
        kla_clean = a_coef * (gas_velocity * 1000) ** b_coef
        
        # 曝气强度校正
        intensity_factor = self.config.aeration_intensity / 100
        kla_clean = kla_clean * intensity_factor
        
        # 污水校正 (α因子)
        alpha = self.constants.OXYGEN_TRANSFER_ALPHA
        
        # 盐度/污染物校正 (β因子)
        beta = self.constants.OXYGEN_TRANSFER_BETA
        
        # 曝气器类型校正
        diffuser_factor = 1.0  # 曝气盘/曝气管
        
        # 实际KLa
        kla_20 = kla_clean * alpha * beta * diffuser_factor
        
        # 温度校正 (Arrhenius方程)
        # KLa(T) = KLa(20) * theta^(T-20)
        theta = self._theta
        kla_actual = kla_20 * (theta ** (T - 20))
        
        # 氧饱和浓度 (mg/L)
        # 基于Henry定律和温度
        cs_20 = 9.07  # mg/L at 20°C
        cs_t = cs_20 * (1 + 0.02 * (T - 20))  # 简化校正
        
        # 大气压校正
        pressure_ratio = self.constants.ATM_PRESSURE / 101325.0
        cs_actual = cs_t * pressure_ratio
        
        # 氧传递速率 (OTR)
        # OTR = KLa * (Cs - C)
        # 假设运行DO = 2 mg/L
        do_target = 2.0
        oxygen_deficit = cs_actual - do_target
        otr = kla_actual * oxygen_deficit  # mg/L/h
        
        # 单位面积氧传递速率
        otr_per_area = otr * water_depth  # g O₂/m²/h
        
        return {
            'kla_20_h1': kla_20,
            'kla_actual_h1': kla_actual,
            'oxygen_saturation_mgl': cs_actual,
            'oxygen_transfer_rate_mgl_h': otr,
            'otr_per_m2_h': otr_per_area,
            'do_target_mgl': do_target,
            'alpha_factor': alpha,
            'beta_factor': beta
        }
    
    def calculate_fouling_prediction(self, shear_params: Dict,
                                       mass_transfer: Dict) -> Dict[str, float]:
        """
        预测膜污染趋势
        
        基于过滤理论和运行参数预测TMP增长
        """
        intensity = self.config.aeration_intensity
        flux = self.config.target_flux_lmh
        mlss = self.config.mlss_mg_l
        srt = self.config.srt_days
        
        # 临界通量计算 (Field等人理论)
        # J_c = 0.03 * τ^0.5 * (1 - MLSS/10000)
        shear_sqrt = np.sqrt(shear_params['tau_avg_pa'] + 0.01)
        mlss_factor = 1 - (mlss / 10000) * 0.3  # MLSS影响因子
        
        critical_flux = 0.03 * shear_sqrt * mlss_factor * 20  # L/m²/h
        
        # 操作通量与临界通量比
        flux_ratio = flux / (critical_flux + 1e-6)
        
        # 污染指数
        # 污染率与通量超临界程度正相关
        if flux_ratio < 0.7:
            fouling_rate_base = 0.5  # mbar/h
            risk = FoulingRisk.LOW
        elif flux_ratio < 1.0:
            fouling_rate_base = 1.5
            risk = FoulingRisk.MEDIUM
        elif flux_ratio < 1.3:
            fouling_rate_base = 4.0
            risk = FoulingRisk.HIGH
        else:
            fouling_rate_base = 10.0
            risk = FoulingRisk.CRITICAL
        
        # MLSS浓度影响
        mlss_factor = 1 + (mlss - 5000) / 5000 * 0.5
        
        # SRT影响 (长SRT减少 SMP，降低污染)
        srt_factor = 1 - (srt - 20) / 40 * 0.3
        srt_factor = max(0.5, srt_factor)
        
        # 剪切力保护因子
        shear_protection = 1 + shear_params['tau_avg_pa'] / 2
        
        # 综合污染率
        fouling_rate = fouling_rate_base * mlss_factor * srt_factor / shear_protection
        
        # TMP累积 (假设反洗周期)
        cleaning_cycle_h = 48  # h
        tmp_increase = fouling_rate * cleaning_cycle_h  # mbar
        
        # 清洗周期预测
        critical_tmp = 50  # mbar (临界跨膜压差)
        cleaning_days = critical_tmp / (fouling_rate * 24) if fouling_rate > 0 else 365
        
        # 膜寿命预测
        # 基于化学清洗次数 (每年8-12次为正常)
        chemical_cleaning_freq = 365 / cleaning_days if cleaning_days < 365 else 1
        if chemical_cleaning_freq < 10:
            membrane_lifetime = 8  # 年
        elif chemical_cleaning_freq < 15:
            membrane_lifetime = 5
        else:
            membrane_lifetime = 3
        
        # 总污染阻力
        fouling_resistance = fouling_rate * 1e11  # 1/m
        
        return {
            'critical_flux_lmh': critical_flux,
            'flux_ratio': flux_ratio,
            'fouling_rate_mbar_h': fouling_rate,
            'tmp_increase_mbar': tmp_increase,
            'cleaning_cycle_days': cleaning_days,
            'membrane_lifetime_years': membrane_lifetime,
            'fouling_resistance_1_m': fouling_resistance,
            'risk_level': risk,
            'mlss_factor': mlss_factor,
            'srt_factor': srt_factor
        }
    
    def calculate_operation_score(self, energy: Dict, shear: Dict,
                                  fouling: Dict) -> float:
        """
        计算运行综合评分 (0-100)
        
        综合考虑能效、剪切控制和污染控制
        """
        # 能耗评分 (SEC < 0.5 kWh/m³ 为满分)
        sec = energy['sec_kwh_m3']
        if sec < 0.3:
            energy_score = 100
        elif sec < 0.5:
            energy_score = 90
        elif sec < 0.8:
            energy_score = 75
        elif sec < 1.2:
            energy_score = 60
        else:
            energy_score = 40
        
        # 剪切评分 (0.5-1.5 Pa 为最佳范围)
        tau_avg = shear['tau_avg_pa']
        if 0.5 <= tau_avg <= 1.5:
            shear_score = 100
        elif 0.3 <= tau_avg < 0.5 or 1.5 < tau_avg <= 2.0:
            shear_score = 85
        elif 0.2 <= tau_avg < 0.3 or 2.0 < tau_avg <= 3.0:
            shear_score = 70
        else:
            shear_score = 50
        
        # 剪切均匀性评分
        uniformity = shear['uniformity_index']
        uniformity_score = uniformity * 100
        
        # 污染风险评分
        risk = fouling['risk_level']
        if risk == FoulingRisk.LOW:
            fouling_score = 100
        elif risk == FoulingRisk.MEDIUM:
            fouling_score = 75
        elif risk == FoulingRisk.HIGH:
            fouling_score = 50
        else:
            fouling_score = 25
        
        # 清洗周期评分
        cleaning_days = fouling['cleaning_cycle_days']
        if cleaning_days >= 60:
            cleaning_score = 100
        elif cleaning_days >= 30:
            cleaning_score = 85
        elif cleaning_days >= 14:
            cleaning_score = 70
        else:
            cleaning_score = 50
        
        # 综合评分 (加权平均)
        weights = {
            'energy': 0.25,
            'shear': 0.25,
            'uniformity': 0.15,
            'fouling': 0.20,
            'cleaning': 0.15
        }
        
        total_score = (
            energy_score * weights['energy'] +
            shear_score * weights['shear'] +
            uniformity_score * weights['uniformity'] +
            fouling_score * weights['fouling'] +
            cleaning_score * weights['cleaning']
        )
        
        return total_score
    
    def generate_recommendations(self, energy: Dict, shear: Dict,
                                 fouling: Dict, mass_transfer: Dict) -> List[str]:
        """
        基于计算结果生成优化建议
        """
        recommendations = []
        
        # 能耗建议
        sec = energy['sec_kwh_m3']
        if sec > 0.8:
            recommendations.append("⚡ 能耗偏高，建议采用脉冲曝气模式，可节能20-30%")
        if self.config.aeration_mode == AerationMode.CONTINUOUS:
            recommendations.append("💡 建议切换至脉冲曝气模式，降低能耗同时保持膜清洁效果")
        
        # 剪切力建议
        tau_avg = shear['tau_avg_pa']
        if tau_avg < 0.3:
            recommendations.append("⚠️ 剪切力偏低(<0.3 Pa)，可能无法有效控制膜污染")
            recommendations.append("💡 建议提高曝气强度至80 Nm³/m²/h以上")
        elif tau_avg > 2.5:
            recommendations.append("⚠️ 剪切力过高(>2.5 Pa)，可能造成膜丝机械损伤")
            recommendations.append("💡 建议降低曝气强度或调整脉冲参数")
        
        # 均匀性建议
        uniformity = shear['uniformity_index']
        if uniformity < 0.7:
            recommendations.append("🎯 剪切均匀性较差，建议调整曝气管间距与膜片间距匹配")
            pipe_spacing = self.config.pipe_spacing_mm
            sheet_spacing = self.config.sheet_spacing_mm
            if abs(pipe_spacing - sheet_spacing) > 30:
                recommendations.append(f"📐 当前曝气管间距({pipe_spacing}mm)与膜片间距({sheet_spacing}mm)不匹配，建议调整为相近值")
        
        # 污染风险建议
        risk = fouling['risk_level']
        if risk in [FoulingRisk.HIGH, FoulingRisk.CRITICAL]:
            recommendations.append("🚨 膜污染风险较高！")
            recommendations.append("💡 措施1: 提高曝气强度增强膜丝振动")
            recommendations.append("💡 措施2: 降低操作通量至临界通量以下")
            recommendations.append("💡 措施3: 调整污泥龄(SRT)改善污泥性质")
        
        flux_ratio = fouling['flux_ratio']
        if flux_ratio > 1.0:
            recommendations.append(f"⚠️ 当前通量({self.config.target_flux_lmh} LMH)超过临界通量({fouling['critical_flux_lmh']:.1f} LMH)")
        
        # 曝气孔径建议
        orifice = self.config.orifice_diameter_mm
        if orifice > 5:
            recommendations.append("🔧 曝气孔径偏大，建议使用2-4mm孔径获得更小气泡")
        elif orifice < 1.5:
            recommendations.append("🔧 曝气孔径偏小，可能导致堵塞风险")
        
        # MLSS建议
        mlss = self.config.mlss_mg_l
        if mlss > 12000:
            recommendations.append("⚠️ MLSS浓度过高(>12000 mg/L)，增加污染风险和能耗")
        elif mlss < 4000:
            recommendations.append("⚠️ MLSS浓度偏低(<4000 mg/L)，生物处理效果可能不足")
        
        # SRT建议
        srt = self.config.srt_days
        if srt < 10:
            recommendations.append("📊 污泥龄偏短，可能影响硝化和污泥性质")
        elif srt > 30:
            recommendations.append("📊 污泥龄偏长，可能增加SMP积累和膜污染")
        
        # 传氧建议
        do = mass_transfer['oxygen_saturation_mgl']
        if self.config.aeration_mode == AerationMode.PULSE:
            recommendations.append("💨 脉冲曝气时注意监测DO波动，确保硝化反应DO需求")
        
        return recommendations
    
    def run_full_calculation(self) -> CalculationResult:
        """
        执行完整工程计算
        
        Returns:
            CalculationResult: 包含所有计算结果的综合对象
        """
        result = CalculationResult()
        
        # 1. 膜面积计算
        sheet_area, fibers_per_sheet, total_fibers = self.calculate_membrane_area()
        result.single_sheet_area_m2 = sheet_area
        result.total_module_area_m2 = sheet_area * self.membrane.sheets_per_module
        result.fiber_count_per_sheet = fibers_per_sheet
        result.total_fiber_count = total_fibers
        
        # 2. 气泡参数
        bubble_params = self.calculate_bubble_parameters()
        result.bubble_diameter_mm = bubble_params['bubble_diameter_mm']
        result.bubble_velocity_ms = bubble_params['terminal_velocity_ms']
        result.bubble_frequency_hz = bubble_params['bubble_frequency_hz']
        result.gas_velocity_ms = bubble_params['gas_velocity_ms']
        
        # 3. 剪切力计算
        shear_params = self.calculate_shear_stress(bubble_params)
        result.avg_shear_pa = shear_params['tau_avg_pa']
        result.max_shear_pa = shear_params['tau_max_pa']
        result.shear_uniformity_index = shear_params['uniformity_index']
        result.shear_stress_profile = shear_params['tau_profile']
        
        # 4. 能耗计算
        energy_params = self.calculate_energy_consumption(
            result.total_module_area_m2, bubble_params
        )
        result.sec_kwh_m3 = energy_params['sec_kwh_m3']
        result.specific_aeration_power = energy_params['specific_power_w_m2']
        result.blower_power_kw = energy_params['blower_power_kw']
        result.annual_energy_kwh = energy_params['annual_energy_kwh']
        result.energy_cost_annual_rmb = energy_params['annual_cost_rmb']
        
        # 5. 传质计算
        mass_transfer_params = self.calculate_mass_transfer(shear_params, bubble_params)
        result.kla_20 = mass_transfer_params['kla_20_h1']
        result.kla_actual = mass_transfer_params['kla_actual_h1']
        result.oxygen_transfer_rate = mass_transfer_params['oxygen_transfer_rate_mgl_h']
        result.do_level_mgl = mass_transfer_params['do_target_mgl']
        
        # 6. 污染预测
        fouling_params = self.calculate_fouling_prediction(shear_params, mass_transfer_params)
        result.critical_flux_lmh = fouling_params['critical_flux_lmh']
        result.fouling_risk = fouling_params['risk_level']
        result.fouling_rate_gap_h = fouling_params['fouling_rate_mbar_h']
        result.fouling_resistance = fouling_params['fouling_resistance_1_m']
        result.cleaning_frequency_days = fouling_params['cleaning_cycle_days']
        result.membrane_lifetime_years = fouling_params['membrane_lifetime_years']
        
        # 7. 运行评分
        result.operation_score = self.calculate_operation_score(
            energy_params, shear_params, fouling_params
        )
        
        # 8. 优化评分
        result.optimization_score = self._calculate_optimization_score(
            energy_params, shear_params, fouling_params
        )
        
        # 9. 生成建议
        result.recommendations = self.generate_recommendations(
            energy_params, shear_params, fouling_params, mass_transfer_params
        )
        
        # 10. 状态信息
        result.status_info = self._get_status_info(
            energy_params, shear_params, fouling_params
        )
        
        # 11. 警告信息
        result.warnings = self._generate_warnings(
            energy_params, shear_params, fouling_params
        )
        
        return result
    
    def _calculate_optimization_score(self, energy: Dict, shear: Dict,
                                       fouling: Dict) -> float:
        """计算优化空间评分"""
        # 计算各参数的优化空间
        sec = energy['sec_kwh_m3']
        energy_gap = max(0, sec - 0.4) / 0.6  # 0.4为理想值
        
        tau_avg = shear['tau_avg_pa']
        tau_gap = abs(tau_avg - 1.0) / 2.0  # 1.0 Pa为理想值
        
        risk = fouling['risk_level']
        fouling_gap = {
            FoulingRisk.LOW: 0,
            FoulingRisk.MEDIUM: 0.3,
            FoulingRisk.HIGH: 0.6,
            FoulingRisk.CRITICAL: 1.0
        }.get(risk, 0.5)
        
        # 综合优化空间
        optimization_gap = (
            energy_gap * 0.4 +
            tau_gap * 0.3 +
            fouling_gap * 0.3
        )
        
        # 评分 = 100 - 优化空间百分比
        return max(0, min(100, 100 - optimization_gap * 50))
    
    def _get_status_info(self, energy: Dict, shear: Dict,
                         fouling: Dict) -> Dict[str, str]:
        """获取状态信息"""
        info = {}
        
        # 曝气状态
        if self.config.aeration_mode == AerationMode.CONTINUOUS:
            info['aeration_mode'] = "连续曝气"
        elif self.config.aeration_mode == AerationMode.PULSE:
            info['aeration_mode'] = f"脉冲曝气 (周期{self.config.pulse_period_s}s)"
        else:
            info['aeration_mode'] = "间歇曝气"
        
        # 能耗等级
        sec = energy['sec_kwh_m3']
        if sec < 0.4:
            info['energy_class'] = "A+ (优秀)"
        elif sec < 0.6:
            info['energy_class'] = "A (良好)"
        elif sec < 0.8:
            info['energy_class'] = "B (中等)"
        else:
            info['energy_class'] = "C (需优化)"
        
        # 剪切状态
        tau = shear['tau_avg_pa']
        if tau < 0.5:
            info['shear_status'] = "偏低"
        elif tau <= 2.0:
            info['shear_status'] = "正常"
        else:
            info['shear_status'] = "偏高"
        
        return info
    
    def _generate_warnings(self, energy: Dict, shear: Dict,
                          fouling: Dict) -> List[str]:
        """生成警告信息"""
        warnings = []
        
        # 高能耗警告
        if energy['sec_kwh_m3'] > 1.0:
            warnings.append(f"⚠️ 能耗偏高: {energy['sec_kwh_m3']:.2f} kWh/m³，建议优化曝气策略")
        
        # 过高剪切警告
        if shear['tau_avg_pa'] > 2.5:
            warnings.append(f"⚠️ 剪切力过高: {shear['tau_avg_pa']:.2f} Pa，可能造成膜损伤")
        
        # 低均匀性警告
        if shear['uniformity_index'] < 0.6:
            warnings.append(f"⚠️ 剪切均匀性差: {shear['uniformity_index']*100:.0f}%，存在局部污染风险")
        
        # 高污染风险警告
        if fouling['risk_level'] in [FoulingRisk.HIGH, FoulingRisk.CRITICAL]:
            warnings.append(f"🚨 污染风险{'严重' if fouling['risk_level']==FoulingRisk.CRITICAL else '较高'}，建议立即调整参数")
        
        # 短清洗周期警告
        if fouling['cleaning_cycle_days'] < 14:
            warnings.append(f"⚠️ 预计清洗周期过短: {fouling['cleaning_cycle_days']:.0f}天，需优化运行条件")
        
        return warnings


def create_calculator(config: Optional[SimulationConfig] = None) -> MBREngineeringCalculator:
    """
    工厂函数：创建MBR计算器实例
    
    Args:
        config: 仿真配置，如果为None则使用默认配置
        
    Returns:
        MBREngineeringCalculator: 配置好的计算器实例
    """
    return MBREngineeringCalculator(config)


def quick_calculate(**kwargs) -> CalculationResult:
    """
    快速计算函数 - 一行代码完成MBR系统计算
    
    Example:
        >>> result = quick_calculate(
        ...     aeration_intensity=100,
        ...     mlss_mg_l=8000,
        ...     target_flux_lmh=20
        ... )
        >>> print(f"SEC: {result.sec_kwh_m3:.3f} kWh/m³")
    """
    config = SimulationConfig(**kwargs)
    calculator = MBREngineeringCalculator(config)
    return calculator.run_full_calculation()


if __name__ == "__main__":
    # 示例计算
    print("=" * 60)
    print("MBR工程计算引擎 - 示例计算")
    print("=" * 60)
    
    # 创建计算器
    calc = MBREngineeringCalculator()
    
    # 修改配置
    calc.config.aeration_intensity = 100
    calc.config.aeration_mode = AerationMode.PULSE
    calc.config.pulse_period_s = 4.0
    calc.config.mlss_mg_l = 8000
    calc.config.target_flux_lmh = 20
    
    # 执行计算
    result = calc.run_full_calculation()
    
    # 输出结果
    print(f"\n📊 膜参数:")
    print(f"   单片膜面积: {result.single_sheet_area_m2:.2f} m²")
    print(f"   模块总膜面积: {result.total_module_area_m2:.2f} m²")
    print(f"   膜丝数量: {result.total_fiber_count}")
    
    print(f"\n⚡ 能耗参数:")
    print(f"   单位体积能耗 SEC: {result.sec_kwh_m3:.3f} kWh/m³")
    print(f"   鼓风机功率: {result.blower_power_kw:.2f} kW")
    print(f"   年能耗: {result.annual_energy_kwh/1000:.1f} MWh")
    print(f"   年费用: ¥{result.energy_cost_annual_rmb:.0f}")
    
    print(f"\n🌊 剪切力参数:")
    print(f"   平均剪切力: {result.avg_shear_pa:.3f} Pa")
    print(f"   最大剪切力: {result.max_shear_pa:.3f} Pa")
    print(f"   均匀性指数: {result.shear_uniformity_index*100:.1f}%")
    
    print(f"\n🫧 气泡参数:")
    print(f"   气泡直径: {result.bubble_diameter_mm:.2f} mm")
    print(f"   上升速度: {result.bubble_velocity_ms:.3f} m/s")
    
    print(f"\n📈 污染预测:")
    print(f"   临界通量: {result.critical_flux_lmh:.1f} LMH")
    print(f"   污染风险: {result.fouling_risk.value}")
    print(f"   预计清洗周期: {result.cleaning_frequency_days:.0f} 天")
    print(f"   预计膜寿命: {result.membrane_lifetime_years} 年")
    
    print(f"\n⭐ 运行评分: {result.operation_score:.1f}/100")
    print(f"   优化评分: {result.optimization_score:.1f}/100")
    
    print(f"\n💡 优化建议:")
    for rec in result.recommendations[:5]:
        print(f"   {rec}")
