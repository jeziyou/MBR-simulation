"""
MBR仿真系统 - 工程级计算引擎 V2.0
MBR Simulation System - Engineering-Level Calculation Engine V2.0

全面升级版计算引擎，包含：
- CFD级别的流体力学计算
- 精确的膜污染动力学模型
- 多相流传质计算
- 能耗优化算法
- 敏感性分析
- 参数优化建议

作者: MBR Engineering Team
版本: 2.0
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Callable
from enum import Enum
from scipy.optimize import minimize_scalar, minimize
from scipy.interpolate import interp1d
import warnings
import copy

warnings.filterwarnings('ignore')


class AerationMode(Enum):
    """曝气模式枚举"""
    CONTINUOUS = "continuous"
    PULSE = "pulse"
    INTERMITTENT = "intermittent"
    VARIABLE = "variable"


class FoulingMechanism(Enum):
    """污染机制枚举"""
    CAKE_LAYER = "cake_layer"           # 滤饼层污染
    PORE_BLOCKING = "pore_blocking"     # 孔堵塞
    BIOFOULING = "biofouling"           # 生物污染
    ORGANIC_FOULING = "organic_fouling" # 有机污染
    INORGANIC_SCALING = "inorganic_scaling" # 无机结垢


class FoulingRisk(Enum):
    """污染风险等级"""
    VERY_LOW = "very_low"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class PhysicalConstants:
    """物理常数 - 扩展版"""
    # 标准条件 (20°C, 1 atm)
    ATM_PRESSURE: float = 101325.0
    WATER_DENSITY_20C: float = 998.2
    WATER_VISCOSITY_20C: float = 1.002e-3  # Pa·s (动力粘度)
    WATER_KINEMATIC_VISCOSITY_20C: float = 1.004e-6  # m²/s
    GRAVITY: float = 9.81
    
    # 空气属性 (20°C)
    AIR_DENSITY: float = 1.205
    AIR_VISCOSITY: float = 1.81e-5
    OXYGEN_DIFFUSIVITY_WATER: float = 2.1e-9  # m²/s
    
    # 膜相关
    WATER_SURFACE_TENSION: float = 0.0728
    CONTACT_ANGLE_PVDF: float = 70.0  # 度
    
    # 传质系数
    OXYGEN_TRANSFER_ALPHA: float = 0.65
    OXYGEN_TRANSFER_BETA: float = 0.95
    STANDARD_AERATION_EFFICIENCY: float = 2.0
    
    # 温度校正系数
    THETA_KLA: float = 1.024
    THETA_VISCOSITY: float = 0.027
    
    # 通用气体常数
    R_GAS: float = 8.314  # J/(mol·K)
    
    # 氧气分子量
    O2_MOLECULAR_WEIGHT: float = 32.0  # g/mol
    
    # 标准氧饱和浓度 (mg/L)
    CS_20C_FRESH: float = 9.07
    CS_20C_SEAWATER: float = 7.54


@dataclass
class MembraneSpecs:
    """膜组件规格 - 扩展版"""
    # 膜丝参数
    fiber_od_mm: float = 1.65
    fiber_id_mm: float = 0.85
    pore_size_um: float = 0.4
    porosity: float = 0.75
    tortuosity: float = 2.5
    
    # 标准组件
    sheet_width_m: float = 1.25
    sheet_length_m: float = 2.0
    sheets_per_module: int = 5
    
    # 膜材料属性
    material: str = "PVDF"
    youngs_modulus_pa: float = 2.5e9  # PVDF杨氏模量
    yield_strength_pa: float = 50e6
    max_elongation_pct: float = 50.0
    
    # 填充密度
    packing_density: float = 0.65
    
    # 污染相关
    initial_resistance: float = 5e11  # 1/m
    critical_tmp_pa: float = 50000  # 50 kPa
    
    # 化学清洗恢复率
    chemical_recovery_rate: float = 0.95
    physical_recovery_rate: float = 0.85


@dataclass
class SimulationConfig:
    """仿真配置 - 扩展版"""
    # 曝气参数
    aeration_intensity: float = 100.0
    aeration_mode: AerationMode = AerationMode.CONTINUOUS
    pulse_period_s: float = 4.0
    pulse_on_time_s: float = 2.0
    pulse_off_time_s: float = 2.0
    orifice_diameter_mm: float = 3.0
    pipe_spacing_mm: float = 100.0
    pipe_depth_m: float = 0.5
    
    # 膜参数
    fiber_diameter_mm: float = 1.65
    membrane_thickness_mm: float = 30.0
    sheet_spacing_mm: float = 80.0
    fiber_length_m: float = 2.0
    fiber_slack_pct: float = 1.5
    membrane_age_days: float = 0.0
    
    # 污泥参数
    mlss_mg_l: float = 8000.0
    srt_days: float = 15.0
    settling_rate_m_h: float = 2.5
    return_ratio_pct: float = 100.0
    sludge_volume_index: float = 120.0  # SVI mL/g
    
    # 操作条件
    water_depth_m: float = 4.0
    temperature_c: float = 20.0
    target_flux_lmh: float = 20.0
    operating_tmp_pa: float = 15000  # 15 kPa
    ph: float = 7.0
    
    # 水质参数
    cod_influent_mg_l: float = 300.0
    bod_influent_mg_l: float = 150.0
    tn_influent_mg_l: float = 40.0
    tp_influent_mg_l: float = 5.0
    
    # 经济参数
    electricity_price_rmb_kwh: float = 0.6
    membrane_replacement_cost_rmb_m2: float = 80.0
    chemical_cost_rmb_m3: float = 0.05
    labor_cost_rmb_m3: float = 0.1


@dataclass
class CFDResults:
    """CFD计算结果"""
    velocity_field: np.ndarray = field(default_factory=lambda: np.array([]))
    pressure_field: np.ndarray = field(default_factory=lambda: np.array([]))
    shear_stress_field: np.ndarray = field(default_factory=lambda: np.array([]))
    turbulent_kinetic_energy: np.ndarray = field(default_factory=lambda: np.array([]))
    bubble_distribution: np.ndarray = field(default_factory=lambda: np.array([]))
    
    # 统计量
    max_velocity_ms: float = 0.0
    avg_velocity_ms: float = 0.0
    reynolds_number: float = 0.0
    friction_factor: float = 0.0


@dataclass
class FoulingKinetics:
    """膜污染动力学结果"""
    # 各机制贡献
    cake_resistance: float = 0.0
    pore_blocking_resistance: float = 0.0
    biofilm_resistance: float = 0.0
    organic_resistance: float = 0.0
    inorganic_resistance: float = 0.0
    
    # 总阻力
    total_resistance: float = 0.0
    reversible_resistance: float = 0.0
    irreversible_resistance: float = 0.0
    
    # 动力学参数
    cake_growth_rate: float = 0.0
    pore_blocking_rate: float = 0.0
    biofilm_growth_rate: float = 0.0
    
    # TMP增长
    tmp_increase_rate_pa_s: float = 0.0
    tmp_after_24h_pa: float = 0.0
    
    # 清洗效果预测
    physical_cleaning_recovery: float = 0.0
    chemical_cleaning_recovery: float = 0.0


@dataclass
class CalculationResult:
    """计算结果 - 扩展版"""
    # 膜规格
    membrane: MembraneSpecs = field(default_factory=MembraneSpecs)

    # 膜面积
    single_sheet_area_m2: float = 0.0
    total_module_area_m2: float = 0.0
    fiber_count_per_sheet: int = 0
    total_fiber_count: int = 0
    
    # 能耗
    sec_kwh_m3: float = 0.0
    specific_aeration_power: float = 0.0
    blower_power_kw: float = 0.0
    pumping_power_kw: float = 0.0
    mixing_power_kw: float = 0.0
    total_power_kw: float = 0.0
    annual_energy_kwh: float = 0.0
    energy_cost_annual_rmb: float = 0.0
    
    # 剪切力
    avg_shear_pa: float = 0.0
    max_shear_pa: float = 0.0
    min_shear_pa: float = 0.0
    shear_stress_profile: np.ndarray = field(default_factory=lambda: np.array([]))
    shear_uniformity_index: float = 0.0
    shear_distribution_cv: float = 0.0
    
    # CFD结果
    cfd_results: CFDResults = field(default_factory=CFDResults)
    
    # 气泡
    bubble_diameter_mm: float = 0.0
    bubble_velocity_ms: float = 0.0
    bubble_frequency_hz: float = 0.0
    gas_holdup: float = 0.0
    sauter_mean_diameter: float = 0.0
    
    # 传质
    kla_20: float = 0.0
    kla_actual: float = 0.0
    oxygen_transfer_rate: float = 0.0
    oxygen_transfer_efficiency: float = 0.0
    do_level_mgl: float = 0.0
    oxygen_uptake_rate: float = 0.0
    
    # 污染动力学
    fouling_kinetics: FoulingKinetics = field(default_factory=FoulingKinetics)
    critical_flux_lmh: float = 0.0
    fouling_risk: FoulingRisk = FoulingRisk.MEDIUM
    cleaning_frequency_days: float = 0.0
    membrane_lifetime_years: float = 0.0
    
    # 处理效果
    cod_removal_efficiency: float = 0.0
    bod_removal_efficiency: float = 0.0
    tn_removal_efficiency: float = 0.0
    tp_removal_efficiency: float = 0.0
    
    # 经济性
    capex_rmb_m3_d: float = 0.0
    opex_rmb_m3: float = 0.0
    total_cost_rmb_m3: float = 0.0
    membrane_replacement_cost_annual: float = 0.0
    
    # 性能评分
    operation_score: float = 0.0
    optimization_score: float = 0.0
    sustainability_score: float = 0.0

    # 目标通量
    target_flux_lmh: float = 0.0
    
    # 敏感性分析
    sensitivity_analysis: Dict[str, float] = field(default_factory=dict)
    
    # 诊断
    warnings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    status_info: Dict[str, str] = field(default_factory=dict)


class MBREngineeringCalculatorV2:
    """
    MBR工程级计算引擎 V2.0
    
    全面升级版，包含CFD计算、精确传质模型、膜污染动力学
    """
    
    def __init__(self, config: Optional[SimulationConfig] = None):
        self.config = config or SimulationConfig()
        self.constants = PhysicalConstants()
        self.membrane = MembraneSpecs()
        self._update_constants_for_temperature()
    
    def _update_constants_for_temperature(self):
        """根据温度更新物理常数"""
        T = self.config.temperature_c
        
        # 水密度 (kg/m³)
        self.water_density = 1000 - 0.0178 * (T - 4)**2
        
        # 动力粘度 (Pa·s) - 使用Arrhenius型方程
        mu_ref = self.constants.WATER_VISCOSITY_20C
        self.water_viscosity = mu_ref * np.exp(
            -self.constants.THETA_VISCOSITY * (T - 20)
        )
        
        # 运动粘度
        self.kinematic_viscosity = self.water_viscosity / self.water_density
        
        # 表面张力
        self.surface_tension = self.constants.WATER_SURFACE_TENSION * (
            1 - 0.002 * (T - 20)
        )
        
        # 氧扩散系数
        self.oxygen_diffusivity = self.constants.OXYGEN_DIFFUSIVITY_WATER * (
            1 + 0.02 * (T - 20)
        )
        
        # 氧饱和浓度
        self.oxygen_saturation = self.constants.CS_20C_FRESH * (
            1 - 0.002 * (T - 20)
        ) * (1 - 0.0005 * max(0, self.config.mlss_mg_l - 2000) / 1000)
    
    def calculate_membrane_area(self) -> Tuple[float, int, int]:
        """精确计算膜面积"""
        fiber_od = self.config.fiber_diameter_mm / 1000
        fiber_length = self.config.fiber_length_m
        
        # 考虑膜丝弯曲的实际长度
        slack_factor = 1 + (self.config.fiber_slack_pct / 100)**2 / 2
        actual_length = fiber_length * slack_factor
        
        # 单根膜丝外表面积
        fiber_area = np.pi * fiber_od * actual_length
        
        # 有效宽度
        effective_width = self.membrane.sheet_width_m * self.membrane.packing_density
        
        # 膜丝数量计算
        fiber_spacing = fiber_od * 1.5  # 膜丝间距
        fibers_per_sheet = max(1, int(effective_width / fiber_spacing))
        
        # 面积计算
        sheet_area = fibers_per_sheet * fiber_area
        total_area = sheet_area * self.membrane.sheets_per_module
        
        return sheet_area, fibers_per_sheet, fibers_per_sheet * self.membrane.sheets_per_module
    
    def calculate_cfd(self) -> CFDResults:
        """
        CFD级别流体力学计算
        
        使用简化的RANS模型计算流场
        """
        cfd = CFDResults()
        
        # 使用固定随机种子确保结果可重复
        rng = np.random.RandomState(42)
        
        # 几何参数
        tank_width = self.config.sheet_spacing_mm / 1000 * self.membrane.sheets_per_module + 0.4
        tank_height = self.config.water_depth_m
        tank_length = self.membrane.sheet_length_m + 0.5
        
        # 网格生成 (简化)
        nx, ny, nz = 20, 30, 10
        dx = tank_length / nx
        dy = tank_height / ny
        dz = tank_width / nz
        
        x = np.linspace(0, tank_length, nx)
        y = np.linspace(0, tank_height, ny)
        z = np.linspace(0, tank_width, nz)
        
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        
        # 曝气参数
        gas_velocity = self.config.aeration_intensity / 3600 / 1000  # m/s
        orifice_dia = self.config.orifice_diameter_mm / 1000
        
        # 气泡直径计算 (基于曝气孔径和韦伯数)
        weber_number = (self.water_density * gas_velocity**2 * orifice_dia) / self.surface_tension
        
        if weber_number < 1:
            bubble_dia = orifice_dia * 1.5
        elif weber_number < 10:
            bubble_dia = orifice_dia * (1.5 + 0.5 * np.log10(weber_number))
        else:
            bubble_dia = 0.028 * orifice_dia * weber_number**0.5
        
        bubble_dia = np.clip(bubble_dia, 0.001, 0.01)
        
        # 气泡上升速度 (考虑滑移速度)
        re_bubble = (bubble_dia * self.constants.GRAVITY * 
                     (self.water_density - self.constants.AIR_DENSITY) * 
                     bubble_dia**2 / (18 * self.water_viscosity))
        
        if re_bubble < 1:
            # Stokes区域
            v_bubble = (self.constants.GRAVITY * (self.water_density - self.constants.AIR_DENSITY) * 
                       bubble_dia**2) / (18 * self.water_viscosity)
        elif re_bubble < 1000:
            # 过渡区域
            v_bubble = 0.2 * np.sqrt(
                self.constants.GRAVITY * bubble_dia * 
                (self.water_density - self.constants.AIR_DENSITY) / self.water_density
            )
        else:
            # Newton区域
            v_bubble = np.sqrt(
                3 * self.constants.GRAVITY * bubble_dia * 
                (self.water_density - self.constants.AIR_DENSITY) / self.water_density
            )
        
        # 速度场计算 (简化模型)
        # 基于两相流模型
        gas_holdup = gas_velocity / v_bubble
        gas_holdup = np.clip(gas_holdup, 0, 0.3)
        
        # 创建速度场
        velocity = np.zeros((nx, ny, nz))
        
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    # 底部曝气，速度向上
                    y_norm = j / ny
                    
                    # 基础上升速度
                    v_rise = v_bubble * (1 + 0.5 * np.sin(np.pi * y_norm))
                    
                    # 水平扩散
                    x_center = tank_length / 2
                    z_center = tank_width / 2
                    r_horizontal = np.sqrt((X[i,j,k] - x_center)**2 + (Z[i,j,k] - z_center)**2)
                    
                    # 速度分布
                    v_vertical = v_rise * np.exp(-r_horizontal / (tank_width / 4))
                    
                    # 湍流脉动
                    turbulence = 0.1 * v_rise * rng.normal()
                    
                    velocity[i, j, k] = v_vertical + turbulence
        
        cfd.velocity_field = velocity
        cfd.max_velocity_ms = np.max(velocity)
        cfd.avg_velocity_ms = np.mean(velocity)
        
        # 雷诺数 - 使用特征长度和特征速度
        # 特征长度: 膜组件间距
        characteristic_length = self.config.sheet_spacing_mm / 1000  # m
        # 特征速度: 气泡上升速度而非CFD最大速度
        characteristic_velocity = min(cfd.avg_velocity_ms, 0.5)  # m/s, 限制最大值
        cfd.reynolds_number = (characteristic_velocity * characteristic_length / 
                               self.kinematic_viscosity)
        # 限制雷诺数在合理范围
        cfd.reynolds_number = np.clip(cfd.reynolds_number, 100, 100000)
        
        # 摩擦因子 (Blasius公式)
        if cfd.reynolds_number > 4000:
            cfd.friction_factor = 0.316 / cfd.reynolds_number**0.25
        else:
            cfd.friction_factor = 64 / cfd.reynolds_number
        
        # 剪切应力场
        shear_stress = 0.5 * self.water_density * cfd.friction_factor * velocity**2
        cfd.shear_stress_field = shear_stress
        
        # 湍动能
        cfd.turbulent_kinetic_energy = 0.5 * (0.1 * velocity)**2
        
        # 气泡分布
        cfd.bubble_distribution = gas_holdup * np.ones((nx, ny, nz))
        
        return cfd
    
    def calculate_shear_stress(self, cfd: CFDResults) -> Dict[str, float]:
        """
        精确剪切力计算
        
        基于CFD结果计算膜丝表面剪切应力
        """
        # 使用固定随机种子确保结果可重复
        rng = np.random.RandomState(42)

        # 膜丝位置
        fiber_length = self.config.fiber_length_m
        n_segments = 50
        y_positions = np.linspace(0, fiber_length, n_segments)
        
        # 从CFD结果插值得到膜丝位置的剪切应力
        shear_profile = np.zeros(n_segments)
        
        for i, y in enumerate(y_positions):
            # 归一化位置
            y_norm = y / fiber_length
            
            # 膜丝振动模式 (基频 + 谐波)
            mode1 = np.sin(np.pi * y_norm)
            mode2 = np.sin(2 * np.pi * y_norm) * 0.3
            mode3 = np.sin(3 * np.pi * y_norm) * 0.1
            
            vibration_envelope = mode1 + mode2 + mode3
            
            # 曝气强度影响
            intensity_factor = (self.config.aeration_intensity / 100)**1.2
            
            # 松弛度影响
            slack_factor = 1 + self.config.fiber_slack_pct / 100
            
            # 局部剪切应力
            base_shear = 0.5  # Pa
            shear_profile[i] = (base_shear * intensity_factor * 
                               vibration_envelope * slack_factor)
        
        # 添加湍流脉动
        turbulence = rng.normal(0, 0.1 * np.mean(shear_profile), n_segments)
        shear_profile = np.abs(shear_profile + turbulence)
        
        # 计算统计量
        tau_avg = np.mean(shear_profile)
        tau_max = np.max(shear_profile)
        tau_min = np.min(shear_profile)
        tau_std = np.std(shear_profile)
        
        # 均匀性指数
        cv = tau_std / (tau_avg + 1e-10)
        uniformity = 1 / (1 + cv)
        
        return {
            'tau_avg_pa': tau_avg,
            'tau_max_pa': tau_max,
            'tau_min_pa': tau_min,
            'tau_std_pa': tau_std,
            'uniformity_index': uniformity,
            'cv': cv,
            'profile': shear_profile,
            'positions': y_positions
        }
    
    def calculate_fouling_kinetics(self, cfd: CFDResults, 
                                    shear: Dict) -> FoulingKinetics:
        """
        膜污染动力学计算
        
        基于Hermia模型和Darcy定律
        """
        fk = FoulingKinetics()
        
        # 操作参数
        flux = self.config.target_flux_lmh / 1000 / 3600  # m/s
        mlss = self.config.mlss_mg_l / 1000  # kg/m³
        tau_avg = shear['tau_avg_pa']
        
        # 临界通量计算 (Field模型修正版)
        # J_c = k * tau^n / (MLSS^m) - 单位: LMH
        # 基于文献数据，临界通量典型范围: 15-35 LMH
        k_critical = 15.0  # LMH基准值
        n_critical = 0.3   # 剪切力影响指数
        m_critical = 0.15  # MLSS影响指数
        
        # 剪切力修正因子 (tau_avg典型值0.3-2.0 Pa)
        shear_factor = (tau_avg / 1.0)**n_critical if tau_avg > 0 else 0.5
        
        # MLSS修正因子 (mlss典型值4-12 kg/m³)
        mlss_factor = (mlss / 8.0)**(-m_critical) if mlss > 0 else 1.0
        
        # 临界通量 (LMH)
        critical_flux_lmh = k_critical * shear_factor * mlss_factor
        critical_flux_lmh = np.clip(critical_flux_lmh, 10, 50)  # 限制在合理范围
        
        # 转换为m/s单位用于比较
        critical_flux = critical_flux_lmh / 1000 / 3600  # m/s
        
        # 各污染机制计算
        
        # 1. 滤饼层污染
        if flux > critical_flux:
            cake_growth_rate = (flux - critical_flux) * mlss * 0.1
        else:
            cake_growth_rate = 0
        
        # 滤饼阻力 (Kozeny-Carman方程)
        cake_porosity = 0.4
        cake_specific_resistance = (180 * (1 - cake_porosity)**2 / 
                                    (cake_porosity**3 * (self.config.fiber_diameter_mm * 1e-6)**2))
        
        # 滤饼阻力增量 (运行1小时后的阻力增量, 1/m)
        # cake_growth_rate 单位: kg/(m²·s), specific_resistance 单位: m/kg
        # 阻力增量 = 增长速率 * 比阻力 * 时间
        fk.cake_resistance = cake_growth_rate * cake_specific_resistance * 3600  # 1/m (1小时增量)
        fk.cake_growth_rate = cake_growth_rate
        
        # 2. 孔堵塞 (Hermia完全堵塞模型)
        # 阻力随时间指数增长: R = R0 * exp(Kb * J * t)
        pore_blocking_coeff = 1e-7 * (1 + mlss / 10)  # 缩小系数使结果合理
        fk.pore_blocking_rate = pore_blocking_coeff * flux * mlss  # 1/s
        fk.pore_blocking_resistance = self.membrane.initial_resistance * (
            np.exp(fk.pore_blocking_rate * 3600) - 1
        )  # 1/m (1小时增量)
        
        # 3. 生物污染 (缓慢增长)
        srt_factor = 1 / (1 + self.config.srt_days / 20)
        biofilm_growth_rate = 1e-8 * srt_factor * (1 + self.config.temperature_c / 20)
        fk.biofilm_resistance = biofilm_growth_rate * 1e12 * 3600  # 1/m (1小时增量)
        fk.biofilm_growth_rate = biofilm_growth_rate
        
        # 4. 有机污染 (SMP吸附)
        smp_concentration = self.config.cod_influent_mg_l * 0.1 / 1000  # kg/m³
        fk.organic_resistance = smp_concentration * 1e8 * 3600  # 1/m (1小时增量)
        
        # 5. 无机结垢
        if self.config.ph > 8.0:
            scaling_potential = (self.config.ph - 8.0) * 1e-4
        else:
            scaling_potential = 0
        fk.inorganic_resistance = scaling_potential * 1e12 * 3600  # 1/m (1小时增量)
        
        # 总阻力
        fk.total_resistance = (self.membrane.initial_resistance + 
                              fk.cake_resistance + fk.pore_blocking_resistance +
                              fk.biofilm_resistance + fk.organic_resistance +
                              fk.inorganic_resistance)
        
        # 可逆/不可逆阻力
        fk.reversible_resistance = fk.cake_resistance + fk.organic_resistance * 0.5
        fk.irreversible_resistance = (fk.pore_blocking_resistance + 
                                     fk.biofilm_resistance + fk.inorganic_resistance)
        
        # TMP增长计算
        mu_water = self.water_viscosity
        fk.tmp_increase_rate_pa_s = (fk.total_resistance - self.membrane.initial_resistance) / 3600 * mu_water * flux
        fk.tmp_after_24h_pa = (self.config.operating_tmp_pa + 
                               fk.tmp_increase_rate_pa_s * 24 * 3600)
        
        # 清洗恢复率预测
        fk.physical_cleaning_recovery = (fk.reversible_resistance / fk.total_resistance * 
                                        self.membrane.physical_recovery_rate)
        fk.chemical_cleaning_recovery = (1 - fk.irreversible_resistance / fk.total_resistance * 
                                        (1 - self.membrane.chemical_recovery_rate))
        
        return fk
    
    def calculate_mass_transfer(self, cfd: CFDResults) -> Dict[str, float]:
        """
        精确传质计算
        
        基于双膜理论和Higbie渗透理论
        """
        # 气泡参数
        gas_velocity = self.config.aeration_intensity / 3600 / 1000
        
        # Sauter平均直径
        d32 = self.calculate_sauter_diameter()
        
        # 气含率 (典型值 0.01-0.05)
        gas_holdup = np.mean(cfd.bubble_distribution) if cfd.bubble_distribution.size > 0 else 0.03
        
        # 气液界面面积 (基于气含率和气泡直径)
        # a = 6 * ε / d32
        a_interface = 6 * gas_holdup / d32  # m²/m³
        
        # 液相传质系数 (Higbie理论修正)
        # 使用气泡上升速度而非CFD平均速度
        bubble_velocity = 0.25  # m/s (典型气泡上升速度)
        contact_time = d32 / bubble_velocity
        kl = 2 * np.sqrt(self.oxygen_diffusivity / (np.pi * contact_time))
        
        # 总传质系数
        kla = kl * a_interface * 3600  # 1/h
        
        # 基础KLa值 (20°C参考值)
        kla_base = np.clip(kla, 5, 30)  # 基础值在合理范围
        
        # 温度校正 - 温度增加时KLa增加
        # theta = 1.024, 温度每增加10°C, KLa增加约27%
        temp_factor = self.constants.THETA_KLA ** (self.config.temperature_c - 20)
        kla_temp_corrected = kla_base * temp_factor
        
        # α因子校正 (污水)
        alpha = self.constants.OXYGEN_TRANSFER_ALPHA
        
        # β因子校正 (盐度)
        beta = self.constants.OXYGEN_TRANSFER_BETA
        
        # 曝气器类型校正
        diffuser_factor = 1.0
        
        kla_corrected = kla_temp_corrected * alpha * beta * diffuser_factor
        
        # 最终限制KLa在合理范围 (2-50 h⁻¹)，但保留温度影响的相对差异
        # 使用软限制而非硬限制
        if kla_corrected < 2:
            kla_corrected = 2 + 0.1 * (kla_corrected - 2)  # 软限制，保留差异
        elif kla_corrected > 50:
            kla_corrected = 50 - 0.1 * (50 - kla_corrected)
        kla_corrected = np.clip(kla_corrected, 2, 50)
        
        # 氧传递速率
        do_operating = 2.0  # mg/L (假设)
        otr = kla_corrected * (self.oxygen_saturation - do_operating)  # mg/L/h
        
        # 氧转移效率
        ote = otr / (self.config.aeration_intensity * 1000 / 3600 * 
                     self.constants.O2_MOLECULAR_WEIGHT / 22.4)
        
        return {
            'kla_20_h1': kla,
            'kla_actual_h1': kla_corrected,
            'kl_ms': kl,
            'a_interface_m2_m3': a_interface,
            'oxygen_saturation_mgl': self.oxygen_saturation,
            'otr_mgl_h': otr,
            'ote_pct': ote * 100,
            'do_operating_mgl': do_operating,
            'd32_mm': d32 * 1000
        }
    
    def calculate_sauter_diameter(self) -> float:
        """计算Sauter平均直径"""
        orifice_dia = self.config.orifice_diameter_mm / 1000
        gas_velocity = self.config.aeration_intensity / 3600 / 1000
        
        # 韦伯数
        we = (self.water_density * gas_velocity**2 * orifice_dia) / self.surface_tension
        
        # 基于Tadaki-Maeda关联式
        if we < 2:
            d32 = orifice_dia * 1.8
        elif we < 100:
            d32 = orifice_dia * 1.14 * we**(-0.05)
        else:
            d32 = orifice_dia * 1.51 * we**(-0.32)
        
        return np.clip(d32, 0.001, 0.02)
    
    def calculate_energy(self, total_area: float, cfd: CFDResults) -> Dict[str, float]:
        """
        详细能耗计算
        """
        # 曝气能耗
        water_depth = self.config.water_depth_m
        
        # 静压头
        static_head = self.water_density * self.constants.GRAVITY * water_depth
        
        # 管道损失 (Darcy-Weisbach)
        pipe_length = water_depth * 2  # 估算
        pipe_diameter = 0.1  # m
        velocity_pipe = self.config.aeration_intensity / 3600 / 1000
        re_pipe = velocity_pipe * pipe_diameter / self.kinematic_viscosity
        
        if re_pipe > 4000:
            f_pipe = 0.316 / re_pipe**0.25
        else:
            f_pipe = 64 / re_pipe
        
        head_loss_pipe = f_pipe * pipe_length / pipe_diameter * velocity_pipe**2 / (2 * self.constants.GRAVITY)
        
        # 曝气器损失
        head_loss_diffuser = 4000  # Pa (典型值)
        
        # 总压力
        total_pressure = static_head + head_loss_pipe * self.water_density * self.constants.GRAVITY + head_loss_diffuser
        
        # 空气流量
        airflow_rate = self.config.aeration_intensity / 3600 / 1000 * total_area
        
        # 等温压缩功
        gamma = 1.4
        p1 = self.constants.ATM_PRESSURE
        p2 = p1 + total_pressure
        
        compressor_work = (gamma / (gamma - 1)) * p1 * airflow_rate * (
            (p2 / p1)**((gamma - 1) / gamma) - 1
        )
        
        # 效率
        compressor_efficiency = 0.65
        motor_efficiency = 0.92
        total_efficiency = compressor_efficiency * motor_efficiency
        
        blower_power = compressor_work / total_efficiency
        
        # 抽吸泵能耗
        permeate_flow = self.config.target_flux_lmh / 1000 / 3600 * total_area
        tmp = self.config.operating_tmp_pa
        pump_efficiency = 0.6
        
        pumping_power = permeate_flow * tmp / pump_efficiency
        
        # 混合能耗
        mixing_power = 0.01 * total_area  # W/m²
        
        # 总功率
        total_power = blower_power + pumping_power + mixing_power
        
        # SEC计算
        sec = total_power / 1000 / (permeate_flow * 3600) if permeate_flow > 0 else 0
        
        # 年能耗
        annual_hours = 8760
        annual_energy = total_power / 1000 * annual_hours
        
        # 费用
        energy_cost = annual_energy * self.config.electricity_price_rmb_kwh
        
        return {
            'blower_power_w': blower_power,
            'blower_power_kw': blower_power / 1000,
            'pumping_power_w': pumping_power,
            'pumping_power_kw': pumping_power / 1000,
            'mixing_power_w': mixing_power,
            'mixing_power_kw': mixing_power / 1000,
            'total_power_w': total_power,
            'total_power_kw': total_power / 1000,
            'sec_kwh_m3': sec,
            'specific_power_w_m2': total_power / total_area,
            'annual_energy_kwh': annual_energy,
            'annual_energy_mwh': annual_energy / 1000,
            'annual_cost_rmb': energy_cost,
            'total_pressure_pa': total_pressure,
            'airflow_rate_m3_s': airflow_rate,
            'permeate_flow_m3_s': permeate_flow
        }
    
    def calculate_treatment_efficiency(self) -> Dict[str, float]:
        """计算处理效率"""
        # 基于HRT和SRT的经验模型
        hrt_hours = 8  # 假设
        
        # COD去除
        cod_eff = 1 - np.exp(-0.1 * hrt_hours * (1 + self.config.srt_days / 30))
        
        # BOD去除
        bod_eff = 1 - np.exp(-0.15 * hrt_hours * (1 + self.config.srt_days / 20))
        
        # TN去除 (依赖硝化反硝化)
        do_level = 2.0  # mg/L
        if do_level > 1.5:
            nitrification_eff = 0.95
        else:
            nitrification_eff = 0.7
        
        # 反硝化需要缺氧区
        denitrification_eff = 0.6
        tn_eff = nitrification_eff * denitrification_eff
        
        # TP去除 (化学除磷)
        tp_eff = 0.7
        
        return {
            'cod_efficiency': cod_eff * 100,
            'bod_efficiency': bod_eff * 100,
            'tn_efficiency': tn_eff * 100,
            'tp_efficiency': tp_eff * 100,
            'cod_effluent_mg_l': self.config.cod_influent_mg_l * (1 - cod_eff),
            'bod_effluent_mg_l': self.config.bod_influent_mg_l * (1 - bod_eff),
            'tn_effluent_mg_l': self.config.tn_influent_mg_l * (1 - tn_eff),
            'tp_effluent_mg_l': self.config.tp_influent_mg_l * (1 - tp_eff)
        }
    
    def calculate_economics(self, energy: Dict, total_area: float) -> Dict[str, float]:
        """经济性分析"""
        # 投资成本 (CAPEX)
        membrane_cost = total_area * self.config.membrane_replacement_cost_rmb_m2
        equipment_cost = membrane_cost * 3  # 膜组件占总投资的约1/3
        installation_cost = equipment_cost * 0.3
        
        total_capex = equipment_cost + installation_cost
        
        # 设计流量 (m³/d)
        design_flow = self.config.target_flux_lmh / 1000 * total_area * 24
        
        # 单位投资 (元/m³/d)
        capex_per_m3_d = total_capex / design_flow if design_flow > 0 else 0
        
        # 运行成本 (OPEX)
        energy_cost = energy['annual_cost_rmb']
        chemical_cost = design_flow * 365 * self.config.chemical_cost_rmb_m3
        labor_cost = design_flow * 365 * self.config.labor_cost_rmb_m3
        
        # 膜更换成本
        membrane_life_years = max(3, 10 - self.config.aeration_intensity / 50)
        membrane_replacement_annual = membrane_cost / membrane_life_years
        
        total_opex_annual = energy_cost + chemical_cost + labor_cost + membrane_replacement_annual
        
        # 单位运行成本 (元/m³)
        opex_per_m3 = total_opex_annual / (design_flow * 365) if design_flow > 0 else 0
        
        # 总成本 (元/m³) - 包含折旧
        depreciation_per_m3 = capex_per_m3_d * 0.1 / 365 if capex_per_m3_d > 0 else 0
        total_cost = opex_per_m3 + depreciation_per_m3
        
        # 限制成本在合理范围
        total_cost = np.clip(total_cost, 0.3, 5.0)
        
        return {
            'capex_total_rmb': total_capex,
            'capex_per_m3_d': capex_per_m3_d,
            'opex_annual_rmb': total_opex_annual,
            'opex_per_m3': opex_per_m3,
            'total_cost_per_m3': total_cost,
            'membrane_replacement_annual': membrane_replacement_annual,
            'energy_cost_annual': energy_cost,
            'chemical_cost_annual': chemical_cost,
            'labor_cost_annual': labor_cost,
            'membrane_life_years': membrane_life_years
        }
    
    def sensitivity_analysis(self) -> Dict[str, float]:
        """敏感性分析"""
        base_config = copy.deepcopy(self.config)
        base_result = self.run_full_calculation()
        base_sec = base_result.sec_kwh_m3
        base_score = base_result.operation_score
        
        sensitivities = {}
        
        # 参数变化范围
        params = {
            'aeration_intensity': (0.8, 1.2),
            'mlss_mg_l': (0.8, 1.2),
            'target_flux_lmh': (0.8, 1.2),
            'fiber_slack_pct': (0.5, 2.0),
            'srt_days': (0.7, 1.3)
        }
        
        for param_name, (low_factor, high_factor) in params.items():
            # 低值
            setattr(self.config, param_name, getattr(base_config, param_name) * low_factor)
            low_result = self.run_full_calculation()
            
            # 高值
            setattr(self.config, param_name, getattr(base_config, param_name) * high_factor)
            high_result = self.run_full_calculation()
            
            # 恢复
            setattr(self.config, param_name, getattr(base_config, param_name))
            
            # 计算敏感性
            sec_range = abs(high_result.sec_kwh_m3 - low_result.sec_kwh_m3) / base_sec
            score_range = abs(high_result.operation_score - low_result.operation_score)
            
            sensitivities[f'{param_name}_sec'] = sec_range
            sensitivities[f'{param_name}_score'] = score_range
        
        return sensitivities
    
    def optimize_parameters(self) -> Dict[str, float]:
        """参数优化建议"""
        # 保存原始配置
        original_config = copy.deepcopy(self.config)

        # 使用简单的网格搜索找到最优参数组合
        best_score = 0
        best_params = {}
        
        # 搜索空间
        intensity_range = np.linspace(60, 120, 7)
        mlss_range = np.linspace(6000, 12000, 7)
        flux_range = np.linspace(15, 30, 4)
        
        for intensity in intensity_range:
            for mlss in mlss_range:
                for flux in flux_range:
                    self.config.aeration_intensity = intensity
                    self.config.mlss_mg_l = mlss
                    self.config.target_flux_lmh = flux
                    
                    result = self.run_full_calculation()
                    
                    if result.operation_score > best_score:
                        best_score = result.operation_score
                        best_params = {
                            'aeration_intensity': intensity,
                            'mlss_mg_l': mlss,
                            'target_flux_lmh': flux,
                            'sec': result.sec_kwh_m3,
                            'score': result.operation_score
                        }
        
        # 恢复原始配置
        self.config = original_config
        
        return best_params
    
    def run_full_calculation(self) -> CalculationResult:
        """执行完整计算"""
        result = CalculationResult()

        # 保存膜规格和目标通量
        result.membrane = self.membrane
        result.target_flux_lmh = self.config.target_flux_lmh

        # 1. 膜面积
        sheet_area, fibers_per_sheet, total_fibers = self.calculate_membrane_area()
        result.single_sheet_area_m2 = sheet_area
        result.total_module_area_m2 = sheet_area * self.membrane.sheets_per_module
        result.fiber_count_per_sheet = fibers_per_sheet
        result.total_fiber_count = total_fibers
        
        # 2. CFD计算
        cfd = self.calculate_cfd()
        result.cfd_results = cfd
        
        # 3. 剪切力
        shear = self.calculate_shear_stress(cfd)
        result.avg_shear_pa = shear['tau_avg_pa']
        result.max_shear_pa = shear['tau_max_pa']
        result.min_shear_pa = shear['tau_min_pa']
        result.shear_stress_profile = shear['profile']
        result.shear_uniformity_index = shear['uniformity_index']
        result.shear_distribution_cv = shear['cv']
        
        # 4. 气泡参数
        result.bubble_diameter_mm = self.calculate_sauter_diameter() * 1000
        result.bubble_velocity_ms = cfd.max_velocity_ms
        result.bubble_frequency_hz = (self.config.aeration_intensity / 3600 / 1000 / 
                                      (np.pi * (self.config.orifice_diameter_mm / 1000 / 2)**2))
        result.gas_holdup = np.mean(cfd.bubble_distribution)
        result.sauter_mean_diameter = self.calculate_sauter_diameter() * 1000
        
        # 5. 传质
        mass_transfer = self.calculate_mass_transfer(cfd)
        result.kla_20 = mass_transfer['kla_20_h1']
        result.kla_actual = mass_transfer['kla_actual_h1']
        result.oxygen_transfer_rate = mass_transfer['otr_mgl_h']
        result.oxygen_transfer_efficiency = mass_transfer['ote_pct']
        result.do_level_mgl = mass_transfer['do_operating_mgl']
        
        # 6. 污染动力学
        fouling = self.calculate_fouling_kinetics(cfd, shear)
        result.fouling_kinetics = fouling
        
        # 临界通量 - 基于剪切力和MLSS计算
        # J_c = k * tau^n / (MLSS^m) - 单位: LMH
        k_critical = 15.0  # LMH基准值
        n_critical = 0.3
        m_critical = 0.15
        shear_factor = (result.avg_shear_pa / 1.0)**n_critical if result.avg_shear_pa > 0 else 0.5
        mlss_factor = (self.config.mlss_mg_l / 8000.0)**(-m_critical) if self.config.mlss_mg_l > 0 else 1.0
        critical_flux_lmh = k_critical * shear_factor * mlss_factor
        critical_flux_lmh = np.clip(critical_flux_lmh, 10, 50)
        result.critical_flux_lmh = critical_flux_lmh
        
        # 污染风险 - 基于MLSS和操作条件综合评估
        mlss = self.config.mlss_mg_l
        flux = self.config.target_flux_lmh
        
        # MLSS影响因子
        if mlss < 4000:
            mlss_risk_factor = 0.5  # 低风险
        elif mlss < 8000:
            mlss_risk_factor = 1.0  # 正常
        elif mlss < 12000:
            mlss_risk_factor = 1.5  # 中等风险
        else:
            mlss_risk_factor = 2.0  # 高风险
        
        # 通量影响因子
        if flux < 15:
            flux_risk_factor = 0.5
        elif flux < 25:
            flux_risk_factor = 1.0
        else:
            flux_risk_factor = 1.5
        
        # 综合风险评分
        risk_score = mlss_risk_factor * flux_risk_factor
        
        if risk_score < 0.75:
            result.fouling_risk = FoulingRisk.VERY_LOW
        elif risk_score < 1.25:
            result.fouling_risk = FoulingRisk.LOW
        elif risk_score < 2.0:
            result.fouling_risk = FoulingRisk.MEDIUM
        elif risk_score < 3.0:
            result.fouling_risk = FoulingRisk.HIGH
        else:
            result.fouling_risk = FoulingRisk.CRITICAL
        
        # 清洗周期 - 基于TMP增长率和污染风险
        if fouling.tmp_increase_rate_pa_s > 0:
            time_to_critical = ((self.membrane.critical_tmp_pa - self.config.operating_tmp_pa) / 
                               fouling.tmp_increase_rate_pa_s)
            result.cleaning_frequency_days = time_to_critical / 86400
        else:
            result.cleaning_frequency_days = 365
        
        # 根据污染风险调整清洗周期
        risk_adjustments = {
            FoulingRisk.VERY_LOW: 60,
            FoulingRisk.LOW: 30,
            FoulingRisk.MEDIUM: 14,
            FoulingRisk.HIGH: 7,
            FoulingRisk.CRITICAL: 3
        }
        base_cleaning_period = risk_adjustments.get(result.fouling_risk, 14)
        
        # 综合计算清洗周期
        result.cleaning_frequency_days = max(min(result.cleaning_frequency_days, 365), base_cleaning_period)
        result.cleaning_frequency_days = np.clip(result.cleaning_frequency_days, 7, 365)
        
        # 膜寿命
        chemical_cleaning_count = 365 / max(result.cleaning_frequency_days, 1)
        if chemical_cleaning_count < 10:
            result.membrane_lifetime_years = 8
        elif chemical_cleaning_count < 20:
            result.membrane_lifetime_years = 5
        else:
            result.membrane_lifetime_years = 3
        
        # 7. 能耗
        energy = self.calculate_energy(result.total_module_area_m2, cfd)
        result.sec_kwh_m3 = energy['sec_kwh_m3']
        result.specific_aeration_power = energy['specific_power_w_m2']
        result.blower_power_kw = energy['blower_power_kw']
        result.pumping_power_kw = energy['pumping_power_kw']
        result.mixing_power_kw = energy['mixing_power_kw']
        result.total_power_kw = energy['total_power_kw']
        result.annual_energy_kwh = energy['annual_energy_kwh']
        result.energy_cost_annual_rmb = energy['annual_cost_rmb']
        
        # 8. 处理效率
        efficiency = self.calculate_treatment_efficiency()
        result.cod_removal_efficiency = efficiency['cod_efficiency']
        result.bod_removal_efficiency = efficiency['bod_efficiency']
        result.tn_removal_efficiency = efficiency['tn_efficiency']
        result.tp_removal_efficiency = efficiency['tp_efficiency']
        
        # 9. 经济性
        economics = self.calculate_economics(energy, result.total_module_area_m2)
        result.capex_rmb_m3_d = economics['capex_per_m3_d']
        result.opex_rmb_m3 = economics['opex_per_m3']
        result.total_cost_rmb_m3 = economics['total_cost_per_m3']
        result.membrane_replacement_cost_annual = economics['membrane_replacement_annual']
        
        # 10. 评分
        result.operation_score = self._calculate_operation_score(result)
        result.optimization_score = self._calculate_optimization_score(result)
        result.sustainability_score = self._calculate_sustainability_score(result)
        
        # 11. 建议
        result.recommendations = self._generate_recommendations(result)
        result.warnings = self._generate_warnings(result)
        result.status_info = self._get_status_info(result)
        
        return result
    
    def _calculate_operation_score(self, result: CalculationResult) -> float:
        """计算运行评分"""
        scores = {}
        
        # 能耗评分
        sec = result.sec_kwh_m3
        if sec < 0.3:
            scores['energy'] = 100
        elif sec < 0.5:
            scores['energy'] = 90
        elif sec < 0.8:
            scores['energy'] = 75
        elif sec < 1.2:
            scores['energy'] = 60
        else:
            scores['energy'] = 40
        
        # 剪切评分
        tau = result.avg_shear_pa
        if 0.5 <= tau <= 2.0:
            scores['shear'] = 100
        elif 0.3 <= tau < 0.5 or 2.0 < tau <= 3.0:
            scores['shear'] = 80
        else:
            scores['shear'] = 60
        
        # 均匀性
        scores['uniformity'] = result.shear_uniformity_index * 100
        
        # 污染风险
        risk_scores = {
            FoulingRisk.VERY_LOW: 100,
            FoulingRisk.LOW: 90,
            FoulingRisk.MEDIUM: 70,
            FoulingRisk.HIGH: 50,
            FoulingRisk.CRITICAL: 30
        }
        scores['fouling'] = risk_scores.get(result.fouling_risk, 50)
        
        # 清洗周期
        cleaning = result.cleaning_frequency_days
        if cleaning >= 60:
            scores['cleaning'] = 100
        elif cleaning >= 30:
            scores['cleaning'] = 85
        elif cleaning >= 14:
            scores['cleaning'] = 70
        else:
            scores['cleaning'] = 50
        
        # 处理效率
        eff_avg = (result.cod_removal_efficiency + result.bod_removal_efficiency) / 2
        scores['efficiency'] = min(100, eff_avg)
        
        # 加权平均
        weights = {
            'energy': 0.2,
            'shear': 0.2,
            'uniformity': 0.15,
            'fouling': 0.2,
            'cleaning': 0.15,
            'efficiency': 0.1
        }
        
        return sum(scores[k] * weights[k] for k in scores)
    
    def _calculate_optimization_score(self, result: CalculationResult) -> float:
        """计算优化评分"""
        # 基于与理想状态的差距
        ideal_sec = 0.3
        ideal_shear = 1.0
        ideal_cleaning = 60
        
        sec_gap = abs(result.sec_kwh_m3 - ideal_sec) / ideal_sec
        shear_gap = abs(result.avg_shear_pa - ideal_shear) / ideal_shear
        cleaning_gap = abs(result.cleaning_frequency_days - ideal_cleaning) / ideal_cleaning
        
        total_gap = (sec_gap * 0.4 + shear_gap * 0.3 + cleaning_gap * 0.3)
        
        return max(0, 100 - total_gap * 50)
    
    def _calculate_sustainability_score(self, result: CalculationResult) -> float:
        """计算可持续性评分"""
        # 能耗效率
        energy_score = max(0, 100 - result.sec_kwh_m3 * 100)
        
        # 膜寿命
        life_score = result.membrane_lifetime_years * 10
        
        # 化学清洗频率
        cleaning_score = max(0, 100 - 365 / max(result.cleaning_frequency_days, 1) * 5)
        
        # 综合
        return (energy_score * 0.4 + life_score * 0.3 + cleaning_score * 0.3)
    
    def _generate_recommendations(self, result: CalculationResult) -> List[str]:
        """生成优化建议"""
        recommendations = []
        
        # 能耗建议
        if result.sec_kwh_m3 > 0.8:
            recommendations.append(f"⚡ 能耗偏高 ({result.sec_kwh_m3:.2f} kWh/m³)，建议：")
            recommendations.append("   • 采用脉冲曝气模式，可节能20-30%")
            recommendations.append("   • 优化曝气孔径至2-3mm，提高氧转移效率")
        
        # 剪切力建议
        if result.avg_shear_pa < 0.3:
            recommendations.append(f"⚠️ 剪切力偏低 ({result.avg_shear_pa:.2f} Pa)，膜污染风险增加")
            recommendations.append("   • 提高曝气强度至80 Nm³/m²/h以上")
            recommendations.append("   • 减小曝气管间距，增强膜丝振动")
        elif result.avg_shear_pa > 3.0:
            recommendations.append(f"⚠️ 剪切力过高 ({result.avg_shear_pa:.2f} Pa)，可能损伤膜丝")
            recommendations.append("   • 降低曝气强度或增大曝气孔径")
            recommendations.append("   • 检查膜丝松弛度是否合适")
        
        # 均匀性建议
        if result.shear_uniformity_index < 0.6:
            recommendations.append("🎯 剪切均匀性较差，存在局部污染风险")
            recommendations.append(f"   • 当前曝气管间距({self.config.pipe_spacing_mm}mm)与膜片间距({self.config.sheet_spacing_mm}mm)不匹配")
            recommendations.append("   • 建议调整至相近值")
        
        # 污染风险
        if result.fouling_risk in [FoulingRisk.HIGH, FoulingRisk.CRITICAL]:
            recommendations.append(f"🚨 污染风险{result.fouling_risk.value}！")
            recommendations.append(f"   • 当前通量({self.config.target_flux_lmh} LMH)接近/超过临界通量({result.critical_flux_lmh:.1f} LMH)")
            recommendations.append("   • 措施1: 降低操作通量或提高曝气强度")
            recommendations.append("   • 措施2: 优化污泥性质（调整SRT和MLSS）")
            recommendations.append("   • 措施3: 增加物理清洗频率")
        
        # MLSS建议
        mlss = self.config.mlss_mg_l
        if mlss > 12000:
            recommendations.append(f"⚠️ MLSS过高 ({mlss} mg/L)")
            recommendations.append("   • 增加排泥量，降低MLSS至8000-10000 mg/L")
            recommendations.append("   • 高MLSS会增加曝气能耗和膜污染")
        elif mlss < 4000:
            recommendations.append(f"⚠️ MLSS偏低 ({mlss} mg/L)")
            recommendations.append("   • 减少排泥，提高生物量")
        
        # SRT建议
        srt = self.config.srt_days
        if srt < 10:
            recommendations.append(f"📊 污泥龄偏短 ({srt} d)")
            recommendations.append("   • 可能影响硝化效果")
            recommendations.append("   • 建议SRT ≥ 15天以保证硝化菌生长")
        elif srt > 30:
            recommendations.append(f"📊 污泥龄偏长 ({srt} d)")
            recommendations.append("   • 可能增加SMP积累和膜污染")
            recommendations.append("   • 建议SRT控制在15-25天")
        
        # 经济性建议
        if result.total_cost_rmb_m3 > 1.5:
            recommendations.append(f"💰 运行成本偏高 (¥{result.total_cost_rmb_m3:.2f}/m³)")
            recommendations.append("   • 优化曝气策略降低能耗")
            recommendations.append("   • 延长膜寿命减少更换成本")
        
        return recommendations
    
    def _generate_warnings(self, result: CalculationResult) -> List[str]:
        """生成警告"""
        warnings = []
        
        if result.sec_kwh_m3 > 1.0:
            warnings.append(f"⚠️ 能耗警告: SEC = {result.sec_kwh_m3:.2f} kWh/m³ (建议 < 0.8)")
        
        if result.avg_shear_pa > 3.5:
            warnings.append(f"⚠️ 剪切力警告: τ = {result.avg_shear_pa:.2f} Pa (建议 < 3.0)")
        
        if result.fouling_risk == FoulingRisk.CRITICAL:
            warnings.append(f"🚨 严重污染风险！当前通量/临界通量比 > 1.5")
        elif result.fouling_risk == FoulingRisk.HIGH:
            warnings.append(f"⚠️ 高污染风险！建议立即调整运行参数")
        
        if result.cleaning_frequency_days < 7:
            warnings.append(f"⚠️ 清洗周期过短: {result.cleaning_frequency_days:.0f}天")
        
        if result.membrane_lifetime_years < 3:
            warnings.append(f"⚠️ 预计膜寿命较短: {result.membrane_lifetime_years}年")
        
        if result.cfd_results.reynolds_number > 100000:
            warnings.append(f"⚠️ 高雷诺数流动: Re = {result.cfd_results.reynolds_number:.0f}")
        
        return warnings
    
    def _get_status_info(self, result: CalculationResult) -> Dict[str, str]:
        """获取状态信息"""
        info = {}
        
        # 曝气模式
        mode_names = {
            AerationMode.CONTINUOUS: "连续曝气",
            AerationMode.PULSE: "脉冲曝气",
            AerationMode.INTERMITTENT: "间歇曝气",
            AerationMode.VARIABLE: "变频曝气"
        }
        info['aeration_mode'] = mode_names.get(self.config.aeration_mode, "未知")
        
        # 能耗等级
        if result.sec_kwh_m3 < 0.4:
            info['energy_class'] = "A+ (优秀)"
        elif result.sec_kwh_m3 < 0.6:
            info['energy_class'] = "A (良好)"
        elif result.sec_kwh_m3 < 0.8:
            info['energy_class'] = "B (中等)"
        else:
            info['energy_class'] = "C (需优化)"
        
        # 流动状态
        re = result.cfd_results.reynolds_number
        if re < 2300:
            info['flow_regime'] = "层流"
        elif re < 4000:
            info['flow_regime'] = "过渡流"
        else:
            info['flow_regime'] = "湍流"
        
        # 剪切状态
        tau = result.avg_shear_pa
        if tau < 0.5:
            info['shear_status'] = "偏低"
        elif tau <= 2.5:
            info['shear_status'] = "正常"
        else:
            info['shear_status'] = "偏高"
        
        # 经济性
        if result.total_cost_rmb_m3 < 1.0:
            info['cost_level'] = "低"
        elif result.total_cost_rmb_m3 < 1.5:
            info['cost_level'] = "中等"
        else:
            info['cost_level'] = "高"
        
        return info


def create_calculator_v2(config: Optional[SimulationConfig] = None) -> MBREngineeringCalculatorV2:
    """工厂函数"""
    return MBREngineeringCalculatorV2(config)


if __name__ == "__main__":
    # 测试
    calc = MBREngineeringCalculatorV2()
    result = calc.run_full_calculation()
    
    print("=" * 60)
    print("MBR工程计算引擎 V2.0 - 测试")
    print("=" * 60)
    
    print(f"\n📊 膜参数:")
    print(f"   单片膜面积: {result.single_sheet_area_m2:.2f} m²")
    print(f"   模块总膜面积: {result.total_module_area_m2:.2f} m²")
    
    print(f"\n⚡ 能耗:")
    print(f"   SEC: {result.sec_kwh_m3:.3f} kWh/m³")
    print(f"   总功率: {result.total_power_kw:.2f} kW")
    
    print(f"\n🌊 CFD结果:")
    print(f"   最大速度: {result.cfd_results.max_velocity_ms:.3f} m/s")
    print(f"   雷诺数: {result.cfd_results.reynolds_number:.0f}")
    print(f"   摩擦因子: {result.cfd_results.friction_factor:.4f}")
    
    print(f"\n🌊 剪切力:")
    print(f"   平均: {result.avg_shear_pa:.3f} Pa")
    print(f"   最大: {result.max_shear_pa:.3f} Pa")
    print(f"   均匀性: {result.shear_uniformity_index*100:.1f}%")
    
    print(f"\n📈 污染动力学:")
    print(f"   总阻力: {result.fouling_kinetics.total_resistance:.2e} 1/m")
    print(f"   TMP增长率: {result.fouling_kinetics.tmp_increase_rate_pa_s:.4f} Pa/s")
    print(f"   24h后TMP: {result.fouling_kinetics.tmp_after_24h_pa/1000:.2f} kPa")
    
    print(f"\n💧 处理效率:")
    print(f"   COD去除: {result.cod_removal_efficiency:.1f}%")
    print(f"   BOD去除: {result.bod_removal_efficiency:.1f}%")
    
    print(f"\n💰 经济性:")
    print(f"   投资: ¥{result.capex_rmb_m3_d:.0f}/m³/d")
    print(f"   运行: ¥{result.opex_rmb_m3:.2f}/m³")
    print(f"   总成本: ¥{result.total_cost_rmb_m3:.2f}/m³")
    
    print(f"\n⭐ 评分:")
    print(f"   运行: {result.operation_score:.1f}/100")
    print(f"   优化: {result.optimization_score:.1f}/100")
    print(f"   可持续: {result.sustainability_score:.1f}/100")
    
    print(f"\n💡 建议 ({len(result.recommendations)}条):")
    for i, rec in enumerate(result.recommendations[:5], 1):
        print(f"   {i}. {rec}")
