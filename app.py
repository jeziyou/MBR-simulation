"""
MBR仿真系统 - Streamlit主应用 V2.0
MBR Simulation System - Streamlit Main Application V2.0

全面升级版，包含：
- 增强版工程计算引擎
- 高级3D可视化
- 实时数据流
- 敏感性分析
- 参数优化
- 经济性分析

作者: MBR Engineering Team
版本: 2.0
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
import json

# 导入V2模块
from engine_v2 import (
    MBREngineeringCalculatorV2, SimulationConfig, AerationMode,
    FoulingRisk, create_calculator_v2
)
from visualization_v2 import MBRVisualizerV2, create_visualizer_v2


# ============== 页面配置 ==============
st.set_page_config(
    page_title="MBR工业仿真系统 v2.0",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============== 自定义CSS样式 ==============
st.markdown("""
<style>
    .main {
        background-color: #01050a;
    }
    
    .main-title {
        font-size: 2.5rem;
        font-weight: bold;
        color: #00f2ff;
        text-align: center;
        margin-bottom: 0.5rem;
        text-shadow: 0 0 20px rgba(0, 242, 255, 0.3);
    }
    
    .sub-title {
        font-size: 1.2rem;
        color: #8aacbf;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background: rgba(8, 18, 35, 0.96);
        border: 1px solid rgba(30, 74, 133, 0.8);
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        border-color: #00f2ff;
        box-shadow: 0 0 15px rgba(0, 242, 255, 0.2);
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #00f2ff;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #8aacbf;
    }
    
    .status-good { color: #00ff88; font-weight: bold; }
    .status-warning { color: #ff9900; font-weight: bold; }
    .status-danger { color: #ff4444; font-weight: bold; }
    
    .recommendation-card {
        background: rgba(0, 242, 255, 0.05);
        border-left: 4px solid #00f2ff;
        padding: 0.8rem;
        margin: 0.5rem 0;
        border-radius: 0 6px 6px 0;
    }
    
    .warning-card {
        background: rgba(255, 153, 0, 0.1);
        border-left: 4px solid #ff9900;
        padding: 0.8rem;
        margin: 0.5rem 0;
        border-radius: 0 6px 6px 0;
    }
    
    .danger-card {
        background: rgba(255, 68, 68, 0.1);
        border-left: 4px solid #ff4444;
        padding: 0.8rem;
        margin: 0.5rem 0;
        border-radius: 0 6px 6px 0;
    }
    
    .info-card {
        background: rgba(0, 136, 255, 0.05);
        border-left: 4px solid #0088ff;
        padding: 0.8rem;
        margin: 0.5rem 0;
        border-radius: 0 6px 6px 0;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: rgba(0, 242, 255, 0.05);
        border: 1px solid rgba(0, 242, 255, 0.2);
        border-radius: 6px 6px 0 0;
        color: #8aacbf;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: rgba(0, 242, 255, 0.15);
        color: #00f2ff;
    }
    
    .stButton > button {
        background-color: rgba(0, 242, 255, 0.1);
        border: 1px solid #00f2ff;
        color: #00f2ff;
        border-radius: 6px;
        font-weight: bold;
        transition: all 0.3s;
    }
    
    .stButton > button:hover {
        background-color: rgba(0, 242, 255, 0.2);
        box-shadow: 0 0 10px rgba(0, 242, 255, 0.3);
    }
    
    .css-1d391kg {
        background-color: rgba(8, 18, 35, 0.96);
    }
</style>
""", unsafe_allow_html=True)


# ============== 预设配置 ==============
PRESETS = {
    "节能模式 (Eco)": {
        "intensity": 60, "pulse_period": 6.0, "pipe_spacing": 250,
        "sheet_spacing": 100, "orifice_diameter": 6.0, "fiber_length": 2.0,
        "fiber_slack": 0.8, "mode": "pulse", "fiber_diameter": 2.8,
        "membrane_thickness": 30, "mlss": 6000, "srt": 20,
        "settling_rate": 2.0, "return_ratio": 80, "target_flux": 15,
        "water_depth": 4.0, "temperature": 20
    },
    "均衡模式 (Balanced)": {
        "intensity": 80, "pulse_period": 4.0, "pipe_spacing": 100,
        "sheet_spacing": 80, "orifice_diameter": 4.0, "fiber_length": 2.0,
        "fiber_slack": 1.5, "mode": "continuous", "fiber_diameter": 1.65,
        "membrane_thickness": 30, "mlss": 8000, "srt": 15,
        "settling_rate": 2.5, "return_ratio": 100, "target_flux": 20,
        "water_depth": 4.0, "temperature": 20
    },
    "高冲刷模式 (Flush)": {
        "intensity": 110, "pulse_period": 3.0, "pipe_spacing": 50,
        "sheet_spacing": 50, "orifice_diameter": 2.5, "fiber_length": 2.0,
        "fiber_slack": 2.5, "mode": "pulse", "fiber_diameter": 1.65,
        "membrane_thickness": 30, "mlss": 10000, "srt": 12,
        "settling_rate": 3.0, "return_ratio": 150, "target_flux": 25,
        "water_depth": 4.0, "temperature": 20
    }
}


# ============== Session State ==============
def init_session_state():
    if 'config' not in st.session_state:
        st.session_state.config = SimulationConfig()
    if 'calculator' not in st.session_state:
        st.session_state.calculator = MBREngineeringCalculatorV2(st.session_state.config)
    if 'result' not in st.session_state:
        st.session_state.result = None
    if 'visualizer' not in st.session_state:
        st.session_state.visualizer = create_visualizer_v2()
    if 'preset_applied' not in st.session_state:
        st.session_state.preset_applied = False
    if 'sensitivity' not in st.session_state:
        st.session_state.sensitivity = None


# ============== 侧边栏 ==============
def render_sidebar():
    with st.sidebar:
        st.markdown("<h2 style='color: #00f2ff; text-align: center;'>⚙️ MBR系统控制</h2>", 
                   unsafe_allow_html=True)
        
        # 场景预设
        st.markdown("---")
        st.subheader("🎯 场景预设")
        preset = st.selectbox("选择预设模式", ["自定义"] + list(PRESETS.keys()), key="preset_select")
        
        if preset != "自定义" and not st.session_state.preset_applied:
            apply_preset(preset)
            st.session_state.preset_applied = True
            st.rerun()
        
        # 曝气参数
        st.markdown("---")
        st.subheader("💨 曝气参数")
        
        mode_options = {"连续曝气": AerationMode.CONTINUOUS, "脉冲曝气": AerationMode.PULSE,
                       "间歇曝气": AerationMode.INTERMITTENT}
        mode_name = st.selectbox("曝气模式", list(mode_options.keys()),
                                index=1 if st.session_state.config.aeration_mode == AerationMode.PULSE else 0)
        st.session_state.config.aeration_mode = mode_options[mode_name]
        
        st.session_state.config.aeration_intensity = st.slider(
            "曝气强度 (Nm³/m²/h)", 50, 150, int(st.session_state.config.aeration_intensity), step=5)
        
        if st.session_state.config.aeration_mode in [AerationMode.PULSE, AerationMode.INTERMITTENT]:
            st.session_state.config.pulse_period_s = st.slider(
                "脉冲周期 (s)", 3.0, 6.0, float(st.session_state.config.pulse_period_s), step=0.5)
        
        st.session_state.config.orifice_diameter_mm = st.slider(
            "曝气孔径 (mm)", 1.0, 15.0, float(st.session_state.config.orifice_diameter_mm), step=0.5)
        st.session_state.config.pipe_spacing_mm = st.slider(
            "曝气管间距 (mm)", 50, 300, int(st.session_state.config.pipe_spacing_mm), step=10)
        
        # 膜参数
        st.markdown("---")
        st.subheader("🔬 膜参数")
        
        fiber_dia_options = {"1.65 mm (标准40m²)": 1.65, "2.8 mm (标准25m²)": 2.8}
        fiber_dia_name = st.selectbox("膜丝外径", list(fiber_dia_options.keys()),
                                     index=0 if st.session_state.config.fiber_diameter_mm == 1.65 else 1)
        st.session_state.config.fiber_diameter_mm = fiber_dia_options[fiber_dia_name]
        
        st.session_state.config.membrane_thickness_mm = st.slider(
            "膜片厚度 (mm)", 10, 100, int(st.session_state.config.membrane_thickness_mm), step=5)
        st.session_state.config.sheet_spacing_mm = st.slider(
            "膜片排列间距 (mm)", 50, 100, int(st.session_state.config.sheet_spacing_mm), step=5)
        st.session_state.config.fiber_length_m = st.slider(
            "膜丝长度 (m)", 0.1, 3.0, float(st.session_state.config.fiber_length_m), step=0.1)
        st.session_state.config.fiber_slack_pct = st.slider(
            "膜丝松弛度 (%)", 0.2, 5.0, float(st.session_state.config.fiber_slack_pct), step=0.1)
        
        # 污泥参数
        st.markdown("---")
        st.subheader("🟤 污泥参数")
        
        st.session_state.config.mlss_mg_l = st.slider(
            "MLSS浓度 (mg/L)", 2000, 15000, int(st.session_state.config.mlss_mg_l), step=500)
        st.session_state.config.srt_days = st.slider(
            "污泥龄 SRT (d)", 5, 40, int(st.session_state.config.srt_days), step=1)
        st.session_state.config.settling_rate_m_h = st.slider(
            "沉降速率 (m/h)", 0.5, 6.0, float(st.session_state.config.settling_rate_m_h), step=0.5)
        st.session_state.config.return_ratio_pct = st.slider(
            "污泥回流比 (%)", 50, 300, int(st.session_state.config.return_ratio_pct), step=10)
        
        # 操作条件
        st.markdown("---")
        st.subheader("⚡ 操作条件")
        
        st.session_state.config.target_flux_lmh = st.slider(
            "目标通量 (LMH)", 10, 40, int(st.session_state.config.target_flux_lmh), step=1)
        st.session_state.config.water_depth_m = st.slider(
            "水深 (m)", 2.0, 6.0, float(st.session_state.config.water_depth_m), step=0.5)
        st.session_state.config.temperature_c = st.slider(
            "水温 (°C)", 5, 35, int(st.session_state.config.temperature_c), step=1)
        
        # 计算按钮
        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🚀 执行计算", use_container_width=True):
                with st.spinner("正在执行工程计算..."):
                    st.session_state.calculator = MBREngineeringCalculatorV2(st.session_state.config)
                    st.session_state.result = st.session_state.calculator.run_full_calculation()
                st.success("✅ 计算完成！")
        with col2:
            if st.button("🔍 敏感性分析", use_container_width=True):
                with st.spinner("正在分析参数敏感性..."):
                    calc = MBREngineeringCalculatorV2(st.session_state.config)
                    st.session_state.sensitivity = calc.sensitivity_analysis()
                st.success("✅ 分析完成！")


def apply_preset(preset_name: str):
    preset = PRESETS[preset_name]
    config = st.session_state.config
    config.aeration_intensity = preset["intensity"]
    config.pulse_period_s = preset["pulse_period"]
    config.pipe_spacing_mm = preset["pipe_spacing"]
    config.sheet_spacing_mm = preset["sheet_spacing"]
    config.orifice_diameter_mm = preset["orifice_diameter"]
    config.fiber_length_m = preset["fiber_length"]
    config.fiber_slack_pct = preset["fiber_slack"]
    config.aeration_mode = AerationMode(preset["mode"])
    config.fiber_diameter_mm = preset["fiber_diameter"]
    config.membrane_thickness_mm = preset["membrane_thickness"]
    config.mlss_mg_l = preset["mlss"]
    config.srt_days = preset["srt"]
    config.settling_rate_m_h = preset["settling_rate"]
    config.return_ratio_pct = preset["return_ratio"]
    config.target_flux_lmh = preset["target_flux"]
    config.water_depth_m = preset["water_depth"]
    config.temperature_c = preset["temperature"]


# ============== 主页面 ==============
def render_header():
    st.markdown("<div class='main-title'>🌊 MBR工业仿真系统 v2.0</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-title'>工程级膜生物反应器仿真与优化平台 | 增强版</div>", unsafe_allow_html=True)


def render_metrics(result):
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    with col1:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-label'>单位体积能耗 SEC</div>
            <div class='metric-value'>{result.sec_kwh_m3:.3f}</div>
            <div class='metric-label'>kWh/m³</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-label'>平均剪切力 τ̄</div>
            <div class='metric-value'>{result.avg_shear_pa:.3f}</div>
            <div class='metric-label'>Pa</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        risk_color = {"very_low": "status-good", "low": "status-good", 
                     "medium": "status-warning", "high": "status-danger", 
                     "critical": "status-danger"}.get(result.fouling_risk.value, "status-warning")
        risk_text = result.fouling_risk.value.upper()
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-label'>污染风险</div>
            <div class='metric-value {risk_color}'>{risk_text}</div>
            <div class='metric-label'>等级评估</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-label'>运行评分</div>
            <div class='metric-value'>{result.operation_score:.0f}</div>
            <div class='metric-label'>/ 100</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col5:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-label'>优化评分</div>
            <div class='metric-value'>{result.optimization_score:.0f}</div>
            <div class='metric-label'>/ 100</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col6:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-label'>预计膜寿命</div>
            <div class='metric-value'>{result.membrane_lifetime_years}</div>
            <div class='metric-label'>年</div>
        </div>
        """, unsafe_allow_html=True)


def render_unified_3d(result):
    """渲染统一的3D场景 - 膜组件 + 气泡 + 污泥"""
    st.subheader("🔬 统一3D可视化场景")
    
    viz = st.session_state.visualizer
    config = st.session_state.config
    
    # 控制选项
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        show_bubbles = st.toggle("显示气泡", value=True)
    with col2:
        show_sludge = st.toggle("显示污泥", value=False)
    with col3:
        show_fibers = st.toggle("显示膜丝", value=True)
    with col4:
        fiber_density = st.slider("膜丝密度", 10, 40, 20, step=5)
    
    # 创建统一3D场景
    fig = viz.create_unified_3d_scene(
        sheet_count=5,
        sheet_width=1.25,
        sheet_length=config.fiber_length_m,
        sheet_spacing=config.sheet_spacing_mm / 1000,
        fiber_diameter=config.fiber_diameter_mm,
        fiber_slack=config.fiber_slack_pct,
        show_fibers=show_fibers,
        fiber_density=fiber_density,
        bubble_count=150,
        aeration_intensity=config.aeration_intensity,
        water_depth=config.water_depth_m,
        show_bubbles=show_bubbles,
        show_sludge=show_sludge,
        sludge_count=200
    )
    
    st.plotly_chart(fig, use_container_width=True, height=700)
    
    # 场景说明
    st.info("""
    💡 **场景说明**: 
    - **膜组件**: 5片膜组件，包含集水管、导轨和膜丝
    - **气泡**: 从曝气管产生，沿膜丝上升，提供剪切力
    - **污泥**: 分布在膜组件周围（可选显示）
    - 可拖拽旋转、滚轮缩放查看细节
    """)
    
    # 第二行 - 单独的可视化选项
    st.markdown("---")
    st.subheader("📊 单独可视化组件")
    
    col5, col6 = st.columns(2)
    
    with col5:
        st.markdown("**膜组件结构**")
        fig_membrane = viz.create_membrane_structure_enhanced(
            sheet_count=5, sheet_width=1.25,
            sheet_length=config.fiber_length_m,
            sheet_spacing=config.sheet_spacing_mm / 1000,
            fiber_diameter=config.fiber_diameter_mm,
            fiber_slack=config.fiber_slack_pct,
            show_fibers=True, fiber_density=20, lighting=True
        )
        st.plotly_chart(fig_membrane, use_container_width=True, height=400)
    
    with col6:
        st.markdown("**动态粒子动画**")
        fig_anim = viz.create_animation_frames(frame_count=30)
        st.plotly_chart(fig_anim, use_container_width=True, height=400)


def render_cfd_results(result):
    st.subheader("🌊 CFD计算结果")
    
    viz = st.session_state.visualizer
    
    if hasattr(result, 'cfd_results') and result.cfd_results.velocity_field.size > 0:
        fig = viz.create_cfd_visualization(result.cfd_results)
        st.plotly_chart(fig, width="stretch")
    else:
        st.info("CFD结果不可用")
    
    # CFD统计信息
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("最大速度", f"{result.cfd_results.max_velocity_ms:.3f} m/s")
    with col2:
        st.metric("平均速度", f"{result.cfd_results.avg_velocity_ms:.3f} m/s")
    with col3:
        st.metric("雷诺数", f"{result.cfd_results.reynolds_number:.0f}")
    with col4:
        st.metric("摩擦因子", f"{result.cfd_results.friction_factor:.4f}")


def render_fouling_analysis(result):
    st.subheader("📈 膜污染动力学分析")
    
    fk = result.fouling_kinetics
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("总阻力", f"{fk.total_resistance:.2e} 1/m")
        st.metric("滤饼阻力", f"{fk.cake_resistance:.2e} 1/m")
        st.metric("孔堵塞阻力", f"{fk.pore_blocking_resistance:.2e} 1/m")
    with col2:
        st.metric("生物膜阻力", f"{fk.biofilm_resistance:.2e} 1/m")
        st.metric("有机污染阻力", f"{fk.organic_resistance:.2e} 1/m")
        st.metric("无机结垢阻力", f"{fk.inorganic_resistance:.2e} 1/m")
    with col3:
        st.metric("TMP增长率", f"{fk.tmp_increase_rate_pa_s:.4f} Pa/s")
        st.metric("24h后TMP", f"{fk.tmp_after_24h_pa/1000:.2f} kPa")
        st.metric("物理清洗恢复率", f"{fk.physical_cleaning_recovery*100:.1f}%")
    
    # 阻力组成饼图
    resistances = {
        '初始阻力': result.membrane.initial_resistance,
        '滤饼层': fk.cake_resistance,
        '孔堵塞': fk.pore_blocking_resistance,
        '生物膜': fk.biofilm_resistance,
        '有机污染': fk.organic_resistance,
        '无机结垢': fk.inorganic_resistance
    }
    
    fig = go.Figure(data=[go.Pie(
        labels=list(resistances.keys()),
        values=list(resistances.values()),
        hole=0.4,
        marker_colors=['#00f2ff', '#ff9900', '#ff4444', '#00ff88', '#0088ff', '#d4a84b']
    )])
    fig.update_layout(
        title="膜阻力组成",
        paper_bgcolor='rgba(1, 5, 10, 0.95)',
        font=dict(color='white')
    )
    st.plotly_chart(fig, width="stretch")


def render_performance_radar(result):
    st.subheader("⭐ 性能雷达图")
    
    viz = st.session_state.visualizer
    fig = viz.create_performance_radar(result)
    st.plotly_chart(fig, width="stretch")


def render_economics(result):
    st.subheader("💰 经济性分析")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("投资成本", f"¥{result.capex_rmb_m3_d:.0f}/m³/d")
    with col2:
        st.metric("运行成本", f"¥{result.opex_rmb_m3:.2f}/m³")
    with col3:
        st.metric("总成本", f"¥{result.total_cost_rmb_m3:.2f}/m³")
    
    # 成本组成
    costs = {
        '能耗': result.energy_cost_annual_rmb,
        '膜更换': result.membrane_replacement_cost_annual,
        '化学药剂': result.energy_cost_annual_rmb * 0.15,
        '人工': result.energy_cost_annual_rmb * 0.25
    }
    
    fig = go.Figure(data=[go.Bar(
        x=list(costs.keys()),
        y=list(costs.values()),
        marker_color=['#00f2ff', '#ff9900', '#00ff88', '#0088ff']
    )])
    fig.update_layout(
        title="年运行成本组成",
        paper_bgcolor='rgba(1, 5, 10, 0.95)',
        plot_bgcolor='rgba(1, 5, 10, 0.95)',
        font=dict(color='white')
    )
    st.plotly_chart(fig, width="stretch")


def render_sensitivity():
    st.subheader("📊 敏感性分析")
    
    if 'sensitivity' in st.session_state and st.session_state.sensitivity:
        viz = st.session_state.visualizer
        fig = viz.create_sensitivity_chart(st.session_state.sensitivity)
        st.plotly_chart(fig, width="stretch")
    else:
        st.info("请点击侧边栏的'敏感性分析'按钮执行分析")


def render_recommendations(result):
    st.subheader("💡 智能优化建议")
    
    if result.warnings:
        st.markdown("#### ⚠️ 警告信息")
        for warning in result.warnings:
            if "严重" in warning or "CRITICAL" in warning:
                st.markdown(f"<div class='danger-card'>{warning}</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='warning-card'>{warning}</div>", unsafe_allow_html=True)
    
    if result.recommendations:
        st.markdown("#### 💡 改进建议")
        for rec in result.recommendations:
            st.markdown(f"<div class='recommendation-card'>{rec}</div>", unsafe_allow_html=True)
    
    # 状态信息
    if result.status_info:
        st.markdown("#### 📋 运行状态")
        for key, value in result.status_info.items():
            st.markdown(f"<div class='info-card'><b>{key}:</b> {value}</div>", unsafe_allow_html=True)


def render_detailed_results(result):
    st.subheader("📋 详细计算结果")
    
    data = {
        "参数类别": ["膜参数", "膜参数", "膜参数", "膜参数",
                   "能耗", "能耗", "能耗", "能耗", "能耗", "能耗",
                   "剪切力", "剪切力", "剪切力", "剪切力",
                   "CFD", "CFD", "CFD",
                   "气泡", "气泡", "气泡",
                   "传质", "传质", "传质",
                   "污染", "污染", "污染", "污染", "污染",
                   "处理效率", "处理效率", "处理效率", "处理效率",
                   "经济性", "经济性", "经济性",
                   "性能", "性能", "性能"],
        "参数名称": [
            "单片膜面积", "模块总膜面积", "每片膜丝数", "总膜丝数",
            "单位体积能耗(SEC)", "鼓风机功率", "抽吸泵功率", "混合功率", "总功率", "年费用",
            "平均剪切力", "最大剪切力", "最小剪切力", "均匀性指数",
            "最大速度", "雷诺数", "摩擦因子",
            "气泡直径", "上升速度", "气含率",
            "KLa(20°C)", "实际KLa", "氧传递速率",
            "临界通量", "污染率", "清洗周期", "膜寿命", "总阻力",
            "COD去除率", "BOD去除率", "TN去除率", "TP去除率",
            "投资成本", "运行成本", "总成本",
            "运行评分", "优化评分", "可持续评分"
        ],
        "数值": [
            f"{result.single_sheet_area_m2:.2f} m²",
            f"{result.total_module_area_m2:.2f} m²",
            f"{result.fiber_count_per_sheet}",
            f"{result.total_fiber_count}",
            f"{result.sec_kwh_m3:.3f} kWh/m³",
            f"{result.blower_power_kw:.2f} kW",
            f"{result.pumping_power_kw:.3f} kW",
            f"{result.mixing_power_kw:.3f} kW",
            f"{result.total_power_kw:.2f} kW",
            f"¥{result.energy_cost_annual_rmb:.0f}",
            f"{result.avg_shear_pa:.3f} Pa",
            f"{result.max_shear_pa:.3f} Pa",
            f"{result.min_shear_pa:.3f} Pa",
            f"{result.shear_uniformity_index*100:.1f}%",
            f"{result.cfd_results.max_velocity_ms:.3f} m/s",
            f"{result.cfd_results.reynolds_number:.0f}",
            f"{result.cfd_results.friction_factor:.4f}",
            f"{result.bubble_diameter_mm:.2f} mm",
            f"{result.bubble_velocity_ms:.3f} m/s",
            f"{result.gas_holdup*100:.1f}%",
            f"{result.kla_20:.1f} h⁻¹",
            f"{result.kla_actual:.1f} h⁻¹",
            f"{result.oxygen_transfer_rate:.1f} mg/L/h",
            f"{result.critical_flux_lmh:.1f} LMH",
            f"{result.fouling_kinetics.tmp_increase_rate_pa_s*1000:.4f} mbar/s",
            f"{result.cleaning_frequency_days:.0f} 天",
            f"{result.membrane_lifetime_years} 年",
            f"{result.fouling_kinetics.total_resistance:.2e} 1/m",
            f"{result.cod_removal_efficiency:.1f}%",
            f"{result.bod_removal_efficiency:.1f}%",
            f"{result.tn_removal_efficiency:.1f}%",
            f"{result.tp_removal_efficiency:.1f}%",
            f"¥{result.capex_rmb_m3_d:.0f}/m³/d",
            f"¥{result.opex_rmb_m3:.2f}/m³",
            f"¥{result.total_cost_rmb_m3:.2f}/m³",
            f"{result.operation_score:.0f}/100",
            f"{result.optimization_score:.0f}/100",
            f"{result.sustainability_score:.0f}/100"
        ]
    }
    
    df = pd.DataFrame(data)
    st.dataframe(df, width="stretch", hide_index=True)


def render_export(result):
    st.subheader("💾 数据导出")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        result_dict = {
            "timestamp": datetime.now().isoformat(),
            "config": {k: str(v) for k, v in st.session_state.config.__dict__.items()},
            "results": {
                "sec_kwh_m3": result.sec_kwh_m3,
                "avg_shear_pa": result.avg_shear_pa,
                "fouling_risk": result.fouling_risk.value,
                "operation_score": result.operation_score,
                "optimization_score": result.optimization_score,
                "sustainability_score": result.sustainability_score,
                "membrane_lifetime_years": result.membrane_lifetime_years,
                "total_cost_rmb_m3": result.total_cost_rmb_m3
            },
            "recommendations": result.recommendations,
            "warnings": result.warnings
        }
        json_str = json.dumps(result_dict, indent=2, ensure_ascii=False)
        st.download_button("📄 导出JSON", json_str,
                          f"MBR_v2_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                          "application/json")
    
    with col2:
        csv_data = {
            "参数": ["SEC", "平均剪切力", "污染风险", "运行评分", "优化评分", "可持续评分",
                    "膜寿命", "总成本", "COD去除率", "BOD去除率"],
            "数值": [result.sec_kwh_m3, result.avg_shear_pa, result.fouling_risk.value,
                    result.operation_score, result.optimization_score, result.sustainability_score,
                    result.membrane_lifetime_years, result.total_cost_rmb_m3,
                    result.cod_removal_efficiency, result.bod_removal_efficiency],
            "单位": ["kWh/m³", "Pa", "", "", "", "", "年", "元/m³", "%", "%"]
        }
        df_csv = pd.DataFrame(csv_data)
        st.download_button("📊 导出CSV", df_csv.to_csv(index=False),
                          f"MBR_v2_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                          "text/csv")
    
    with col3:
        report = f"""
MBR仿真系统 V2.0 计算报告
========================
生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

【运行参数】
曝气强度: {st.session_state.config.aeration_intensity} Nm³/m²/h
曝气模式: {st.session_state.config.aeration_mode.value}
MLSS: {st.session_state.config.mlss_mg_l} mg/L
SRT: {st.session_state.config.srt_days} d
目标通量: {st.session_state.config.target_flux_lmh} LMH

【计算结果】
SEC: {result.sec_kwh_m3:.3f} kWh/m³
平均剪切力: {result.avg_shear_pa:.3f} Pa
污染风险: {result.fouling_risk.value}
运行评分: {result.operation_score:.0f}/100
优化评分: {result.optimization_score:.0f}/100
可持续评分: {result.sustainability_score:.0f}/100
预计膜寿命: {result.membrane_lifetime_years} 年
预计清洗周期: {result.cleaning_frequency_days:.0f} 天
临界通量: {result.critical_flux_lmh:.1f} LMH
总成本: ¥{result.total_cost_rmb_m3:.2f}/m³

【优化建议】
"""
        for i, rec in enumerate(result.recommendations, 1):
            report += f"{i}. {rec}\n"
        
        st.download_button("📝 导出报告", report,
                          f"MBR_v2_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                          "text/plain")


# ============== 主程序 ==============
def main():
    init_session_state()
    render_header()
    render_sidebar()
    
    result = st.session_state.result
    
    if result is None:
        with st.spinner("正在初始化计算..."):
            st.session_state.calculator = MBREngineeringCalculatorV2(st.session_state.config)
            st.session_state.result = st.session_state.calculator.run_full_calculation()
        result = st.session_state.result
    
    render_metrics(result)
    
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "🔬 3D可视化", "🌊 CFD分析", "📈 污染动力学",
        "⭐ 性能雷达", "💰 经济性", "📊 敏感性", "💡 建议 & 导出"
    ])
    
    with tab1:
        render_unified_3d(result)
    
    with tab2:
        render_cfd_results(result)
    
    with tab3:
        render_fouling_analysis(result)
    
    with tab4:
        render_performance_radar(result)
    
    with tab5:
        render_economics(result)
    
    with tab6:
        render_sensitivity()
    
    with tab7:
        render_recommendations(result)
        st.markdown("---")
        render_detailed_results(result)
        st.markdown("---")
        render_export(result)
    
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #4a6a85; font-size: 0.8rem;'>
        MBR工业仿真系统 v2.0 | 工程级CFD计算引擎 | © 2024 MBR Engineering Team
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
