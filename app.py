"""
MBR 工业仿真系统 - Streamlit Web 应用 V8.0
MBR Industrial Simulation System - Streamlit Web App V8.0

基于 V8.0 单文件核心模块，提供完整的交互式 Web 界面：
- 侧边栏参数配置 + 场景预设
- KPI 指标仪表盘
- 阻力串联模型 / EPS-SMP / 化学清洗 / 全生命周期成本
- 3D 可视化 + 优化 + 灵敏度分析
"""

import sys
import os
import time

# 确保当前目录在 path 中
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import json
from typing import Optional, Dict, Any

# 从 V8.0 单文件模块导入
from mbr_simulation_single_file import (
    SimulationConfig, MBREngineeringCalculator, MBRVisualizer,
    AerationMode, FoulingRisk, ScenarioPreset, SCENARIO_PRESETS,
    AerationPhysics, CalculationResult, ParameterOptimizer,
    SensitivityAnalyzer, ScenarioComparator,
)

# ─────────────────────────────────────────────────────────
# 页面配置
# ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MBR 工业仿真系统 V8.0",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────
# CSS 样式 (深色工业主题)
# ─────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* 全局背景 */
    .stApp {
        background: linear-gradient(135deg, #0a1628 0%, #0d1f35 100%);
    }

    /* 主标题 */
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #00d4ff, #0099ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.0rem;
        color: #6b8aaa;
        text-align: center;
        margin-bottom: 1.5rem;
    }

    /* KPI 卡片 */
    .kpi-card {
        background: rgba(10, 25, 50, 0.85);
        border: 1px solid rgba(0, 200, 255, 0.2);
        border-radius: 12px;
        padding: 1.0rem 0.8rem;
        text-align: center;
        transition: all 0.3s;
    }
    .kpi-card:hover {
        border-color: rgba(0, 200, 255, 0.6);
        box-shadow: 0 0 20px rgba(0, 200, 255, 0.1);
    }
    .kpi-label {
        font-size: 0.75rem;
        color: #6b8aaa;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .kpi-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #e0f0ff;
        margin: 0.3rem 0;
    }
    .kpi-unit {
        font-size: 0.7rem;
        color: #5a7a95;
    }

    /* 状态标签 */
    .badge-green {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 10px;
        background: rgba(0, 200, 100, 0.15);
        color: #00c864;
        font-weight: 600;
        font-size: 0.8rem;
    }
    .badge-yellow {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 10px;
        background: rgba(255, 180, 0, 0.15);
        color: #ffb400;
        font-weight: 600;
        font-size: 0.8rem;
    }
    .badge-red {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 10px;
        background: rgba(255, 60, 60, 0.15);
        color: #ff3c3c;
        font-weight: 600;
        font-size: 0.8rem;
    }

    /* 建议卡片 */
    .advice-urgent {
        background: rgba(255, 60, 60, 0.08);
        border-left: 3px solid #ff3c3c;
        padding: 0.5rem 0.8rem;
        margin: 0.3rem 0;
        border-radius: 0 6px 6px 0;
        font-size: 0.85rem;
    }
    .advice-optimize {
        background: rgba(255, 180, 0, 0.08);
        border-left: 3px solid #ffb400;
        padding: 0.5rem 0.8rem;
        margin: 0.3rem 0;
        border-radius: 0 6px 6px 0;
        font-size: 0.85rem;
    }
    .advice-longterm {
        background: rgba(0, 180, 255, 0.08);
        border-left: 3px solid #00b4ff;
        padding: 0.5rem 0.8rem;
        margin: 0.3rem 0;
        border-radius: 0 6px 6px 0;
        font-size: 0.85rem;
    }

    /* 信息行 */
    .info-row {
        display: flex;
        justify-content: space-between;
        padding: 0.3rem 0;
        border-bottom: 1px solid rgba(255,255,255,0.05);
    }
    .info-label { color: #6b8aaa; font-size: 0.85rem; }
    .info-value { color: #c0d8f0; font-weight: 600; font-size: 0.85rem; }

    /* 侧边栏 */
    section[data-testid="stSidebar"] {
        background: rgba(8, 18, 35, 0.97);
        border-right: 1px solid rgba(0, 200, 255, 0.1);
    }

    /* 按钮 */
    .stButton > button {
        background: linear-gradient(135deg, #0066cc, #0099ff) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 700 !important;
        padding: 0.5rem 1.5rem !important;
        transition: all 0.3s !important;
    }
    .stButton > button:hover {
        box-shadow: 0 0 15px rgba(0, 150, 255, 0.4) !important;
        transform: translateY(-1px);
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────
# Session State 初始化
# ─────────────────────────────────────────────────────────
def init_session():
    defaults = {
        "config": None,
        "result": None,
        "calculator": None,
        "preset_applied": False,
        "opt_result": None,
        "sensitivity": None,
        "comparison": None,
        "viz_3d_fig": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ─────────────────────────────────────────────────────────
# 侧边栏
# ─────────────────────────────────────────────────────────
def render_sidebar():
    with st.sidebar:
        st.markdown("### ⚙️ 参数配置")

        # ── 场景预设 ──
        preset_names = {p.value: SCENARIO_PRESETS[p]["name"] for p in ScenarioPreset}
        preset_choice = st.selectbox(
            "📋 场景预设",
            ["自定义"] + list(preset_names.keys()),
            format_func=lambda x: "自定义" if x == "自定义" else f"{x} - {preset_names[x]}"
        )

        if preset_choice != "自定义":
            preset = ScenarioPreset(preset_choice)
            cfg = SimulationConfig.from_preset(preset)
            st.session_state.config = cfg
            st.session_state.preset_applied = True
        else:
            if st.session_state.config is None:
                st.session_state.config = SimulationConfig()

        cfg = st.session_state.config

        st.markdown("---")

        # ── 曝气参数 ──
        st.markdown("#### 💨 曝气参数")
        cfg.aeration_intensity = st.slider("曝气强度 (Nm³/m²/h)", 30, 200, int(cfg.aeration_intensity), 5)
        cfg.orifice_diameter_mm = st.slider("曝气孔径 (mm)", 2.0, 12.0, float(cfg.orifice_diameter_mm), 0.5)
        mode_names = {"连续": AerationMode.CONTINUOUS, "脉冲": AerationMode.PULSE,
                      "间歇": AerationMode.INTERMITTENT, "循环": AerationMode.CYCLIC}
        mode_sel = st.selectbox("曝气模式", list(mode_names.keys()),
                                index=list(mode_names.values()).index(cfg.aeration_mode) if cfg.aeration_mode in mode_names.values() else 0)
        cfg.aeration_mode = mode_names[mode_sel]

        # ── 膜参数 ──
        st.markdown("#### 🔬 膜参数")
        cfg.sheet_count = st.slider("膜片数", 5, 50, cfg.sheet_count, 1)
        cfg.target_flux_lmh = st.slider("目标通量 (LMH)", 8, 35, int(cfg.target_flux_lmh), 1)
        cfg.membrane_area_m2 = st.slider("膜面积 (m²)", 10, 100, int(cfg.membrane_area_m2), 5)

        # ── 运行参数 ──
        st.markdown("#### 🟤 运行参数")
        cfg.mlss_mg_l = st.slider("MLSS (mg/L)", 2000, 15000, int(cfg.mlss_mg_l), 500)
        cfg.srt_days = st.slider("SRT (天)", 5, 40, int(cfg.srt_days), 1)
        cfg.temperature_c = st.slider("水温 (°C)", 5, 35, int(cfg.temperature_c), 1)
        cfg.ph = st.slider("pH", 5.5, 9.0, float(cfg.ph), 0.1)
        cfg.do_setpoint_mg_l = st.slider("DO 设定点 (mg/L)", 0.5, 5.0, float(cfg.do_setpoint_mg_l), 0.5)

        # ── 进水水质 ──
        st.markdown("#### 💧 进水水质")
        cfg.cod_influent_mg_l = st.number_input("COD进水 (mg/L)", 50, 2000, int(cfg.cod_influent_mg_l), 50)
        cfg.nh4_influent_mg_l = st.number_input("NH₄-N进水 (mg/L)", 5, 100, int(cfg.nh4_influent_mg_l), 5)
        cfg.tn_influent_mg_l = st.number_input("TN进水 (mg/L)", 10, 150, int(cfg.tn_influent_mg_l), 5)
        cfg.tp_influent_mg_l = st.number_input("TP进水 (mg/L)", 1, 30, int(cfg.tp_influent_mg_l), 1)

        st.markdown("---")

        # ── 运行按钮 ──
        col1, col2 = st.columns(2)
        with col1:
            run_clicked = st.button("🚀 运行仿真", use_container_width=True, type="primary")
        with col2:
            opt_clicked = st.button("🔍 自动优化", use_container_width=True)

        st.markdown("---")
        col3, col4 = st.columns(2)
        with col3:
            sens_clicked = st.button("📊 灵敏度", use_container_width=True)
        with col4:
            comp_clicked = st.button("📋 场景对比", use_container_width=True)

        if run_clicked:
            with st.spinner("计算中..."):
                calc = MBREngineeringCalculator(cfg)
                st.session_state.result = calc.run_full_calculation()
                st.session_state.calculator = calc
            st.rerun()

        if opt_clicked:
            with st.spinner("优化中 (评估500个参数组合)..."):
                calc = MBREngineeringCalculator(cfg)
                st.session_state.result = calc.run_full_calculation()
                optimizer = ParameterOptimizer(cfg)
                st.session_state.opt_result = optimizer.optimize(max_results=10)
            st.rerun()

        if sens_clicked:
            with st.spinner("灵敏度分析中..."):
                calc = MBREngineeringCalculator(cfg)
                st.session_state.result = calc.run_full_calculation()
                analyzer = SensitivityAnalyzer(cfg)
                st.session_state.sensitivity = analyzer.analyze()
            st.rerun()

        if comp_clicked:
            with st.spinner("运行5个场景对比..."):
                comparator = ScenarioComparator()
                st.session_state.comparison = comparator.compare_all_presets()
            st.rerun()


# ─────────────────────────────────────────────────────────
# 主区域
# ─────────────────────────────────────────────────────────

def render_header():
    st.markdown("<div class='main-title'>🌊 MBR 工业仿真系统 V8.0</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-title'>阻力串联模型 · EPS/SMP · ASM1 · 全生命周期 · OpenFOAM</div>",
                unsafe_allow_html=True)


def render_kpi_dashboard(result: CalculationResult):
    """顶部 KPI 仪表盘"""
    cols = st.columns(7)

    risk = result.fouling_risk
    if risk in (FoulingRisk.VERY_LOW, FoulingRisk.LOW):
        risk_badge = "badge-green"
    elif risk == FoulingRisk.MEDIUM:
        risk_badge = "badge-yellow"
    else:
        risk_badge = "badge-red"

    kpis = [
        ("综合评分", f"{result.overall:.0f}", "/100", ""),
        ("污染风险", risk.value.upper(), "", risk_badge),
        ("SEC", f"{result.sec_kwh_m3:.3f}", "kWh/m³", ""),
        ("膜面剪切", f"{result.avg_shear_pa:.2f}", "Pa", ""),
        ("运行成本", f"¥{result.total_cost:.3f}", "/m³", ""),
        ("膜寿命", f"{result.membrane_life_yr:.0f}", "年", ""),
        ("碳足迹", f"{result.carbon_kgco2:.3f}", "kgCO₂/m³", ""),
    ]

    for col, (label, value, unit, badge) in zip(cols, kpis):
        with col:
            if badge:
                value_html = f"<span class='{badge}'>{value}</span>"
            else:
                value_html = value
            st.markdown(f"""
            <div class='kpi-card'>
                <div class='kpi-label'>{label}</div>
                <div class='kpi-value'>{value_html}</div>
                <div class='kpi-unit'>{unit}</div>
            </div>
            """, unsafe_allow_html=True)


def render_calculation_result(result: CalculationResult, cfg: SimulationConfig):
    """主结果展示 - 标签页"""
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📋 综合报告", "🔬 膜污染分析", "🧪 生化处理", "💰 经济分析",
        "📊 图表", "💡 诊断建议"
    ])

    with tab1:
        render_summary_tab(result, cfg)

    with tab2:
        render_fouling_tab(result)

    with tab3:
        render_biochem_tab(result)

    with tab4:
        render_economics_tab(result, cfg)

    with tab5:
        render_charts_tab(result)

    with tab6:
        render_diagnosis_tab(result)


def render_summary_tab(result, cfg):
    """综合报告"""
    st.markdown("### 📋 运行参数与关键指标")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**运行参数**")
        rows = [
            ("曝气强度", f"{cfg.aeration_intensity:.0f} Nm³/m²/h ({cfg.aeration_mode.value})"),
            ("曝气孔径", f"{cfg.orifice_diameter_mm:.0f} mm"),
            ("膜片数 / 面积", f"{cfg.sheet_count} 片 / {cfg.membrane_area_m2:.0f} m²"),
            ("MLSS", f"{cfg.mlss_mg_l:.0f} mg/L"),
            ("通量", f"{cfg.target_flux_lmh:.1f} LMH"),
            ("SRT / HRT", f"{cfg.srt_days:.0f} 天 / {cfg.hrt_hours:.1f} h"),
            ("水温 / pH / DO", f"{cfg.temperature_c:.0f}°C / {cfg.ph:.1f} / {cfg.do_setpoint_mg_l:.1f} mg/L"),
        ]
        for label, val in rows:
            st.markdown(f"<div class='info-row'><span class='info-label'>{label}</span><span class='info-value'>{val}</span></div>", unsafe_allow_html=True)

    with col2:
        st.markdown("**关键指标**")
        rows = [
            ("α因子", f"{result.alpha_factor:.3f}"),
            ("KLa", f"{result.kla_actual:.1f} h⁻¹"),
            ("SAE", f"{result.sae:.1f} kgO₂/kWh"),
            ("气泡 d32", f"{result.bubble_d32_mm:.2f} mm"),
            ("气含率", f"{result.gas_holdup:.4f}"),
            ("循环流速", f"{result.crossflow_vel_ms:.3f} m/s"),
            ("膜丝振幅", f"{result.fiber_amplitude_mm:.2f} mm @ {result.fiber_frequency_hz:.1f} Hz"),
        ]
        for label, val in rows:
            st.markdown(f"<div class='info-row'><span class='info-label'>{label}</span><span class='info-value'>{val}</span></div>", unsafe_allow_html=True)


def render_fouling_tab(result):
    """膜污染分析"""
    st.markdown("### 🔬 膜污染与化学清洗")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**阻力串联模型**")
        resistances = [
            ("R_m (膜固有)", result.r_m, "×10¹² m⁻¹"),
            ("R_cake (滤饼层)", result.r_cake, "×10¹² m⁻¹"),
            ("R_pore (膜孔堵塞)", result.r_pore, "×10¹² m⁻¹"),
            ("R_irr (不可逆)", result.r_irr, "×10¹² m⁻¹"),
            ("R_total", result.r_total, "×10¹² m⁻¹"),
        ]
        for label, val, unit in resistances:
            st.markdown(f"<div class='info-row'><span class='info-label'>{label}</span><span class='info-value'>{val:.2f} {unit}</span></div>", unsafe_allow_html=True)

        # 阻力饼图
        labels = ["R_m", "R_cake", "R_pore", "R_irr"]
        values = [max(result.r_m, 0.01), max(result.r_cake, 0.01), max(result.r_pore, 0.01), max(result.r_irr, 0.01)]
        fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=0.5,
                                      marker=dict(colors=["#3366cc", "#ff9933", "#ff4444", "#9933cc"]),
                                      textinfo="label+percent")])
        fig.update_layout(height=250, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                          font=dict(color="#c0d8f0"), margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("**EPS / SMP 动态**")
        rows = [
            ("EPS", f"{result.eps_mg_gvss:.0f} mg/gVSS"),
            ("SMP", f"{result.smp_mg_l:.0f} mg/L"),
            ("PS/PN 比率", f"{result.ps_pn_ratio:.2f}"),
            ("SVI", f"{result.svi_ml_g:.0f} mL/g"),
        ]
        for label, val in rows:
            st.markdown(f"<div class='info-row'><span class='info-label'>{label}</span><span class='info-value'>{val}</span></div>", unsafe_allow_html=True)

        st.markdown("**污染速率**")
        st.markdown(f"<div class='info-row'><span class='info-label'>临界通量</span><span class='info-value'>{result.critical_flux_lmh:.1f} LMH</span></div>", unsafe_allow_html=True)
        st.markdown(f"<div class='info-row'><span class='info-label'>污染速率</span><span class='info-value'>{result.fouling_rate_pa_d:.0f} Pa/天</span></div>", unsafe_allow_html=True)
        st.markdown(f"<div class='info-row'><span class='info-label'>清洗周期</span><span class='info-value'>{result.cleaning_days:.0f} 天</span></div>", unsafe_allow_html=True)

    with col3:
        st.markdown("**化学清洗策略**")
        rows = [
            ("CEB 间隔", f"{result.ceb_frequency_days:.0f} 天"),
            ("CIP 间隔", f"{result.cip_frequency_days:.0f} 天"),
            ("NaClO 消耗", f"{result.naclo_consumption_kg_y:.1f} kg/年"),
            ("柠檬酸消耗", f"{result.citric_consumption_kg_y:.1f} kg/年"),
            ("清洗效率", f"{result.cleaning_efficiency_current:.0%}"),
            ("膜寿命", f"{result.membrane_life_yr:.0f} 年"),
        ]
        for label, val in rows:
            st.markdown(f"<div class='info-row'><span class='info-label'>{label}</span><span class='info-value'>{val}</span></div>", unsafe_allow_html=True)

        # TMP 演化图
        if result.tmp_evolution:
            days = list(range(len(result.tmp_evolution)))
            tmp_kpa = [t / 1000 for t in result.tmp_evolution]
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=days, y=tmp_kpa, mode="lines+markers",
                                     line=dict(color="#ff9933", width=2), name="TMP"))
            fig.update_layout(title="TMP 30天演化", height=250,
                              paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                              font=dict(color="#c0d8f0"),
                              xaxis=dict(title="天", gridcolor="rgba(255,255,255,0.05)"),
                              yaxis=dict(title="kPa", gridcolor="rgba(255,255,255,0.05)"),
                              margin=dict(l=0, r=0, t=30, b=0))
            st.plotly_chart(fig, use_container_width=True)


def render_biochem_tab(result):
    """生化处理"""
    st.markdown("### 🧪 生化处理效率 (简化ASM1)")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**处理效率**")
        effs = [
            ("COD", result.cod_eff),
            ("BOD", result.bod_eff),
            ("NH₄-N", result.nh4_eff),
            ("TN", result.tn_eff),
            ("TP", result.tp_eff),
        ]
        for label, val in effs:
            color = "#00c864" if val >= 90 else ("#ffb400" if val >= 70 else "#ff3c3c")
            st.markdown(f"""
            <div style="margin: 0.4rem 0;">
                <span style="color:#6b8aaa; font-size:0.85rem;">{label}</span>
                <div style="background:rgba(255,255,255,0.05); border-radius:4px; height:20px; margin-top:2px;">
                    <div style="background:{color}; width:{val}%; height:100%; border-radius:4px;
                         display:flex; align-items:center; justify-content:flex-end; padding-right:6px;">
                        <span style="font-size:0.7rem; font-weight:700; color:white;">{val:.1f}%</span>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    with col2:
        st.markdown("**出水水质**")
        rows = [
            ("出水 NO₃-N", f"{result.no3_effluent_mg_l:.1f} mg/L"),
            ("污泥产量", f"{result.sludge_kgds_d:.1f} kgDS/天"),
        ]
        for label, val in rows:
            st.markdown(f"<div class='info-row'><span class='info-label'>{label}</span><span class='info-value'>{val}</span></div>", unsafe_allow_html=True)

        # 评分
        st.markdown("**综合评分**")
        scores = [
            ("运行评分", result.op_score),
            ("优化评分", result.opt_score),
            ("可持续评分", result.sus_score),
            ("综合评分", result.overall),
        ]
        for label, val in scores:
            color = "#00c864" if val >= 80 else ("#ffb400" if val >= 60 else "#ff3c3c")
            st.markdown(f"""
            <div style="margin: 0.4rem 0;">
                <span style="color:#6b8aaa; font-size:0.8rem;">{label}</span>
                <span style="float:right; color:{color}; font-weight:700;">{val:.1f}</span>
                <div style="background:rgba(255,255,255,0.05); border-radius:4px; height:8px; margin-top:2px;">
                    <div style="background:{color}; width:{val}%; height:100%; border-radius:4px;"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)


def render_economics_tab(result, cfg):
    """经济分析"""
    st.markdown("### 💰 全生命周期成本分析")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**CAPEX**")
        st.markdown(f"<div style='font-size:2rem; font-weight:700; color:#00b4ff;'>¥{result.capex_total:,.0f}</div>", unsafe_allow_html=True)
        st.markdown(f"<span style='color:#6b8aaa;'>建设投资 ({cfg.capex_per_m3d_rmb:.0f} 元/m³/d)</span>", unsafe_allow_html=True)

    with col2:
        st.markdown("**OPEX 细分**")
        opex = [
            ("电费", result.energy_cost),
            ("膜更换", result.membrane_cost),
            ("化学品", result.chemical_cost),
            ("污泥处置", result.sludge_cost),
        ]
        for label, val in opex:
            st.markdown(f"<div class='info-row'><span class='info-label'>{label}</span><span class='info-value'>¥{val:.3f}/m³</span></div>", unsafe_allow_html=True)
        st.markdown(f"<div class='info-row' style='border-top:1px solid rgba(255,255,255,0.1);'><span class='info-label' style='font-weight:700;'>总成本</span><span class='info-value' style='color:#00c864;'>¥{result.total_cost:.3f}/m³</span></div>", unsafe_allow_html=True)

    with col3:
        st.markdown("**投资回报**")
        npv_color = "#00c864" if result.npv_rmb > 0 else "#ff3c3c"
        st.markdown(f"<div style='font-size:1.5rem; font-weight:700; color:{npv_color};'>NPV ¥{result.npv_rmb:,.0f}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='info-row'><span class='info-label'>投资回收期</span><span class='info-value'>{result.payback_years:.1f} 年</span></div>", unsafe_allow_html=True)
        st.markdown(f"<div class='info-row'><span class='info-label'>碳足迹</span><span class='info-value'>{result.carbon_kgco2:.3f} kgCO₂e/m³</span></div>", unsafe_allow_html=True)

    # 成本饼图
    labels = ["电费", "膜更换", "化学品", "污泥处置"]
    values = [max(result.energy_cost, 0.001), max(result.membrane_cost, 0.001),
              max(result.chemical_cost, 0.001), max(result.sludge_cost, 0.001)]
    fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=0.5,
                                  marker=dict(colors=["#ffb400", "#ff3c3c", "#9933cc", "#6b8aaa"]),
                                  textinfo="label+percent")])
    fig.update_layout(height=250, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(color="#c0d8f0"), margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig, use_container_width=True)


def render_charts_tab(result):
    """图表"""
    st.markdown("### 📊 可视化图表")

    viz = MBRVisualizer()

    col1, col2 = st.columns(2)

    with col1:
        # 雷达图
        fig_radar = viz.create_radar(result)
        if fig_radar:
            st.plotly_chart(fig_radar, use_container_width=True)

        # 仪表盘
        fig_gauge = viz.create_gauge(result)
        if fig_gauge:
            st.plotly_chart(fig_gauge, use_container_width=True)

    with col2:
        # 成本饼图
        fig_cost = viz.create_cost_breakdown(result)
        if fig_cost:
            st.plotly_chart(fig_cost, use_container_width=True)

        # TMP演化
        fig_tmp = viz.create_tmp_evolution(result)
        if fig_tmp:
            st.plotly_chart(fig_tmp, use_container_width=True)


def render_diagnosis_tab(result):
    """诊断建议"""
    st.markdown("### 💡 智能诊断与建议")

    # 根因分析
    if result.root_causes:
        st.markdown("#### 🔍 根因分析")
        for i, rc in enumerate(result.root_causes, 1):
            st.markdown(f"<div class='info-row'><span>{i}.</span><span>{rc}</span></div>", unsafe_allow_html=True)

    # 警告
    if result.warnings:
        st.markdown("#### ⚠️ 警告")
        for w in result.warnings:
            st.warning(w)

    # 分级建议
    if result.recommendations:
        st.markdown("#### 🎯 分级建议")
        for rec in result.recommendations:
            if "紧急" in rec:
                st.markdown(f"<div class='advice-urgent'>🔴 {rec}</div>", unsafe_allow_html=True)
            elif "优化" in rec:
                st.markdown(f"<div class='advice-optimize'>🟡 {rec}</div>", unsafe_allow_html=True)
            elif "长期" in rec:
                st.markdown(f"<div class='advice-longterm'>🔵 {rec}</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='advice-longterm'>✅ {rec}</div>", unsafe_allow_html=True)


def render_optimization_results():
    """优化结果"""
    opt = st.session_state.opt_result
    if not opt:
        return

    st.markdown("### 🔍 参数自动优化结果")

    bl = opt["baseline"]
    imp = opt["improvement"]

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("基准 SEC", f"{bl['sec']:.3f} kWh/m³")
    with col2:
        st.metric("改进 SEC", f"{imp['sec_delta']:+.3f} kWh/m³", f"{imp['sec_pct']:+.1f}%")
    with col3:
        st.metric("综合评分", f"{imp['overall_delta']:+.1f} 分")

    # Top 5 表格
    st.markdown("**Top 5 最优方案**")
    rows = []
    for i, s in enumerate(opt["solutions"][:5]):
        c = s["config"]
        rows.append({
            "排名": i + 1, "曝气": f"{c['aeration_intensity']:.0f}",
            "孔径": f"{c['orifice_diameter_mm']:.0f} mm",
            "通量": f"{c['target_flux_lmh']:.0f} LMH",
            "MLSS": f"{c['mlss_mg_l']:.0f} mg/L",
            "SEC": f"{s['sec']:.3f}", "风险": s['fouling_risk'],
            "评分": f"{s['overall']:.1f}",
        })
    st.dataframe(rows, use_container_width=True, hide_index=True)

    st.markdown(f"**推荐**: {imp['fouling_risk_change']}")

    # 推荐操作
    best = opt["solutions"][0]
    bc = best["config"]
    st.markdown("**推荐调整**:")
    st.code(f"曝气: {bc['aeration_intensity']:.0f} Nm³/m²/h | 孔径: {bc['orifice_diameter_mm']:.0f} mm | "
            f"通量: {bc['target_flux_lmh']:.0f} LMH | MLSS: {bc['mlss_mg_l']:.0f} mg/L")


def render_sensitivity_results():
    """灵敏度分析结果"""
    sens = st.session_state.sensitivity
    if not sens:
        return

    st.markdown("### 📊 灵敏度分析 (OAT ±20%)")

    rows = []
    for s in sens[:7]:
        rows.append({
            "参数": s["name"], "基准值": f"{s['base_value']:.4g}",
            "↓影响": f"{s['impact_low']:+.2f}", "↑影响": f"{s['impact_high']:+.2f}",
            "最大影响": f"{s['max_impact']:.2f}", "方向": s["direction"],
        })
    st.dataframe(rows, use_container_width=True, hide_index=True)

    st.info(f"💡 **关键参数**: '{sens[0]['name']}' 对综合评分影响最大 (±{sens[0]['max_impact']:.1f} 分)")


def render_comparison_results():
    """场景对比结果"""
    comp = st.session_state.comparison
    if not comp:
        return

    st.markdown("### 📋 多场景对比")

    results = comp["results"]
    rankings = comp["rankings"]

    rows = []
    for key, data in sorted(results.items(), key=lambda x: rankings.get(x[0], 99)):
        r = data["result"]
        c = data["config"]
        rows.append({
            "排名": rankings.get(key, "-"), "场景": data["name"],
            "曝气": f"{c.aeration_intensity:.0f}", "通量": f"{c.target_flux_lmh:.0f}",
            "SEC": f"{r.sec_kwh_m3:.3f}", "评分": f"{r.overall:.1f}",
            "风险": r.fouling_risk.value, "成本": f"¥{r.total_cost:.3f}",
        })

    st.dataframe(rows, use_container_width=True, hide_index=True)

    best_key = min(rankings, key=lambda k: rankings[k])
    best = results[best_key]
    st.success(f"🏆 推荐: **{best['name']}** (评分 {best['result'].overall:.1f}) | "
               f"SEC: {best['result'].sec_kwh_m3:.3f} | 成本: ¥{best['result'].total_cost:.3f}")


# ─────────────────────────────────────────────────────────
# 主入口
# ─────────────────────────────────────────────────────────
def main():
    init_session()

    render_header()
    render_sidebar()

    result = st.session_state.result

    # 首次加载自动计算
    if result is None:
        if st.session_state.config is None:
            st.session_state.config = SimulationConfig()
        with st.spinner("正在初始化计算..."):
            calc = MBREngineeringCalculator(st.session_state.config)
            st.session_state.result = calc.run_full_calculation()
        result = st.session_state.result
        st.rerun()

    if result:
        st.markdown("---")
        render_kpi_dashboard(result)

        st.markdown("---")
        render_calculation_result(result, st.session_state.config)

        # 优化结果
        if st.session_state.opt_result:
            st.markdown("---")
            render_optimization_results()

        # 灵敏度结果
        if st.session_state.sensitivity:
            st.markdown("---")
            render_sensitivity_results()

        # 场景对比结果
        if st.session_state.comparison:
            st.markdown("---")
            render_comparison_results()

    st.markdown("---")
    st.markdown("<div style='text-align:center; color:#4a6a85; font-size:0.75rem; padding:1rem;'>"
                "MBR 工业仿真系统 V8.0 | 阻力串联模型 · EPS/SMP · ASM1 · 全生命周期 · OpenFOAM</div>",
                unsafe_allow_html=True)


if __name__ == "__main__":
    main()