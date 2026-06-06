#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于历史模拟法和方差-协方差法的宁德时代VaR计算对比
"""

import yfinance as yf
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from datetime import datetime
import json
import os
import warnings
warnings.filterwarnings('ignore')

# ============ 字体配置 ============

plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 200

OUTPUT_DIR = 'D:/VaR_Results/'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============ 1. 获取数据 ============
print("=" * 60)
print("1. 获取宁德时代(300750.SZ)股票数据")
print("=" * 60)

ticker = yf.Ticker('300750.SZ')
df = ticker.history(start='2024-01-01', end='2026-05-28', auto_adjust=True)
df = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
df.index = df.index.tz_localize(None)

# 计算日收益率
df['Return'] = df['Close'].pct_change()
df = df.dropna()

print(f"数据时间范围: {df.index[0].strftime('%Y-%m-%d')} ~ {df.index[-1].strftime('%Y-%m-%d')}")
print(f"数据条数: {len(df)}")
print(f"最新收盘价: {df['Close'].iloc[-1]:.2f} 元")

# ============ 2. 描述性统计 ============
print("\n" + "=" * 60)
print("2. 收益率描述性统计")
print("=" * 60)

returns = df['Return'].values
desc_stats = {
    '样本数': len(returns),
    '均值': np.mean(returns),
    '标准差': np.std(returns, ddof=1),
    '偏度': stats.skew(returns),
    '峰度': stats.kurtosis(returns),
    '最小值': np.min(returns),
    '最大值': np.max(returns),
    '中位数': np.median(returns),
    'JB统计量': stats.jarque_bera(returns)[0],
    'JB p值': stats.jarque_bera(returns)[1],
}

for k, v in desc_stats.items():
    if isinstance(v, float):
        print(f"  {k}: {v:.6f}")
    else:
        print(f"  {k}: {v}")

# ============ 3. VaR 计算 ============
print("\n" + "=" * 60)
print("3. VaR计算")
print("=" * 60)

confidence_levels = [0.90, 0.95, 0.99]
window = 250  # 滚动窗口

# --- 3.1 历史模拟法 ---
print("\n--- 历史模拟法 ---")
hs_var_results = {}
for cl in confidence_levels:
    var_val = -np.percentile(returns, (1 - cl) * 100)
    hs_var_results[cl] = var_val
    print(f"  置信水平 {cl*100:.0f}%: VaR = {var_val:.6f} ({var_val*100:.4f}%)")

# --- 3.2 方差-协方差法（正态分布假设）---
print("\n--- 方差-协方差法（正态分布）---")
vc_var_results = {}
mu = np.mean(returns)
sigma = np.std(returns, ddof=1)
for cl in confidence_levels:
    z = stats.norm.ppf(cl)
    var_val = -(mu - z * sigma)
    vc_var_results[cl] = var_val
    print(f"  置信水平 {cl*100:.0f}%: VaR = {var_val:.6f} ({var_val*100:.4f}%), z = {z:.4f}")

# --- 3.3 方差-协方差法（t分布）---
print("\n--- 方差-协方差法（t分布）---")
df_t, loc_t, scale_t = stats.t.fit(returns)
vc_t_var_results = {}
for cl in confidence_levels:
    var_val = -stats.t.ppf(1 - cl, df=df_t, loc=loc_t, scale=scale_t)
    vc_t_var_results[cl] = var_val
    print(f"  置信水平 {cl*100:.0f}%: VaR = {var_val:.6f} ({var_val*100:.4f}%), t自由度 = {df_t:.2f}")

# ============ 4. 滚动VaR计算 ============
print("\n" + "=" * 60)
print("4. 滚动VaR计算")
print("=" * 60)

rolling_var_hs = pd.Series(index=df.index[window:], dtype=float)
rolling_var_vc = pd.Series(index=df.index[window:], dtype=float)
rolling_var_vc_t = pd.Series(index=df.index[window:], dtype=float)

cl_rolling = 0.95
z_95 = stats.norm.ppf(cl_rolling)

for i in range(window, len(returns)):
    window_returns = returns[i-window:i]
    # 历史模拟法
    rolling_var_hs.iloc[i-window] = -np.percentile(window_returns, (1 - cl_rolling) * 100)
    # 方差-协方差法（正态）
    mu_w = np.mean(window_returns)
    sigma_w = np.std(window_returns, ddof=1)
    rolling_var_vc.iloc[i-window] = -(mu_w - z_95 * sigma_w)
    # 方差-协方差法（t分布）
    df_tw, loc_tw, scale_tw = stats.t.fit(window_returns)
    rolling_var_vc_t.iloc[i-window] = -stats.t.ppf(1 - cl_rolling, df=df_tw, loc=loc_tw, scale=scale_tw)

print(f"滚动VaR计算完成，窗口大小: {window}, 置信水平: {cl_rolling*100:.0f}%")

# ============ 5. 回测检验（Kupiec检验）===============
print("\n" + "=" * 60)
print("5. 回测检验（Kupiec检验）")
print("=" * 60)

def kupiec_test(returns, var_values, cl):
    """Kupiec比例失效检验"""
    violations = (returns < -var_values).astype(int)
    n = len(returns)
    x = violations.sum()
    p = 1 - cl
    p_hat = x / n if n > 0 else 0
    
    if x == 0 or x == n:
        return None, None, p_hat, x
    
    lr = 2 * (x * np.log(p_hat / p) + (n - x) * np.log((1 - p_hat) / (1 - p)))
    p_value = 1 - stats.chi2.cdf(lr, df=1)
    
    return lr, p_value, p_hat, x

# 对全样本VaR进行回测
backtest_results = {}
for cl in confidence_levels:
    hs_violations = (returns < -hs_var_results[cl]).sum()
    vc_violations = (returns < -vc_var_results[cl]).sum()
    vc_t_violations = (returns < -vc_t_var_results[cl]).sum()
    
    expected = (1 - cl) * len(returns)
    
    backtest_results[cl] = {
        'hs_violations': int(hs_violations),
        'vc_violations': int(vc_violations),
        'vc_t_violations': int(vc_t_violations),
        'expected': expected,
        'total': len(returns)
    }
    
    # Kupiec检验
    lr_hs, pv_hs, ph_hs, x_hs = kupiec_test(returns, np.full(len(returns), hs_var_results[cl]), cl)
    lr_vc, pv_vc, ph_vc, x_vc = kupiec_test(returns, np.full(len(returns), vc_var_results[cl]), cl)
    lr_vc_t, pv_vc_t, ph_vc_t, x_vc_t = kupiec_test(returns, np.full(len(returns), vc_t_var_results[cl]), cl)
    
    backtest_results[cl]['hs_lr'] = lr_hs
    backtest_results[cl]['hs_pvalue'] = pv_hs
    backtest_results[cl]['vc_lr'] = lr_vc
    backtest_results[cl]['vc_pvalue'] = pv_vc
    backtest_results[cl]['vc_t_lr'] = lr_vc_t
    backtest_results[cl]['vc_t_pvalue'] = pv_vc_t
    
    print(f"\n置信水平 {cl*100:.0f}%:")
    print(f"  预期突破次数: {expected:.1f}")
    print(f"  历史模拟法: 突破{int(hs_violations)}次, LR={lr_hs:.4f}, p={pv_hs:.4f}" if lr_hs else f"  历史模拟法: 突破{int(hs_violations)}次")
    print(f"  方差-协方差法(正态): 突破{int(vc_violations)}次, LR={lr_vc:.4f}, p={pv_vc:.4f}" if lr_vc else f"  方差-协方差法(正态): 突破{int(vc_violations)}次")
    print(f"  方差-协方差法(t分布): 突破{int(vc_t_violations)}次, LR={lr_vc_t:.4f}, p={pv_vc_t:.4f}" if lr_vc_t else f"  方差-协方差法(t分布): 突破{int(vc_t_violations)}次")

# ============ 6. 生成图表 ============
print("\n" + "=" * 60)
print("6. 生成图表")
print("=" * 60)

# --- 图1: 收盘价走势图 ---
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(df.index, df['Close'], color='#2E86AB', linewidth=1.2, label='收盘价')
ax.fill_between(df.index, df['Low'], df['High'], alpha=0.15, color='#2E86AB', label='日内波幅')
ax.set_title('宁德时代(300750.SZ)收盘价走势', fontsize=14, fontweight='bold')
ax.set_xlabel('日期', fontsize=11)
ax.set_ylabel('价格(元)', fontsize=11)
ax.legend(loc='best', fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/fig1_price_trend.png', bbox_inches='tight')
plt.close()
print("  图1: 收盘价走势图 已保存")

# --- 图2: 日收益率时序图 ---
fig, ax = plt.subplots(figsize=(10, 5))
colors = ['#E74C3C' if r < 0 else '#27AE60' for r in returns]
ax.bar(df.index, returns, color=colors, width=1.5, alpha=0.7)
ax.axhline(y=0, color='black', linewidth=0.5)
ax.set_title('宁德时代日收益率时序图', fontsize=14, fontweight='bold')
ax.set_xlabel('日期', fontsize=11)
ax.set_ylabel('收益率', fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/fig2_returns_timeseries.png', bbox_inches='tight')
plt.close()
print("  图2: 日收益率时序图 已保存")

# --- 图3: 收益率分布直方图 + 正态拟合 + t分布拟合 ---
fig, ax = plt.subplots(figsize=(10, 6))
x_range = np.linspace(returns.min(), returns.max(), 300)

ax.hist(returns, bins=50, density=True, alpha=0.6, color='#3498DB', edgecolor='white', label='实际分布')
ax.plot(x_range, stats.norm.pdf(x_range, mu, sigma), 'r-', linewidth=2, label='正态分布拟合')
ax.plot(x_range, stats.t.pdf(x_range, df=df_t, loc=loc_t, scale=scale_t), 'g--', linewidth=2, label=f't分布拟合(自由度={df_t:.1f})')

ax.set_title('宁德时代日收益率分布与理论分布拟合', fontsize=14, fontweight='bold')
ax.set_xlabel('收益率', fontsize=11)
ax.set_ylabel('概率密度', fontsize=11)
ax.legend(loc='best', fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/fig3_return_distribution.png', bbox_inches='tight')
plt.close()
print("  图3: 收益率分布直方图 已保存")

# --- 图4: QQ图 ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# 正态QQ图
stats.probplot(returns, dist='norm', plot=ax1)
ax1.set_title('正态Q-Q图', fontsize=13, fontweight='bold')
ax1.get_lines()[0].set(markerfacecolor='#3498DB', markeredgecolor='#3498DB', markersize=3)
ax1.get_lines()[1].set(color='#E74C3C', linewidth=2)

# t分布QQ图
theoretical_q = stats.t.ppf(np.linspace(0.01, 0.99, len(returns)), df=df_t)
sorted_returns = np.sort(returns)
ax2.scatter(theoretical_q, sorted_returns, s=5, color='#3498DB', alpha=0.6)
lims = [min(theoretical_q.min(), sorted_returns.min()), max(theoretical_q.max(), sorted_returns.max())]
ax2.plot(lims, lims, 'r-', linewidth=2)
ax2.set_title(f't分布Q-Q图(自由度={df_t:.1f})', fontsize=13, fontweight='bold')
ax2.set_xlabel('理论分位数')
ax2.set_ylabel('样本分位数')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/fig4_qq_plot.png', bbox_inches='tight')
plt.close()
print("  图4: QQ图 已保存")

# --- 图5: 滚动VaR对比图 ---
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(df.index[window:], -returns[window:], color='gray', alpha=0.4, linewidth=0.5, label='日损失率')
ax.plot(rolling_var_hs.index, rolling_var_hs.values, color='#E74C3C', linewidth=1.5, label='历史模拟法VaR')
ax.plot(rolling_var_vc.index, rolling_var_vc.values, color='#2E86AB', linewidth=1.5, label='方差-协方差法VaR(正态)')
ax.plot(rolling_var_vc_t.index, rolling_var_vc_t.values, color='#27AE60', linewidth=1.5, linestyle='--', label='方差-协方差法VaR(t分布)')
ax.set_title(f'滚动VaR对比(95%置信水平, 窗口={window}天)', fontsize=14, fontweight='bold')
ax.set_xlabel('日期', fontsize=11)
ax.set_ylabel('VaR / 损失率', fontsize=11)
ax.legend(loc='best', fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/fig5_rolling_var.png', bbox_inches='tight')
plt.close()
print("  图5: 滚动VaR对比图 已保存")

# --- 图6: VaR突破点可视化 ---
fig, ax = plt.subplots(figsize=(12, 6))
actual_losses = -returns[window:]
hs_var_vals = rolling_var_hs.values
vc_var_vals = rolling_var_vc.values
vc_t_var_vals = rolling_var_vc_t.values

ax.plot(df.index[window:], actual_losses, color='gray', alpha=0.5, linewidth=0.5, label='日损失率')

# 标出历史模拟法突破点
hs_violations_mask = actual_losses > hs_var_vals
ax.scatter(df.index[window:][hs_violations_mask], actual_losses[hs_violations_mask], 
           color='#E74C3C', s=20, zorder=5, label=f'历史模拟法突破点({hs_violations_mask.sum()}次)')

# 标出方差-协方差法突破点
vc_violations_mask = actual_losses > vc_var_vals
ax.scatter(df.index[window:][vc_violations_mask], actual_losses[vc_violations_mask],
           color='#2E86AB', s=20, zorder=5, marker='^', label=f'方差-协方差法突破点({vc_violations_mask.sum()}次)')

ax.plot(rolling_var_hs.index, rolling_var_hs.values, color='#E74C3C', linewidth=1.2, alpha=0.8)
ax.plot(rolling_var_vc.index, rolling_var_vc.values, color='#2E86AB', linewidth=1.2, alpha=0.8)

ax.set_title(f'VaR回测突破点可视化(95%置信水平)', fontsize=14, fontweight='bold')
ax.set_xlabel('日期', fontsize=11)
ax.set_ylabel('损失率 / VaR', fontsize=11)
ax.legend(loc='best', fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/fig6_var_violations.png', bbox_inches='tight')
plt.close()
print("  图6: VaR突破点可视化 已保存")

# --- 图7: 不同置信水平下VaR对比柱状图 ---
fig, ax = plt.subplots(figsize=(8, 5))
x_pos = np.arange(len(confidence_levels))
width = 0.25

hs_vals = [hs_var_results[cl] * 100 for cl in confidence_levels]
vc_vals = [vc_var_results[cl] * 100 for cl in confidence_levels]
vc_t_vals = [vc_t_var_results[cl] * 100 for cl in confidence_levels]

bars1 = ax.bar(x_pos - width, hs_vals, width, label='历史模拟法', color='#E74C3C', alpha=0.85)
bars2 = ax.bar(x_pos, vc_vals, width, label='方差-协方差法(正态)', color='#2E86AB', alpha=0.85)
bars3 = ax.bar(x_pos + width, vc_t_vals, width, label='方差-协方差法(t分布)', color='#27AE60', alpha=0.85)

# 添加数值标签
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}%', xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)

ax.set_xlabel('置信水平', fontsize=11)
ax.set_ylabel('VaR (%)', fontsize=11)
ax.set_title('不同置信水平下VaR对比', fontsize=14, fontweight='bold')
ax.set_xticks(x_pos)
ax.set_xticklabels([f'{cl*100:.0f}%' for cl in confidence_levels])
ax.legend(loc='best', fontsize=10)
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/fig7_var_comparison_bar.png', bbox_inches='tight')
plt.close()
print("  图7: VaR对比柱状图 已保存")

# ============ 7. 保存计算结果为JSON ============
results_data = {
    'data_info': {
        'stock': '宁德时代(300750.SZ)',
        'start_date': df.index[0].strftime('%Y-%m-%d'),
        'end_date': df.index[-1].strftime('%Y-%m-%d'),
        'sample_size': len(returns),
        'latest_price': float(df['Close'].iloc[-1])
    },
    'descriptive_stats': {k: float(v) for k, v in desc_stats.items()},
    'var_results': {
        'historical_simulation': {f'{cl}': float(hs_var_results[cl]) for cl in confidence_levels},
        'variance_covariance_normal': {f'{cl}': float(vc_var_results[cl]) for cl in confidence_levels},
        'variance_covariance_t': {f'{cl}': float(vc_t_var_results[cl]) for cl in confidence_levels},
    },
    't_distribution_params': {
        'df': float(df_t),
        'loc': float(loc_t),
        'scale': float(scale_t)
    },
    'normal_params': {
        'mu': float(mu),
        'sigma': float(sigma)
    },
    'backtest': {}
}

for cl in confidence_levels:
    results_data['backtest'][f'{cl}'] = {
        'expected_violations': float(backtest_results[cl]['expected']),
        'hs_violations': backtest_results[cl]['hs_violations'],
        'vc_violations': backtest_results[cl]['vc_violations'],
        'vc_t_violations': backtest_results[cl]['vc_t_violations'],
        'total': backtest_results[cl]['total'],
        'hs_lr': float(backtest_results[cl]['hs_lr']) if backtest_results[cl]['hs_lr'] else None,
        'hs_pvalue': float(backtest_results[cl]['hs_pvalue']) if backtest_results[cl]['hs_pvalue'] else None,
        'vc_lr': float(backtest_results[cl]['vc_lr']) if backtest_results[cl]['vc_lr'] else None,
        'vc_pvalue': float(backtest_results[cl]['vc_pvalue']) if backtest_results[cl]['vc_pvalue'] else None,
        'vc_t_lr': float(backtest_results[cl]['vc_t_lr']) if backtest_results[cl]['vc_t_lr'] else None,
        'vc_t_pvalue': float(backtest_results[cl]['vc_t_pvalue']) if backtest_results[cl]['vc_t_pvalue'] else None,
    }

with open(f'{OUTPUT_DIR}/var_results.json', 'w', encoding='utf-8') as f:
    json.dump(results_data, f, ensure_ascii=False, indent=2)

print("完成!")
