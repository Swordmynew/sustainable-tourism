"""
可持续旅游示例 - 动态经济评分 & 再投资机制
（已直接在 visitor_number_forecast 中进行修正，避免负值）
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import product
import re
import warnings
from scipy.optimize import minimize
from mpl_toolkits.mplot3d import Axes3D  # For 3D plotting

# ★改用 SARIMAX，并设置 m=2 表示两季
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings("ignore")

# -------------------- 数据读取 --------------------
file_paths = [
    "1.xlsx",  # Impact of tourism on local residents
    "2.xlsx",  # Basic information of Juneau
    "3.xlsx",  # Local residents' positive and negative views on tourism
    "4.xlsx",  # Average number of tourism-related workers per household
    "5.xlsx",  # Impact of tourists on residents in 2023
    "6.xlsx"   # Cruise revenue in 2023
]

data = []
for file in file_paths:
    try:
        df = pd.read_excel(file, sheet_name='Sheet1')
        df.columns = df.columns.str.strip()
        data.append(df)
    except FileNotFoundError:
        print(f"Error: The file '{file}' was not found.")
        exit(1)
    except Exception as e:
        print(f"Error reading '{file}': {e}")
        exit(1)

# -------------------- 数据提取与清洗 --------------------
# 1. Impact of tourism on local residents (1.xlsx)
impact_data = data[0][['Impact', 'Very affected (%)', 'Somewhat affected (%)',
                       'Very + Somewhat Affected (%)', 'Not affected (%)', "Don't know (%)"]]

for col in ['Very affected (%)', 'Somewhat affected (%)', 'Very + Somewhat Affected (%)',
            'Not affected (%)', "Don't know (%)"]:
    impact_data[col] = pd.to_numeric(
        impact_data[col].astype(str).str.replace('%', '').str.strip(),
        errors='coerce'
    )
impact_data['Total Affected (%)'] = impact_data['Very affected (%)'] + impact_data['Somewhat affected (%)']

# 将此初始值视为“环境越好分数越大”时的**初始值**（过去逻辑是破坏分数）
environmental_score_initial = impact_data['Total Affected (%)'].mean()

# 2. Basic information of Juneau (2.xlsx)
basic_info = data[1]
print("Columns in 2.xlsx Sheet1 after stripping:")
print(basic_info.columns.tolist())

expected_columns = {
    'Per_Capita_Income_USD': ['人均收入（美元）', '人均收入 (美元)', '人均收入USD', 'Income (USD)'],
    'Temperature_F': ['气温（℉）', '气温 (℉)', 'Temperature (F)', 'Temperature_F'],
    'Population': ['人口（人）', '人口', 'Population'],
    'Number_of_Tourists_per_year': ['游客数量（人/年）', '游客数量 (人/年)', 'Number of Tourists (per year)',
                                    'Number of Tourists']
}


def find_column(df, possible_names):
    for name in possible_names:
        if name in df.columns:
            return name
    raise KeyError(f"None of the expected columns {possible_names} found in DataFrame.")


try:
    income_col = find_column(basic_info, expected_columns['Per_Capita_Income_USD'])
    temp_col = find_column(basic_info, expected_columns['Temperature_F'])
    population_col = find_column(basic_info, expected_columns['Population'])
    tourists_col = find_column(basic_info, expected_columns['Number_of_Tourists_per_year'])

    for col in [income_col, temp_col, population_col, tourists_col]:
        basic_info[col] = pd.to_numeric(
            basic_info[col].astype(str).str.replace(',', '').str.strip(),
            errors='coerce'
        )

    average_income = basic_info[income_col].mean()
    average_temperature = basic_info[temp_col].mean()
    average_population = basic_info[population_col].mean()
    average_visitors = basic_info[tourists_col].mean()

    total_resident_income = average_income * average_population
    print(f"Total Resident Income: ${total_resident_income:,.2f}")

except KeyError as e:
    print(f"Error: {e}")
    print("请检查2.xlsx中的列名，并确保它们与预期的一致。")
    exit(1)

# 3. Local residents' positive and negative views on tourism (3.xlsx)
social_impact_data = data[2]
for col in ['积极影响（%）', '消极影响（%）']:
    social_impact_data[col] = pd.to_numeric(
        social_impact_data[col].astype(str).str.replace('%', '').str.strip(),
        errors='coerce'
    )
social_positive = social_impact_data['积极影响（%）'].mean()
social_negative = social_impact_data['消极影响（%）'].mean()
social_score_initial = social_positive - social_negative

# 4. Average number of tourism-related workers per household (4.xlsx)
employment_data = data[3]
employment_col = 'Household Member Employed (%)'
if employment_col in employment_data.columns:
    def clean_employment_data(entry):
        if isinstance(entry, str):
            nums = re.findall(r'-?\d+\.?\d*', entry)
            if nums:
                return float(nums[0])
            else:
                return np.nan
        elif isinstance(entry, (int, float)):
            return entry
        else:
            return np.nan

    employment_data[employment_col] = employment_data[employment_col].apply(clean_employment_data)
    employment_data[employment_col] = pd.to_numeric(employment_data[employment_col], errors='coerce')
    employment_data[employment_col].fillna(employment_data[employment_col].mean(), inplace=True)
    average_employment = employment_data[employment_col].mean()
else:
    raise KeyError(f"列 '{employment_col}' 未在 4.xlsx 的 Sheet1 中找到。")

# 5. Impact of tourists on residents in 2023 (5.xlsx)
impact_changes = data[4]
for col in ['Change 2022-23 (%)']:
    impact_changes[col] = pd.to_numeric(
        impact_changes[col].astype(str).str.replace('%', '').str.strip(),
        errors='coerce'
    )
average_impact_change = impact_changes['Change 2022-23 (%)'].mean()

# 6. Cruise revenue in 2023 (6.xlsx)
revenue_data = data[5]
if 'Revenues' in revenue_data.columns:
    revenue_data['Revenues'] = pd.to_numeric(
        revenue_data['Revenues'].astype(str).str.replace(',', '').str.strip(),
        errors='coerce'
    )
    cruise_revenue = revenue_data['Revenues'].sum()
    print(f"Cruise Revenue: ${cruise_revenue:,.2f}")
else:
    raise KeyError("列 'Revenues' 未在 6.xlsx 的 Sheet1 中找到。")

total_revenue = (average_income * average_population) + cruise_revenue
print(f"Total Revenue: ${total_revenue:,.2f}")

# 每位游客平均消费
average_spending = cruise_revenue / 1600000
print(f"Average Spending per Visitor: ${average_spending:.2f}")

# -------------------- 模型参数设置 --------------------
ENVIRONMENTAL_IMPACT_PER_DOLLAR = 0.1
INFRASTRUCTURE_IMPACT_PER_DOLLAR = 0.02
POLICY_EFFECTIVENESS_SCALE = 5000
POLICY_IMPACT_PER_DOLLAR_SOCIAL = 0.0001

# -------------------- 环境和社会影响模型（已修改，环境与社会互相影响） --------------------
def environmental_impact_model(environmental_score, additional_expenditure, social_score):
    """
    更平衡：synergy_factor=0.05（低于原先0.1），
    截断从-50改为-20，让负值存在但不会无限下挫。
    """
    # 基础环境改善(对数形式不变)
    improvement_base = ENVIRONMENTAL_IMPACT_PER_DOLLAR * np.log1p(additional_expenditure)

    synergy_factor = 0.05  # 下调以免环境得分过度放大
    # 假设 social_score=10 为中性水平
    social_adjust = synergy_factor * (social_score - 10)

    improvement = improvement_base + social_adjust
    new_environmental_score = environmental_score + improvement

    # 在极端情况下，最低不低于-20
    new_environmental_score = max(-20, new_environmental_score)
    return new_environmental_score


def social_impact_model(social_score, infrastructure_investment, policy_support, environment_score):
    """
    同理 synergy_factor=0.05，截断值从-50改为-20。
    """
    infrastructure_improvement = INFRASTRUCTURE_IMPACT_PER_DOLLAR * np.log1p(infrastructure_investment)
    policy_improvement = POLICY_IMPACT_PER_DOLLAR_SOCIAL * policy_support

    synergy_factor = 0.05
    # environment_score=10为中性水平
    env_adjust = synergy_factor * (environment_score - 10)

    new_social_score = social_score + infrastructure_improvement + policy_improvement + env_adjust
    new_social_score = max(-20, new_social_score)
    return new_social_score


# ========== 修改点 2: 放大 economic_score_model 中对环境/社会的依赖 ==========

def economic_score_model(prev_econ_score, current_revenue, environment_score, social_score, alpha=0.5):
    """
    synergy_factor 从0.01提高到0.03，让经济评分对环、社更敏感。
    依旧保留对 revenue 的比重，但容许一定负值范围(-10)。
    """
    base_part = (current_revenue / total_revenue)

    synergy_factor = 0.03  # 提高以平衡环境、社会的主导地位
    bonus = synergy_factor * (environment_score + social_score)

    new_score = alpha * prev_econ_score + (1 - alpha) * (base_part + bonus)
    new_score = max(-10, new_score)  # 允许一定负值
    return new_score

# -------------------- 动态权重函数 --------------------
def get_dynamic_weights(season, economic_score, env_score, soc_score):
    """
    在季节上做小幅度(1.1)增减，而不是1.2，避免对环境/经济过度偏好。
    其余逻辑同之前的“评分+10后归一”，减少过多减法。
    """
    econ_val = economic_score + 10
    env_val  = env_score + 10
    soc_val  = soc_score + 10
    sta_val  = 10.0

    # peak 季节对环境加 1.1 倍；off 季节对经济加 1.1 倍
    if season == 'peak':
        env_val *= 1.1
    else:
        econ_val *= 1.1

    total = econ_val + env_val + soc_val + sta_val
    if total <= 0:
        return {
            'economic': 0.25,
            'environment': 0.25,
            'social': 0.25,
            'stability': 0.25
        }

    return {
        'economic':   econ_val / total,
        'environment': env_val / total,
        'social':     soc_val / total,
        'stability':  sta_val / total
    }

# -------------------- 可持续性评分公式 --------------------
def sustainable_tourism_score(weights, economic_score, environment_score, social_score, stability):
    score = (
        weights['economic'] * economic_score
        + weights['environment'] * environment_score
        + weights['social'] * social_score
        + weights['stability'] * stability
    )
    return score

# -------------------- 三维图表绘制函数 --------------------
def plot_3d_environment_social_sustainability(df):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(df['Environment Score'], df['Social Score'], df['Sustainability Score'],
               c='b', marker='o', alpha=0.6)
    ax.set_xlabel('Environment Score')
    ax.set_ylabel('Social Score')
    ax.set_zlabel('Sustainability Score')
    ax.set_title('Environment vs. Social vs. Sustainability Score')
    plt.show()


def plot_3d_investment_sustainability(allocation_history):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(allocation_history['Environment Investment ($)'],
               allocation_history['Infrastructure Investment ($)'],
               allocation_history['Sustainability Score'],
               c='g', marker='^', alpha=0.6)
    ax.set_xlabel('Environment Investment ($)')
    ax.set_ylabel('Infrastructure Investment ($)')
    ax.set_zlabel('Sustainability Score')
    ax.set_title('Investment Allocation vs. Sustainability Score')
    plt.show()


def plot_3d_visitors_revenue_sustainability(df):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(df['Number of Tourists'], df['Revenue Adjusted ($)'], df['Sustainability Score'],
               c='r', marker='s', alpha=0.6)
    ax.set_xlabel('Number of Tourists')
    ax.set_ylabel('Revenue Adjusted ($)')
    ax.set_zlabel('Sustainability Score')
    ax.set_title('Number of Tourists vs. Revenue vs. Sustainability Score')
    plt.show()

# -------------------- 计算初始稳定性 --------------------
try:
    visitor_variability = basic_info[tourists_col].std() / average_visitors
    employment_variability = employment_data['Household Member Employed (%)'].std() / average_employment
    stability_initial = 1 - (visitor_variability + employment_variability) / 2
except Exception as e:
    print(f"Error calculating initial stability: {e}")
    stability_initial = 0.8

# -------------------- 初始经济评分 --------------------
base_revenue_ratio = (average_visitors * average_spending) / total_revenue
economic_score_initial = base_revenue_ratio

# 演示：计算峰季、淡季可持续评分
weights_peak = get_dynamic_weights('peak', economic_score_initial, environmental_score_initial, social_score_initial)
weights_off = get_dynamic_weights('off', economic_score_initial, environmental_score_initial, social_score_initial)

peak_score = sustainable_tourism_score(
    weights_peak, economic_score_initial, environmental_score_initial, social_score_initial, stability_initial
)
off_peak_score = sustainable_tourism_score(
    weights_off, economic_score_initial, environmental_score_initial, social_score_initial, stability_initial
)
print("Peak Season Sustainability Score:", peak_score)
print("Off-Peak Season Sustainability Score:", off_peak_score)

# 打印正在优化的因素 & 约束
print("正在优化的因素: [Environment Investment, Infrastructure Investment, Policy Support]")
print("约束: ")
print(f"  1) Environmental Score >= env_threshold")
print(f"  2) Social Score >= soc_threshold")
print(f"  3) Total Investment <= budget")

def sustainability_score_optimized(dynamic_weights, econ_score, env_score, soc_score, stability):
    return sustainable_tourism_score(dynamic_weights, econ_score, env_score, soc_score, stability)

# -------------------- 优化函数（修改为互相影响版本） --------------------
def optimize_sustainability_with_history(budget, env_threshold=10, soc_threshold=20,
                                         scale=POLICY_EFFECTIVENESS_SCALE):
    x0 = [budget * 0.2, budget * 0.3, budget * 0.5]

    def objective(x):
        env_expenditure, infra_expenditure, policy_expenditure = x

        # 1) 先用 旧social_score 来更新环境
        tmp_env_score = environmental_impact_model(
            environmental_score_initial,
            env_expenditure,
            social_score_initial
        )
        # 2) 再用 新环境评分 来更新社会
        tmp_soc_score = social_impact_model(
            social_score_initial,
            infra_expenditure,
            policy_expenditure,
            tmp_env_score
        )

        visitor_number = average_visitors / (1 + policy_expenditure / scale)
        if visitor_number < 0:
            visitor_number = 0

        current_revenue = visitor_number * average_spending
        new_econ_score = economic_score_model(
            economic_score_initial,
            current_revenue,
            tmp_env_score,
            tmp_soc_score,
            alpha=0.5
        )

        w_dyn = get_dynamic_weights('peak', new_econ_score, tmp_env_score, tmp_soc_score)
        score_ = sustainability_score_optimized(w_dyn, new_econ_score, tmp_env_score, tmp_soc_score, stability_initial)
        return -score_

    # 约束中也要用互相影响后的评分
    constraints = [
        {
            'type': 'ineq',
            'fun': lambda x: (
                environmental_impact_model(
                    environmental_score_initial,
                    x[0],
                    social_score_initial
                )
            ) - env_threshold
        },
        {
            'type': 'ineq',
            'fun': lambda x: (
                social_impact_model(
                    social_score_initial,
                    x[1],
                    x[2],
                    environmental_impact_model(
                        environmental_score_initial,
                        x[0],
                        social_score_initial
                    )
                )
            ) - soc_threshold
        },
        {
            'type': 'ineq',
            'fun': lambda x: budget - (x[0] + x[1] + x[2])
        },
    ]

    bounds = [(0, budget), (0, budget), (0, budget)]
    result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)

    if result.success:
        env_expenditure, infra_expenditure, policy_expenditure = result.x
        # 重新计算最终评分
        tmp_env_score = environmental_impact_model(
            environmental_score_initial,
            env_expenditure,
            social_score_initial
        )
        tmp_soc_score = social_impact_model(
            social_score_initial,
            infra_expenditure,
            policy_expenditure,
            tmp_env_score
        )

        visitor_number = average_visitors / (1 + policy_expenditure / scale)
        if visitor_number < 0:
            visitor_number = 0

        current_revenue = visitor_number * average_spending
        new_econ_score = economic_score_model(
            economic_score_initial,
            current_revenue,
            tmp_env_score,
            tmp_soc_score,
            alpha=0.5
        )

        w_dyn = get_dynamic_weights('peak', new_econ_score, tmp_env_score, tmp_soc_score)
        final_score = sustainability_score_optimized(w_dyn, new_econ_score, tmp_env_score, tmp_soc_score, stability_initial)

        best_allocation = {
            'Environment Investment ($)': round(env_expenditure, 2),
            'Infrastructure Investment ($)': round(infra_expenditure, 2),
            'Policy Support ($)': round(policy_expenditure, 2),
            'Economic Score': round(new_econ_score, 4),
            'Final Environment Score': round(tmp_env_score, 2),
            'Final Social Score': round(tmp_soc_score, 2),
            'Sustainability Score': round(final_score, 2)
        }
        print("Optimization successful!")
        allocation_history = pd.DataFrame([best_allocation])
    else:
        print("Optimization failed. Using default allocation.")
        best_allocation = None
        allocation_history = pd.DataFrame()

    return best_allocation, allocation_history

total_budget = 100000
optimal_allocation, allocation_history = optimize_sustainability_with_history(total_budget)
print("Optimal Investment Allocation:")
print(optimal_allocation)

# -------------------- 敏感度分析（保持不变，只是更大采样） --------------------
def sensitivity_analysis_combined():
    """
    对经济评分(econ)、环境评分(env)、社会评分(soc)、稳定性(stab)进行大范围敏感度采样，
    并输出 Pearson & Spearman 相关系数以查看三大评分是否呈现非 0 相关。
    """
    # 将 econ_range 从 ±1.0 扩到 ±3.0
    econ_range = 30
    # 将 env_range 和 soc_range 扩到 ±40
    env_range = 40
    soc_range = 40

    # 采样步数从 11 提高到 21，以获得更高分辨率
    econ_steps = 11
    env_steps = 11
    soc_steps = 11
    # 稳定性步数可保持不变，或略增
    stab_steps = 5

    econ_factors = np.linspace(economic_score_initial - econ_range,
                               economic_score_initial + econ_range,
                               econ_steps)
    env_factors = np.linspace(environmental_score_initial - env_range,
                              environmental_score_initial + env_range,
                              env_steps)
    social_factors = np.linspace(social_score_initial - soc_range,
                                 social_score_initial + soc_range,
                                 soc_steps)
    stability_factors = np.linspace(0.6, 1.0, stab_steps)

    results = []
    for econ, env, soc, stab in product(econ_factors, env_factors, social_factors, stability_factors):
        w_dyn = get_dynamic_weights('peak', econ, env, soc)
        score_ = sustainable_tourism_score(w_dyn, econ, env, soc, stab)
        results.append({
            'Economic Score': econ,
            'Environment Score': env,
            'Social Score': soc,
            'Stability': stab,
            'Sustainability Score': score_
        })

    df_results = pd.DataFrame(results)

    # 计算 Spearman & Pearson 相关
    corr_spearman = df_results.corr(method='spearman')
    corr_pearson = df_results.corr(method='pearson')

    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    sns.heatmap(corr_spearman, annot=True, cmap='coolwarm', fmt=".2f")
    # plt.title('Spearman Correlation Matrix')

    plt.subplot(1, 2, 2)
    sns.heatmap(corr_pearson, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Pearson Correlation Matrix')
    plt.tight_layout()
    plt.show()

    print("Spearman Correlation with Sustainability Score:")
    print(corr_spearman['Sustainability Score'].sort_values(ascending=False))
    print("\nPearson Correlation with Sustainability Score:")
    print(corr_pearson['Sustainability Score'].sort_values(ascending=False))

    # 其他可视化保持不变
    sns.pairplot(df_results,
                 vars=['Economic Score', 'Environment Score', 'Social Score', 'Stability'],
                 kind='scatter', diag_kind='kde')
    plt.suptitle('Pairplot of Factors vs. Sustainability Score', y=1.02)
    plt.show()

    plt.figure(figsize=(12, 6))
    sns.lineplot(x='Environment Score', y='Sustainability Score', data=df_results, label='Environment')
    sns.lineplot(x='Economic Score', y='Sustainability Score', data=df_results, label='Economic')
    sns.lineplot(x='Social Score', y='Sustainability Score', data=df_results, label='Social')
    sns.lineplot(x='Stability', y='Sustainability Score', data=df_results, label='Stability')
    plt.title('Sensitivity Analysis')
    plt.xlabel('Factor Value')
    plt.ylabel('Sustainability Score')
    plt.legend()
    plt.show()

    # 如果需要 3D 图
    plot_3d_environment_social_sustainability(df_results)

    return df_results

df_results = sensitivity_analysis_combined()
print("从相关性矩阵可看出：环境评分、社会评分与经济评分彼此影响更明显；具体结果需结合模型参数与实际数据。")

# -------------------- SARIMAX(m=2) 预测模型 (已加对数变换和截断) --------------------
def visitor_number_forecast(historical_visitors, policy_support_invest,
                            scale=POLICY_EFFECTIVENESS_SCALE, periods=5):
    """
    使用 SARIMAX(季节周期 m=2) 来预测游客数量，并考虑 policy_support_invest 对游客数量的影响。
    对历史数据做对数变换 -> 反变换时做截断 => 避免出现负游客。
    """
    try:
        # 如果尚未 fit，就 fit 一个对数模型
        if not hasattr(visitor_number_forecast, "model_fit_sarimax"):
            # 防止历史数据里有0或负值 => 取对数时出错
            hist_vis_log = np.log1p(historical_visitors.clip(lower=0))

            model = SARIMAX(hist_vis_log,
                            order=(1, 1, 1),
                            seasonal_order=(1, 1, 1, 2),
                            enforce_stationarity=False,
                            enforce_invertibility=False)
            model_fit_sarimax = model.fit(disp=False)
            visitor_number_forecast.model_fit_sarimax = model_fit_sarimax
        else:
            model_fit_sarimax = visitor_number_forecast.model_fit_sarimax

        forecast_log = model_fit_sarimax.forecast(steps=periods)
        # 反变换
        forecast_raw = np.expm1(forecast_log)
        # clamp到0以上
        forecast_raw = np.clip(forecast_raw, 0, None)

        adjusted_forecast = []
        for i, vis in enumerate(forecast_raw):
            if optimal_allocation:
                policy_invest = optimal_allocation['Policy Support ($)'] / periods
            else:
                policy_invest = policy_support_invest / periods

            # 再做政策干预 => 仍防止出现负数
            visitor = vis / (1 + (policy_invest / scale) ** 0.5)
            visitor = max(0, visitor)
            adjusted_forecast.append(visitor)

        return pd.Series(adjusted_forecast)

    except Exception as e:
        print(f"Error forecasting visitor numbers (SARIMAX m=2): {e}")
        # 如果报错，返回一个简单的常数序列(非负)
        return pd.Series([max(0, average_visitors)] * periods)

def environment_score_decay(env_score, visitors, threshold=500000, alpha=1e-6):
    if visitors <= threshold:
        return env_score
    exceed = visitors - threshold
    decay = np.exp(alpha * exceed) - 1
    return max(0, env_score - decay)

def social_score_decay(soc_score, visitors, threshold=500000, beta=1e-6):
    if visitors <= threshold:
        return soc_score
    exceed = visitors - threshold
    decay = np.exp(beta * exceed) - 1
    return soc_score - decay

def adjust_revenue_by_env_and_soc(revenue_original, env_score, soc_score,
                                  alpha_env=0.0002, alpha_soc=0.0002):
    environment_quality = max(0, env_score)
    social_quality = max(0, soc_score)
    factor = 1.0 - alpha_env * environment_quality - alpha_soc * social_quality
    factor = max(0.1, factor)  # 保证不会变成负
    return revenue_original * factor

# -------------------- 未来预测（环境-社会互相影响） --------------------
def future_prediction(visitor_forecast, current_env_score, current_soc_score,
                      current_econ_score, years=5,
                      env_invest_per_year=10000,
                      infra_invest_per_year=8000,
                      policy_invest_per_year=7000,
                      env_threshold=500000,
                      soc_threshold=500000,
                      alpha=1e-6,
                      beta=1e-6):
    predictions = []
    env_score = current_env_score
    soc_score = current_soc_score
    econ_score = current_econ_score

    reinvest_rate = 0.2
    extra_fund = 0.0

    for year in range(years):
        year_env_invest = env_invest_per_year + extra_fund * 0.3
        year_infra_invest = infra_invest_per_year + extra_fund * 0.3
        year_policy_invest = policy_invest_per_year + extra_fund * 0.4

        # ★ 修改：环境与社会互相影响
        env_score = environmental_impact_model(env_score, year_env_invest, soc_score)
        soc_score = social_impact_model(soc_score, year_infra_invest, year_policy_invest, env_score)

        if year < len(visitor_forecast):
            vis_forecast = visitor_forecast.iloc[year]
        else:
            vis_forecast = visitor_forecast.iloc[-1]

        env_score = environment_score_decay(env_score, vis_forecast, threshold=env_threshold, alpha=alpha)
        soc_score = social_score_decay(soc_score, vis_forecast, threshold=soc_threshold, beta=beta)

        revenue_original = vis_forecast * average_spending
        revenue_adjusted = adjust_revenue_by_env_and_soc(revenue_original, env_score, soc_score)

        econ_score = economic_score_model(econ_score, revenue_adjusted, env_score, soc_score, alpha=0.5)
        extra_fund = reinvest_rate * revenue_adjusted

        w_dyn = get_dynamic_weights('peak', econ_score, env_score, soc_score)
        stability = stability_initial
        score = sustainable_tourism_score(w_dyn, econ_score, env_score, soc_score, stability)

        predictions.append({
            'Year': 2024 + year,
            'Number of Tourists': vis_forecast,
            'Revenue Original ($)': revenue_original,
            'Revenue Adjusted ($)': revenue_adjusted,
            'Economic Score': econ_score,
            'Environment Score': env_score,
            'Social Score': soc_score,
            'Sustainability Score': score,
            'Extra Fund for Next Year': extra_fund
        })

    return pd.DataFrame(predictions)

if optimal_allocation:
    visitor_forecast = visitor_number_forecast(
        basic_info[tourists_col],
        optimal_allocation['Policy Support ($)']
    )
else:
    visitor_forecast = visitor_number_forecast(basic_info[tourists_col], 0)

future_scores = future_prediction(visitor_forecast,
                                  environmental_score_initial,
                                  social_score_initial,
                                  economic_score_initial,
                                  years=5,
                                  env_invest_per_year=10000,
                                  infra_invest_per_year=8000,
                                  policy_invest_per_year=7000,
                                  env_threshold=500000,
                                  soc_threshold=500000,
                                  alpha=1e-6,
                                  beta=1e-6)

print("\n=== Future 5-Year Sustainability Score Predictions ===")
print(future_scores)

plt.figure(figsize=(10, 6))
sns.lineplot(x='Year', y='Sustainability Score', data=future_scores,
             marker='o', linestyle='-', color='b')
plt.xlabel('Year')
plt.ylabel('Sustainability Score')
plt.title('Future 5-Year Sustainability Score Prediction (with Dynamic Economic Score)')
plt.grid(True)
plt.show()

plot_3d_visitors_revenue_sustainability(future_scores)

def plot_optimal_allocation(allocation):
    labels = ['Environment Investment', 'Infrastructure Investment', 'Policy Support']
    sizes = [allocation['Environment Investment ($)'],
             allocation['Infrastructure Investment ($)'],
             allocation['Policy Support ($)']]
    colors = ['#ff9999', '#66b3ff', '#99ff99']
    explode = (0.1, 0, 0)

    plt.figure(figsize=(8, 8))
    plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
            shadow=True, startangle=140)
    plt.title('Optimal Investment Allocation')
    plt.axis('equal')
    plt.show()

if optimal_allocation is not None:
    plot_optimal_allocation(optimal_allocation)
else:
    print("No optimal allocation found that satisfies the thresholds.")

print("\n模型概要：")
print("1) 环境评分与社会评分在 environmental_impact_model / social_impact_model 中互相影响；")
print("2) 经济评分放大对环境与社会的依赖，可见 synergy_factor=0.001；")
print("3) 优化与预测时，都先更新环境->再更新社会，让三大评分不再独立；")
print("4) 额外收入再投资：每年有部分收入投入下一年的环境/基础设施/政策。")
print("5) 敏感度分析与相关性矩阵应能看出更显著的三者交互。")