# 更规范的导入方式（在文件顶部统一导入）
import pandas as pd
import numpy as np
import datetime
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor  # 移至文件开头

def adaptive_regression1(merged):
    """执行三个独立回归：
    1. H_buy ~   H_buy_lag1  + rt_lag1
    2. H_sell ~  H_sell_lag1 + rt_lag1
    3. H_zero ~  H_zero_lag1 + rt_lag1
    """
    results = {}

    # 遍历三个被解释变量
    for dep_var in ['H_buy', 'H_sell', 'H_zero']:

        # 动态生成公式
        base_formula = f'{dep_var} ~ {dep_var}_lag1 + rt_lag1 + vix_lag1 + HLV_lag1 + turnover + sentiment2 + sentiment3'


        formula = base_formula
        model_type = "简化模型"

        # 模型拟合
        model = smf.ols(formula=formula, data=merged)
        result = model.fit()

        # 存储结果
        results[dep_var] = {
            'model': model,
            'result': result,
            'formula': formula,
            'model_type': model_type
        }

    return results

def adaptive_regression2(merged):
    """执行三个独立回归：
    1. H_buy ~   H_buy_lag1  + turnover
    2. H_sell ~  H_sell_lag1 + turnover + NMV
    3. H_zero ~  H_zero_lag1 + turnover + NMV
    """
    results = {}

    # 遍历三个被解释变量
    for dep_var in ['H_buy', 'H_sell', 'H_zero']:

        # 动态生成公式
        base_formula = f'{dep_var} ~ {dep_var}_lag1 + order_imbalance_lag1 + \
        sentiment2 + sentiment3 + sentiment2 + inter_rt + rt_max_lag1 + rt_min_lag1'


        formula = base_formula
        model_type = "简化模型"

        # 模型拟合
        model = smf.ols(formula=formula, data=merged)
        result = model.fit()
        result = result.get_robustcov_results(cov_type='HAC', maxlags=5, use_correction=True)

        # 存储结果
        results[dep_var] = {
            'model': model,
            'result': result,
            'formula': formula,
            'model_type': model_type,
            'counts': merged.shape[0]
        }

    return results

def get_H_data(year, month, group=10):
    def safe_calculate_H(runs, total_runs):
        p_s = 1 / 3
        x = (runs + 0.5 - total_runs * p_s * (1 - p_s)) / np.sqrt(total_runs)
        sigma_sq = p_s * (1 - p_s) - 3 * (p_s ** 2) * (1 - p_s) ** 2

        if sigma_sq < 1e-10:
            return np.nan * runs
        H = x / np.sqrt(sigma_sq)
        H[np.isinf(H)] = np.nan
        return H

    herbing_path = 'data/herbing/'
    data = pd.read_feather(herbing_path + str(year) + str(month).zfill(2) + '.fea')
    # 保留237根分钟数据
    data = data[data['bar_cnt'] < 238]
    # 每group根合成一个分组
    data.loc[:, 'group'] = data['bar_cnt'] // group + 1
    re_cols = {'a': 'H_buy', 'b': 'H_sell', 'c': 'H_zero', 'd': 'original_total_trades', 'prod_code': 'scr_num'}
    result = data.groupby(['prod_code', 'trade_date', 'group'])[['a', 'b', 'c', 'd']].sum().reset_index().rename(
        columns=re_cols)
    # result.loc[:,'mood'] = result['H_buy']/(result['H_sell']+0.1)
    # 计算herbing因子
    for key in ['H_buy', 'H_sell', 'H_zero']:
        result.loc[:, key] = safe_calculate_H(result[key], result['original_total_trades'])
    # 处理时间
    time_delta = datetime.timedelta(minutes=9 * 60 + 30) + result['group'].apply(
        lambda x: datetime.timedelta(minutes=x * group) if x < 120 / group else datetime.timedelta(
            minutes=x * group + 90))
    result['date_time'] = result['trade_date'] + time_delta

    return result[['scr_num', 'date_time', 'H_buy', 'H_sell', 'H_zero', 'group']]


def res_sout(results, f):
    for dep_var, res in results.items():
        print(f"\n{'=' * 40}", file=f)
        print(f" 模型: {dep_var}", file=f)
        print(f" 类型: {res['model_type']}", file=f)
        print(f" 公式: {res['formula']}", file=f)
        print(f" 样本数: {res['counts']}", file=f)
        try:
            print(f" R²: {res['result'].rsquared:.4f}", file=f)
        except:
            print(f" R²: {res['result'].prsquared:.4f}", file=f)

        # 输出系数表
        print("\n系数估计:", file=f)
        print(res['result'].summary().tables[1], file=f)

        # 多重共线性诊断
        vif = pd.DataFrame()
        vif["Variable"] = res['model'].exog_names
        vif["VIF"] = [variance_inflation_factor(res['model'].exog, i)
                      for i in range(res['model'].exog.shape[1])]
        print("\n方差膨胀因子(VIF):", file=f)
        print(vif, file=f)


def res_sout1(results):
    for dep_var, res in results.items():
        print(f"\n{'=' * 40}")
        print(f" 模型: {dep_var}")
        print(f" 类型: {res['model_type']}")
        print(f" 公式: {res['formula']}")
        print(f" 样本数: {res['counts']}")
        try:
            print(f" R²: {res['result'].rsquared:.4f}")
        except:
            print(f" R²: {res['result'].prsquared:.4f}")

        # 输出系数表
        print("\n系数估计:")
        print(res['result'].summary().tables[1])
        # 多重共线性诊断
        vif = pd.DataFrame()
        vif["Variable"] = res['model'].exog_names
        vif["VIF"] = [variance_inflation_factor(res['model'].exog, i)
                     for i in range(res['model'].exog.shape[1])]
        print("\n方差膨胀因子(VIF):")
        print(vif)



if __name__ == "__main__":

    ###按照股票池收益率回归-5分钟
    factorpath = 'data/factor/'
    # 按照年度进行回归
    output = open('regression_6_all_530.txt', 'w')

    # 获取成分股

    ZZ500 = pd.read_feather('data/index_components/000905_components.fea')
    ZZ500.rename(columns={'trade_date': 'date_time'}, inplace=True)
    components = ZZ500['scr_num'].drop_duplicates().to_list()
    ZZ500.loc[:, 'ZZ500'] = 1
    ZZ500.set_index(['date_time', 'scr_num'], inplace=True)
    ZZ500.sort_index(inplace=True)


    # 读取MV和换手率数据
    mv_data = pd.read_feather(factorpath + 'MV.fea')  # ['scr_num','date_time','turnover','NMV']
    mv_data.set_index(['date_time', 'scr_num'], inplace=True)
    mv_data = mv_data.loc[:, components, :]
    mv_data.sort_index(inplace=True)
    mv_data = mv_data.drop_duplicates()
    # 滞后一天，第一天用后值填充
    mv_data = mv_data.groupby(level=1).shift(1)
    mv_data = mv_data.groupby(level=1).fillna(method='bfill')
    # mv_data.loc[:, 'Q'] = mv_data.groupby(level=0).apply(
    #     lambda x: pd.qcut(x['turnover'], q=4, labels=['Q4', 'Q3', 'Q2', 'Q1']).droplevel(0))  # Q1组换手率最大
    mv_data.loc[:, 'NMV'] = np.log(mv_data['NMV'])

    # 读取sentiment
    se_data = pd.read_feather(factorpath + '/sentiment/sentiment.fea')  # ['scr_num','date_time','sentiment']
    se_data.set_index(['date_time', 'scr_num'], inplace=True)
    se_data = se_data.loc[:, components, :]
    se_data.loc[:,'sentiment2'] = np.log(se_data['FINANCEVALUE']+1)
    se_data.loc[:, 'sentiment3'] = np.log(se_data['SECURITYVALUE'] + 1)
    se_data.sort_index(inplace=True)
    # 滞后一天，第一天用后值填充
    se_data = se_data.groupby(level=1).shift(1)
    se_data = se_data.groupby(level=1).fillna(method='bfill')
    mv_data = pd.merge(mv_data, se_data, how='outer', left_index=True,
                       right_index=True).sort_index()
    print('null data of sentiment counts is: ', mv_data.sentiment1.isnull().sum())

    mv_data = pd.merge(mv_data, ZZ500[['ZZ500']], how='outer', left_index=True,
                       right_index=True).sort_index()
    mv_data.loc[:, 'ZZ500'] = mv_data['ZZ500'].fillna(0)
    mv_data.loc[:, 'sentiment1'] = mv_data['sentiment1'].fillna(0)
    mv_data.loc[:, 'sentiment2'] = mv_data['sentiment2'].fillna(0)
    mv_data.loc[:, 'sentiment3'] = mv_data['sentiment3'].fillna(0)

    res_mean_q = []
    res_median_q = []
    res_min_q = []
    res_max_q = []
    res_std_q = []

    # 按照
    for year in range(2020, 2025):
        # 读取HLV
        HLV = pd.read_feather(factorpath + 'HLV_5m_%s.feather' % year)
        # 读取vix
        vix = pd.read_feather(factorpath + '/VIX/300VIX_2020_2024_5m.fea' )
        HLV = pd.merge(HLV, vix, how='inner', on='date_time')

        HLV.set_index(['date_time', 'scr_num'], inplace=True)
        HLV.rename(columns={'return': 'rt'}, inplace=True)
        HLV = HLV.loc[:, components, :]
        HLV.sort_index(inplace=True)

        # 读取MV和换手率数据
        merge_factor = pd.merge(HLV, mv_data[
            ['turnover', 'NMV', 'sentiment1', 'sentiment2', 'sentiment3', 'ZZ500']], how='outer',
                                left_index=True,
                                right_index=True).sort_index()
        merge_factor = merge_factor.groupby(level=1).fillna(method='ffill')
        merge_factor = merge_factor.reindex(HLV.index)
        merge_data = []

        for month in range(1, 13):
            print('monthly regression: %s-%s' % (year, month))
            if year == 2025 and month == 4:
                break
            tmp_h_data = get_H_data(year, month, group=5)
            str_time = tmp_h_data['date_time'].min()
            end_time = tmp_h_data['date_time'].max()
            tmp_h_data = tmp_h_data.set_index(['date_time', 'scr_num']).sort_index()
            tmp_h_data = tmp_h_data.loc[:, components, :]
            tmp_HLV = merge_factor.loc[str_time:end_time]
            tmp_merge = pd.merge(tmp_h_data, tmp_HLV, left_index=True, right_index=True)
            time_index = tmp_merge.index.get_level_values('date_time').time

            # 生成滞后项
            for factor in ['H_buy', 'H_sell', 'H_zero', 'HLV', 'vix','rt']:  # 'HLV', 'rt', 'H_buy', 'H_sell', 'H_zero'
                for lag in range(1, 2):
                    tmp_merge.loc[:, factor + '_lag%s' % lag] = tmp_merge[factor].groupby(level=1).shift(lag)
                    tmp_merge.loc[time_index == datetime.time(9, 30 + 5 * lag), factor + '_lag%s' % lag] = np.nan
            # for compo in ['ZZ500']:
            #     print('monthly regression without Q in %s: %s-%s' % (compo, year, month), file=output)
            #     tmp_res = adaptive_regression8(tmp_merge[tmp_merge[compo] == 1].dropna())
            #     res_sout(tmp_res, output)
            merge_data.append(tmp_merge)

        merge_data = pd.concat(merge_data)

    #print('yearly regression: %s' % (year))
    for compo in ['ZZ500']:
        reg_data = merge_data[merge_data[compo] == 1]
        reg_data = reg_data.loc[:, ['rt_lag1', 'H_buy', 'H_sell', 'H_zero', 'H_buy_lag1', 'H_sell_lag1',
                                    'H_zero_lag1', 'order_imbalance_lag1','sentiment2', 'sentiment3', 'inter_rt']]
        reg_data.loc[:, 'rt_max_lag1'] = reg_data.groupby(level=0)['rt_lag1'].apply(lambda x: x >= x.quantile(0.95))
        reg_data.loc[:, 'rt_min_lag1'] = reg_data.groupby(level=0)['rt_lag1'].apply(lambda x: x <= x.quantile(0.05))
        reg_data.loc[:, 'rt_max_lag1'] = reg_data.loc[:, 'rt_max_lag1'].replace({True: 1.0, False: 0.0})
        reg_data.loc[:, 'rt_min_lag1'] = reg_data.loc[:, 'rt_min_lag1'].replace({True: 1.0, False: 0.0})
        #print('yearly regression without Q in %s: %s' % (compo, year), file=output)
        # tmp_res = adaptive_regression1(reg_data.dropna())
        # res_sout(tmp_res, output)
        # res_sout1(tmp_res)
        tmp_res = adaptive_regression2(reg_data.dropna())
        res_sout(tmp_res, output)
        res_sout1(tmp_res)

        #疫情1：2020.1.20-2020.2.3
        print('疫情1：2020.1.20-2020.2.3')
        print('疫情1：2020.1.20-2020.2.3', file=output)
        start = datetime.datetime(year=2020, month=1, day=20, hour=0, minute=0, second=1)
        end = datetime.datetime(year=2020, month=2, day=3, hour=23, minute=59, second=59)
        tmp_data = reg_data.loc[start:end]
        tmp_res = adaptive_regression2(tmp_data.dropna())
        res_sout(tmp_res, output)
        res_sout1(tmp_res)

        # 疫情2：2020.1.20-2020.5.14
        print('疫情2：2020.1.20-2020.5.14')
        print('疫情2：2020.1.20-2020.5.14', file=output)
        start = datetime.datetime(year=2020, month=1, day=20, hour=0, minute=0, second=1)
        end = datetime.datetime(year=2020, month=5, day=14, hour=23, minute=59, second=59)
        tmp_data = reg_data.loc[start:end]
        tmp_res = adaptive_regression2(tmp_data.dropna())
        res_sout(tmp_res, output)
        res_sout1(tmp_res)

        # 恒大1：2021.12.6-2021.12.17
        print('恒大1：2021.12.6-2021.12.17')
        print('恒大1：2021.12.6-2021.12.17', file=output)
        start = datetime.datetime(year=2021, month=12, day=6, hour=0, minute=0, second=1)
        end = datetime.datetime(year=2021, month=12, day=17, hour=23, minute=59, second=59)
        tmp_data = reg_data.loc[start:end]
        tmp_res = adaptive_regression2(tmp_data.dropna())
        res_sout(tmp_res, output)
        res_sout1(tmp_res)

        # 恒大2：2020.9.24-2021.12.17
        print('恒大2：2020.9.24-2021.12.17')
        print('恒大2：2020.9.24-2021.12.17', file=output)
        start = datetime.datetime(year=2020, month=9, day=24, hour=0, minute=0, second=1)
        end = datetime.datetime(year=2021, month=12, day=17, hour=23, minute=59, second=59)
        tmp_data = reg_data.loc[start:end]
        tmp_res = adaptive_regression2(tmp_data.dropna())
        res_sout(tmp_res, output)
        res_sout1(tmp_res)


    output.close()