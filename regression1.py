# 更规范的导入方式（在文件顶部统一导入）
import pandas as pd
import numpy as np
import datetime
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor  # 移至文件开头

def adaptive_regression1(merged):
    """执行三个独立回归：
    1. H_buy ~   H_buy_lag1  + vix_lag1
    2. H_sell ~  H_sell_lag1 + vix_lag1
    3. H_zero ~  H_zero_lag1 + vix_lag1
    """
    results = {}

    # 遍历三个被解释变量
    for dep_var in ['H_buy', 'H_sell', 'H_zero']:

        # 动态生成公式
        base_formula = f'{dep_var} ~ {dep_var}_lag1 + sentiment2'


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

def adaptive_regression3(merged):
    """执行三个独立回归：
    1. H_buy ~   H_buy_lag1  + vix_lag1
    2. H_sell ~  H_sell_lag1 + vix_lag1
    3. H_zero ~  H_zero_lag1 + vix_lag1
    """
    results = {}

    # 遍历三个被解释变量
    for dep_var in ['H_buy', 'H_sell', 'H_zero']:

        # 动态生成公式
        base_formula = f'{dep_var} ~ {dep_var}_lag1 + sentiment3'


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

def adaptive_regression2(merged):
    """执行三个独立回归：
    1. H_buy ~   H_buy_lag1  + sentiment + sentiment_finance + sentiment_security
    2. H_sell ~  H_sell_lag1 + sentiment + sentiment_finance + sentiment_security
    3. H_zero ~  H_zero_lag1 + sentiment + sentiment_finance + sentiment_security
    """
    results = {}

    # 遍历三个被解释变量
    for dep_var in ['H_buy', 'H_sell', 'H_zero']:

        # 动态生成公式
        base_formula = f'{dep_var} ~ {dep_var}_lag1  + inter_rt'


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

def adaptive_regression4(merged):
    """执行三个独立回归：
    1. H_buy ~   H_buy_lag1  + sentiment + sentiment_finance + sentiment_security
    2. H_sell ~  H_sell_lag1 + sentiment + sentiment_finance + sentiment_security
    3. H_zero ~  H_zero_lag1 + sentiment + sentiment_finance + sentiment_security
    """
    results = {}

    # 遍历三个被解释变量
    for dep_var in ['H_buy', 'H_sell', 'H_zero']:

        # 动态生成公式
        base_formula = f'{dep_var} ~ {dep_var}_lag1  + vix_lag1'


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
            'model_type': model_type
        }

    return results

def adaptive_regression5(merged):
    """执行三个独立回归：
    1. H_buy ~   H_buy_lag1  + sentiment + sentiment_finance + sentiment_security
    2. H_sell ~  H_sell_lag1 + sentiment + sentiment_finance + sentiment_security
    3. H_zero ~  H_zero_lag1 + sentiment + sentiment_finance + sentiment_security
    """
    results = {}

    # 遍历三个被解释变量
    for dep_var in ['H_buy', 'H_sell', 'H_zero']:

        # 动态生成公式
        base_formula = f'{dep_var} ~ {dep_var}_lag1  + sentiment2 + sentiment3'


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



# ===== 主执行流程 =====
if __name__ == "__main__":

    ###按照股票池收益率回归-5分钟
    factorpath = 'data/factor/'
    # 按照年度进行回归
    output = open('regression_1_all_705.txt', 'w')

    # 获取成分股

    ZZ500 = pd.read_feather('data/index_components/000905_components.fea')
    ZZ500.rename(columns={'trade_date': 'date_time'}, inplace=True)
    components = ZZ500['scr_num'].drop_duplicates().to_list()
    ZZ500.loc[:, 'ZZ500'] = 1
    ZZ500.set_index(['date_time', 'scr_num'], inplace=True)
    #ZZ500 = ZZ500.drop_duplicates()
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
    #mv_data.loc[:, 'NMV'] = np.log(mv_data['NMV'])

    # 读取日k线计算隔夜收益率
    quote_data = pd.read_feather(factorpath + '/quote.fea')
    quote_data.loc[:, 'date_time'] = pd.to_datetime(quote_data['date_time'])
    quote_data.set_index(['date_time', 'scr_num'], inplace=True)
    quote_data = quote_data.loc[:, components, :]
    quote_data.sort_index(inplace=True)
    quote_data = quote_data.drop_duplicates()
    quote_data.loc[:, 'inter_rt'] = quote_data['open_px']/quote_data['preclose_px']-1
    quote_data.loc[:, 'inter_rt'] = quote_data['inter_rt'].groupby(level=1).shift(1)
    mv_data = pd.merge(mv_data, quote_data[['inter_rt']], how='outer', left_index=True,
                       right_index=True).sort_index()

    # 读取sentiment
    se_data = pd.read_feather(factorpath + '/sentiment/sentiment.fea')  # ['scr_num','date_time','sentiment']
    se_data.set_index(['date_time', 'scr_num'], inplace=True)
    se_data = se_data.drop_duplicates()
    se_data = se_data.loc[:, components, :]
    # se_data.loc[:,'sentiment2'] = np.log(se_data['FINANCEVALUE']+1)
    # se_data.loc[:, 'sentiment3'] = np.log(se_data['SECURITYVALUE'] + 1)

    se_data.sort_index(inplace=True)
    # 滞后一天，第一天用后值填充
    se_data = se_data.groupby(level=1).shift(1)
    #se_data = se_data.groupby(level=1).fillna(method='bfill')
    mv_data = pd.merge(mv_data, se_data, how='outer', left_index=True,
                       right_index=True).sort_index()
    # mv_data.loc[:, 'sentiment2'] = np.log((mv_data['FINANCEVALUE']+1) / mv_data['NMV'])
    # mv_data.loc[:, 'sentiment3'] = np.log((mv_data['SECURITYVALUE']+1) / mv_data['NMV'])
    mv_data.loc[:, 'sentiment2'] = mv_data['FINANCEVALUE'] / mv_data['NMV']
    mv_data.loc[:, 'sentiment3'] = mv_data['SECURITYVALUE'] / mv_data['NMV']
    mv_data.loc[:, 'NMV'] = np.log(mv_data['NMV'])

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

    merge_data = []

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
            ['sentiment2', 'sentiment3', 'inter_rt', 'ZZ500']], how='outer',
                                left_index=True,
                                right_index=True).sort_index()
        merge_factor = merge_factor.groupby(level=1).fillna(method='ffill')
        merge_factor = merge_factor.reindex(HLV.index)

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
                    tmp_merge.loc[time_index == datetime.time(9, 30+5*lag), factor + '_lag%s' % lag] = np.nan
            # for compo in ['ZZ500']:
            #     print('monthly regression without Q in %s: %s-%s' % (compo, year, month), file=output)
            #     tmp_res = adaptive_regression8(tmp_merge[tmp_merge[compo] == 1].dropna())
            #     res_sout(tmp_res, output)
            merge_data.append(tmp_merge)

    merge_data = pd.concat(merge_data)

    merge_data.reset_index().to_feather('r1_data.fea')

    #print('yearly regression: %s' % (year))
    for compo in ['ZZ500']:
        #print('yearly regression without Q in all: %s' % (compo), file=output)
        res_data = merge_data[merge_data[compo] == 1]
        res_data = res_data.loc[:, ['H_buy', 'H_sell', 'H_zero', 'H_buy_lag1', 'H_sell_lag1', 'H_zero_lag1',
                                    'inter_rt', 'sentiment2', 'sentiment3',]]
        tmp_res = adaptive_regression1(res_data.dropna())
        res_sout(tmp_res, output)
        res_sout1(tmp_res)
        tmp_res = adaptive_regression3(res_data.dropna())
        res_sout(tmp_res, output)
        res_sout1(tmp_res)
        tmp_res = adaptive_regression2(res_data.dropna())
        res_sout(tmp_res, output)
        res_sout1(tmp_res)
        # tmp_res = adaptive_regression4(merge_data[merge_data[compo] == 1].dropna())
        # res_sout(tmp_res, output)
        # res_sout1(tmp_res)
        tmp_res = adaptive_regression5(res_data.dropna())
        res_sout(tmp_res, output)
        res_sout1(tmp_res)


    output.close()