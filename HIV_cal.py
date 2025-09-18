import os

import pandas as pd
from tqdm import tqdm
import datetime
import numpy as np
'''
def read_large_csv(file_path):
    # 预计算总行数用于进度条
    try:
        with open(file_path, 'r') as f:
            total_rows = sum(1 for _ in f) - 1  # 减去header行
    except:
        total_rows = None

    # 分块读取配置
    chunks = pd.read_csv(file_path, chunksize=50000)
    df_list = []
    
    # 需要处理的数值列
    numeric_cols = ['最高价', '最低价', '前收盘价', '交易时段序号']
    
    for chunk in tqdm(chunks, 
                     total=total_rows//50000+1 if total_rows else None,
                     desc="处理分块",
                     unit="chunk"):
        # 处理千分位分隔符并转换数值类型
        for col in numeric_cols:
            # 移除逗号并转换为数值
            chunk[col] = (
                chunk[col]
                .astype(str)
                .str.replace(',', '')
                .pipe(pd.to_numeric, errors='coerce')
            )
        
        # 过滤有效数据
        filtered = chunk[
            (chunk['交易时段序号'] <= 230) &
            (chunk['交易时段序号'].notna()) &
            (chunk['最高价'].notna()) &
            (chunk['最低价'].notna()) &
            (chunk['前收盘价'].notna())
        ].copy()
        
        # 类型转换保障
        filtered['交易时段序号'] = filtered['交易时段序号'].astype(int)
        
        # 创建分组键
        filtered['group'] = (filtered['交易时段序号'] - 1) // 10 + 1
        
        df_list.append(filtered)
    
    return pd.concat(df_list, ignore_index=True)
'''
# 主处理流程
if __name__ == "__main__":
    '''
    #df = read_large_csv('20180103-31_processed.csv')
    datapath = 'data/Kline_1m/'
    factorpath = 'data/factor/'
    time_filters = [datetime.time(9, 30), datetime.time(14, 58), datetime.time(14, 59), datetime.time(15, 0)]
    RV = []
    for filename in os.listdir(datapath):
        print(filename)
        year = filename[:4]
        if filename.endswith('.feather'):
            df = pd.read_feather(datapath + filename)
            df.sort_values(['date_time', 'scr_num'])
            # 去掉每天第一根和最后三个数据
            df = df[df.date_time.apply(lambda x: True if x.time() not in time_filters else False)]
            # 每10分钟计算HLV
            df.set_index('date_time', inplace=True)
            result = df.groupby('scr_num').resample('5min', closed='right', label='right').agg({
                                                                                                'preclose_px': 'first',
                                                                                                'close_px': 'last'
                                                                                                }).dropna().reset_index()
            result.loc[:,'r_2'] = np.square(np.log(result['close_px']) - np.log(result['preclose_px']))

            result.loc[:, 'r_4'] = np.power(np.log(result['close_px']) - np.log(result['preclose_px']), 4)

            result.loc[:,'date_time'] = result['date_time'].apply(lambda x: x.date())

            result = result.groupby(['scr_num', 'date_time'])['r_2', 'r_4'].sum().rename(columns={'r_2': 'RV', 'r_4': 'RQ'})

            result.loc[:, 'RQ'] = result['RQ']*240/5/3*10000

            result.loc[:, 'RV'] = result['RQ'] * 100

            RV.append(result)

    RV = pd.concat(RV)

    RV.reset_index().to_feather(factorpath + 'RV.feather')
    '''


    datapath = 'data/Kline_1m/'
    factorpath = 'data/factor/'
    time_filters = []
    #time_filters = [datetime.time(9,30), datetime.time(14,58),datetime.time(14,59), datetime.time(15,0)]
    res = []
    for filename in os.listdir(datapath):
        print(filename)
        year = filename[:4]
        if filename.endswith('.feather'):
            df = pd.read_feather(datapath+filename)
            df.sort_values(['date_time','scr_num'])
            #去掉每天第一根和最后三个数据
            df = df[df.date_time.apply(lambda x: True if x.time() not in time_filters else False)]
            #每10分钟计算HLV
            df.set_index('date_time', inplace=True)
            result = df.groupby('scr_num').resample('1d', closed = 'right', label='right').agg({'high_px':'max',
                                                        'open_px': 'first',
                                                        'low_px': 'min',
                                                        'preclose_px':'first',
                                                        'close_px':'last'
                                                        }).dropna().reset_index()
            # result.loc[:,'HLV'] = (
            #                         (result['high_px'] - result['low_px']) /
            #                         result['preclose_px'].replace(0, pd.NA)  # 防止除以零
            #                 ) * 100
            # result.loc[:, 'Volatility_Park'] = np.sqrt(
            #                                np.log(result['high_px'] / result['low_px'])/4/np.log(2)
            #                        )
            # result.loc[:, 'Volatility_RS'] = np.sqrt(
            #                                np.log(result['high_px']/result['open_px'])*np.log(result['high_px']/result['close_px']) +
            #                                np.log(result['low_px']/result['open_px'])*np.log(result['low_px']/result['close_px'])
            #                        )
            #
            # result.loc[:, 'return'] = np.log(result['close_px']/result['preclose_px'])

            # result = result[['scr_num', 'date_time', 'HLV', 'Volatility_Park', 'Volatility_RS', 'return']]
            # result.to_feather(factorpath+'HLV_5m_%s.feather'%year)
            result = result[['scr_num', 'date_time', 'open_px', 'low_px', 'preclose_px', 'close_px']]
            res.append(result)
    res = pd.concat(res, ignore_index=True)
    res.to_feather(factorpath + 'daily_quote.feather')




    '''
    # 关键修正：添加股票代码和交易所代码到分组键
    result = df.groupby(['股票代码', '交易所代码', '日期', 'group']).agg(
        max_high=('最高价', 'max'),
        min_low=('最低价', 'min'),
        first_prev_close=('前收盘价', 'first')
    ).reset_index()
    
    # 计算HLV时增加异常处理
    result['HLV'] = (
        (result['max_high'] - result['min_low']) / 
        result['first_prev_close'].replace(0, pd.NA)  # 防止除以零
    ) * 100
    
    # 直接使用分组后的数据（不再需要merge）
    final_df = result[['股票代码', '交易所代码', '日期', 'group', 'HLV']].rename(
        columns={'group': '新序号'}
    )

    # 关键排序步骤：按股票代码->日期->时间段排序
    final_df = final_df.sort_values(['股票代码', '日期', '新序号'])
    
    final_df.to_csv('10min_grouped_HLV.csv', index=False)
    print("修正后的结果示例：")
    print(final_df.head(10))
    '''
