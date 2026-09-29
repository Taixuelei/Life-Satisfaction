import pandas as pd
import os
import warnings

warnings.filterwarnings('ignore')


def merge_pisa_data(folder_path, output_file):
    """
    合并指定文件夹下所有国家的PISA2018问卷数据

    参数:
    folder_path: 包含各国数据子文件夹的路径
    output_file: 输出合并文件的路径
    """

    # 获取所有子文件夹
    subfolders = [f for f in os.listdir(folder_path)
                  if os.path.isdir(os.path.join(folder_path, f))]

    print(f"找到 {len(subfolders)} 个子文件夹")

    if not subfolders:
        print("错误: 没有找到子文件夹")
        return

    # 存储所有数据框
    all_dataframes = []
    processed_countries = []

    # 先收集所有文件的列信息，确定统一的列顺序
    all_columns = set()

    # 第一步：收集所有列名
    for folder in subfolders:
        folder_full_path = os.path.join(folder_path, folder)
        excel_files = [f for f in os.listdir(folder_full_path)
                       if f.endswith('.xlsx') or f.endswith('.xls')]

        if not excel_files:
            print(f"警告: 文件夹 {folder} 中没有Excel文件")
            continue

        for excel_file in excel_files:
            file_path = os.path.join(folder_full_path, excel_file)
            try:
                # 只读取列名，不读取数据
                df_temp = pd.read_excel(file_path, nrows=0)
                all_columns.update(df_temp.columns.tolist())
                print(f"  从 {excel_file} 收集到 {len(df_temp.columns)} 列")
            except Exception as e:
                print(f"读取 {folder}/{excel_file} 的列信息时出错: {e}")

    if not all_columns:
        print("错误: 没有收集到任何列信息")
        return

    # 定义基础列（根据你的需求调整）
    base_columns = ['CNTRYID', 'CNT', 'CNTSCHID', 'CNTSTUID', 'CYC', 'NatCen',
                    'STRATUM', 'SUBNATIO', 'OECD', 'ADMINMODE', 'LANGTEST_QQQ',
                    'LANGTEST_COG', 'LANGTEST_PAQ', 'BOOKID']

    # 确定列顺序：基础列在前，然后按字母顺序排列其他列
    base_cols_in_data = [col for col in base_columns if col in all_columns]
    other_cols = sorted([col for col in all_columns if col not in base_columns])
    column_order = base_cols_in_data + other_cols

    print(f"总列数: {len(column_order)}")

    # 第二步：读取和处理所有文件
    for folder in subfolders:
        folder_full_path = os.path.join(folder_path, folder)
        excel_files = [f for f in os.listdir(folder_full_path)
                       if f.endswith('.xlsx') or f.endswith('.xls')]

        if not excel_files:
            continue

        for excel_file in excel_files:
            print(f"处理: {folder} -> {excel_file}")

            try:
                # 读取整个文件
                file_path = os.path.join(folder_full_path, excel_file)
                df = pd.read_excel(file_path)

                # 显示数据基本信息
                print(f"  原始数据形状: {df.shape}, 列数: {len(df.columns)}")

                # 添加源文件信息
                df['_source_folder'] = folder
                df['_source_file'] = excel_file

                # 确保所有需要的列都存在，缺失的列填充为NaN
                for col in column_order + ['_source_folder', '_source_file']:
                    if col not in df.columns:
                        df[col] = pd.NA

                # 重新排列列顺序
                df = df[column_order + ['_source_folder', '_source_file']]

                all_dataframes.append(df)
                processed_countries.append(folder)

                # 显示处理后信息
                print(f"  处理后形状: {df.shape}, 行数: {len(df)}")

                # 及时释放内存
                del df

            except Exception as e:
                print(f"处理 {folder}/{excel_file} 时出错: {e}")
                import traceback
                traceback.print_exc()

    if not all_dataframes:
        print("错误: 没有成功读取任何数据文件")
        return

    # 第三步：一次性合并所有数据框
    print("正在合并所有数据...")
    try:
        # 使用pd.concat一次性合并
        merged_df = pd.concat(all_dataframes, axis=0, ignore_index=True, sort=False)

        print(f"合并后数据形状: {merged_df.shape}")
        print(f"合并后列数: {len(merged_df.columns)}")

        # 确保输出文件路径有效
        if not output_file.endswith('.xlsx'):
            output_file = output_file + '.xlsx'

        # 确保输出目录存在
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"创建输出目录: {output_dir}")

        # 保存到Excel文件
        print(f"正在保存到 {output_file}...")

        # 使用openpyxl引擎，并设置内存优化选项
        merged_df.to_excel(
            output_file,
            index=False,
            engine='openpyxl'
        )

        # 输出统计信息
        print(f"\n合并完成!")
        print(f"总行数: {len(merged_df)}")
        print(f"总列数: {len(merged_df.columns)}")
        print(f"处理的国家数量: {len(set(processed_countries))}")
        print(f"处理的国家列表: {set(processed_countries)}")

        # 显示列信息
        print("\n前20列:")
        for i, col in enumerate(merged_df.columns[:20], 1):
            print(f"{i:4d}. {col}")

        print("\n后10列:")
        for i, col in enumerate(merged_df.columns[-10:], len(merged_df.columns) - 9):
            print(f"{i:4d}. {col}")

        # 释放内存
        del merged_df
        del all_dataframes

    except Exception as e:
        print(f"合并数据时出错: {e}")
        import traceback
        traceback.print_exc()


# 使用示例
if __name__ == "__main__":
    # 文件夹路径和输出文件路径 - 请根据实际情况修改
    # 注意：输出文件路径需要是一个完整的文件名，不是目录

    # 输入文件夹路径
    input_folder = r"D:\心理\李荆广老师课题\01处理数据\02 按问题筛选出的数据-python处理\SPSS_SCH_QQQ\变量列表1\集体文化主义国家"

    # 输出文件路径 - 确保这是一个完整的文件名
    output_file = r"D:\心理\李荆广老师课题\01处理数据\03 合并后的结果-python处理\SPSS_SCH_QQQ\变量列表1\集体文化主义国家.xlsx"

    print("=" * 60)
    print(f"输入文件夹: {input_folder}")
    print(f"输出文件: {output_file}")
    print("=" * 60)

    # 执行合并
    merge_pisa_data(input_folder, output_file)

    print("\n" + "=" * 60)
    print("处理完成!")
    print("=" * 60)