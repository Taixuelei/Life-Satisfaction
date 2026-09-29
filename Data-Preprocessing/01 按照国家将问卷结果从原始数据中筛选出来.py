import pyreadstat
import pandas as pd
import os
import re
import time


def parse_country_file(txt_file_path):
    """
    解析国家列表txt文件，提取代码和对应的中文名
    """
    country_dict = {}
    codes = []

    print(f"解析国家列表文件: {txt_file_path}")

    try:
        with open(txt_file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        for line_num, line in enumerate(lines, 1):
            line = line.strip()
            if not line:
                continue

            # 使用正则表达式匹配格式：代码 | 英文名 | 中文名
            match = re.match(r'(\d+)\s*\|\s*(.*?)\s*\|\s*(.+)', line)

            if match:
                code = int(match.group(1))
                chinese_name = match.group(3).strip()

                # 清洗文件名：替换特殊字符
                safe_name = re.sub(r'[\\/*?:"<>|]', '_', chinese_name)
                safe_name = safe_name.replace(' ', '_')  # 空格替换为下划线

                country_dict[code] = {
                    'original_name': chinese_name,
                    'safe_name': safe_name,
                    'english_name': match.group(2).strip()
                }
                codes.append(code)
                print(f"  第{line_num}行: 代码 {code} -> '{chinese_name}'")
            else:
                print(f"  警告: 第{line_num}行格式无法解析: {line}")

    except FileNotFoundError:
        print(f"错误: 找不到文件 {txt_file_path}")
        return {}, []
    except Exception as e:
        print(f"解析文件时出错: {e}")
        return {}, []

    print(f"解析完成，共找到 {len(codes)} 个国家/地区")
    return country_dict, codes


def save_to_excel_simple(df, filepath, country_name, english_name, code):
    """
    简化的Excel保存函数，避免复杂的元数据处理
    """
    try:
        print(f"     正在保存Excel文件...")
        start_time = time.time()

        # 使用pandas的to_excel直接保存，性能更好
        df.to_excel(filepath, index=False, sheet_name='PISA2018数据')

        # 获取文件大小
        file_size = os.path.getsize(filepath) / 1024 / 1024
        save_time = time.time() - start_time

        return True, file_size, save_time
    except Exception as e:
        print(f"     保存Excel文件出错: {e}")

        # 如果失败，尝试使用csv格式作为后备
        try:
            csv_path = filepath.replace('.xlsx', '_fallback.csv')
            df.to_csv(csv_path, index=False, encoding='utf-8-sig')
            print(f"     已保存为CSV后备文件: {csv_path}")
        except Exception as e2:
            print(f"     保存后备文件也失败: {e2}")

        return False, 0, 0


def filter_and_save_pisa_data(input_file_path, output_dir, country_dict, country_codes):
    """
    根据国家代码筛选数据并保存为三种格式
    """

    print("\n" + "=" * 60)
    print("开始处理PISA数据...")
    print("=" * 60)

    # 1. 读取原始数据
    print(f"\n1. 读取原始数据文件: {input_file_path}")
    start_time = time.time()

    try:
        # 使用metadataonly=False确保获取完整数据
        df, meta = pyreadstat.read_sav(input_file_path)
        read_time = time.time() - start_time
        print(f"   ✓ 读取成功!")
        print(f"   - 数据维度: {df.shape[0]:,}行 × {df.shape[1]}列")
        print(f"   - 读取耗时: {read_time:.2f}秒")

        # 检查CNTRYID列是否存在
        if 'CNTRYID' not in df.columns:
            print("   ✗ 错误: 数据中未找到'CNTRYID'列")

            # 尝试查找类似的列名
            similar_cols = [col for col in df.columns if 'country' in col.lower() or 'cntry' in col.lower()]
            if similar_cols:
                print(f"   可能相关的列: {similar_cols}")
                # 尝试使用第一个匹配的列
                cntry_col = similar_cols[0]
                print(f"   尝试使用列: {cntry_col}")
                df = df.rename(columns={cntry_col: 'CNTRYID'})
            else:
                return [], []

    except Exception as e:
        print(f"   ✗ 读取文件失败: {e}")
        return []

    # 2. 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 3. 筛选所有目标国家的数据
    print(f"\n2. 筛选目标国家数据...")
    df_filtered_all = df[df['CNTRYID'].isin(country_codes)].copy()
    total_filtered = len(df_filtered_all)

    print(f"   ✓ 总筛选结果: {total_filtered:,}行")

    if total_filtered == 0:
        print("   ✗ 警告: 没有筛选到任何数据!")
        unique_codes = df['CNTRYID'].unique()[:20]
        print(f"   - 数据中存在的国家代码(前20个): {unique_codes}")
        return []

    # 4. 按国家分别保存三种格式
    print(f"\n3. 按国家分别保存三种格式文件...")

    saved_files = []
    country_stats = []

    # 先处理数据量较少的国家，避免内存问题
    country_data_counts = []
    for code in country_codes:
        if code in country_dict:
            df_country = df_filtered_all[df_filtered_all['CNTRYID'] == code]
            country_data_counts.append((code, len(df_country)))

    # 按数据量从小到大排序
    country_data_counts.sort(key=lambda x: x[1])

    for code, count in country_data_counts:
        if code not in country_dict:
            continue

        country_info = country_dict[code]
        country_name = country_info['original_name']
        safe_name = country_info['safe_name']
        english_name = country_info['english_name']

        df_country = df_filtered_all[df_filtered_all['CNTRYID'] == code].copy()

        if len(df_country) == 0:
            print(f"\n   ⚠  代码 {code} ({country_name}) 没有数据")
            continue

        # 为每个国家创建子目录
        country_dir = os.path.join(output_dir, safe_name)
        os.makedirs(country_dir, exist_ok=True)

        base_filename = f"PISA2018_{safe_name}"

        print(f"\n   ▸ 处理: {country_name} ({english_name}, 代码: {code})")
        print(f"     样本量: {len(df_country):,}行")

        # 保存统计信息
        country_stats.append({
            '国家': country_name,
            '代码': code,
            '样本量': len(df_country),
            '英文名': english_name
        })

        # 保存为SAV格式（SPSS格式）
        sav_path = os.path.join(country_dir, f"{base_filename}.sav")
        try:
            sav_start = time.time()
            pyreadstat.write_sav(
                df_country,
                sav_path,
                file_label=f"PISA 2018 - {country_name} ({english_name})",
                column_labels=meta.column_labels if hasattr(meta, 'column_labels') and isinstance(meta.column_labels,
                                                                                                  dict) else {},
                variable_value_labels=meta.variable_value_labels if hasattr(meta,
                                                                            'variable_value_labels') and isinstance(
                    meta.variable_value_labels, dict) else {},
                note=f"国家: {country_name}, 代码: {code}, 生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}"
            )
            sav_time = time.time() - sav_start
            sav_size = os.path.getsize(sav_path) / 1024 / 1024
            print(f"     ✓ SAV文件: {os.path.basename(sav_path)} ({sav_size:.2f} MB, {sav_time:.1f}秒)")
            saved_files.append(sav_path)
        except Exception as e:
            print(f"     ✗ 保存SAV文件失败: {e}")

        # 保存为CSV格式（通用文本格式）
        csv_path = os.path.join(country_dir, f"{base_filename}.csv")
        try:
            csv_start = time.time()
            df_country.to_csv(csv_path, index=False, encoding='utf-8-sig')
            csv_time = time.time() - csv_start
            csv_size = os.path.getsize(csv_path) / 1024 / 1024
            print(f"     ✓ CSV文件: {os.path.basename(csv_path)} ({csv_size:.2f} MB, {csv_time:.1f}秒)")
            saved_files.append(csv_path)
        except Exception as e:
            print(f"     ✗ 保存CSV文件失败: {e}")

        # 保存为Excel格式（XLSX格式）- 使用简化版本
        excel_path = os.path.join(country_dir, f"{base_filename}.xlsx")
        success, excel_size, excel_time = save_to_excel_simple(
            df_country, excel_path, country_name, english_name, code
        )

        if success:
            print(f"     ✓ Excel文件: {os.path.basename(excel_path)} ({excel_size:.2f} MB, {excel_time:.1f}秒)")
            saved_files.append(excel_path)

    # 5. 保存合并文件（所有选中国家）
    print(f"\n4. 保存合并文件（所有国家）...")

    if len(df_filtered_all) > 0:
        merged_dir = os.path.join(output_dir, "所有国家合并")
        os.makedirs(merged_dir, exist_ok=True)

        # 保存合并的SAV文件
        merged_sav = os.path.join(merged_dir, "PISA2018_所有选中国家.sav")
        try:
            pyreadstat.write_sav(
                df_filtered_all,
                merged_sav,
                file_label="PISA 2018 - 所有选中国家合并",
                column_labels=meta.column_labels if hasattr(meta, 'column_labels') and isinstance(meta.column_labels,
                                                                                                  dict) else {},
                variable_value_labels=meta.variable_value_labels if hasattr(meta,
                                                                            'variable_value_labels') and isinstance(
                    meta.variable_value_labels, dict) else {}
            )
            sav_size = os.path.getsize(merged_sav) / 1024 / 1024
            print(f"   ✓ 合并SAV文件: {os.path.basename(merged_sav)} ({sav_size:.2f} MB)")
            saved_files.append(merged_sav)
        except Exception as e:
            print(f"   ✗ 保存合并SAV文件失败: {e}")

        # 保存合并的CSV文件
        merged_csv = os.path.join(merged_dir, "PISA2018_所有选中国家.csv")
        try:
            df_filtered_all.to_csv(merged_csv, index=False, encoding='utf-8-sig')
            csv_size = os.path.getsize(merged_csv) / 1024 / 1024
            print(f"   ✓ 合并CSV文件: {os.path.basename(merged_csv)} ({csv_size:.2f} MB)")
            saved_files.append(merged_csv)
        except Exception as e:
            print(f"   ✗ 保存合并CSV文件失败: {e}")

        # 保存合并的Excel文件
        merged_excel = os.path.join(merged_dir, "PISA2018_所有选中国家.xlsx")
        success, excel_size, excel_time = save_to_excel_simple(
            df_filtered_all, merged_excel, "所有选中国家", "Selected Countries", 0
        )

        if success:
            print(f"   ✓ 合并Excel文件: {os.path.basename(merged_excel)} ({excel_size:.2f} MB)")
            saved_files.append(merged_excel)

        # 保存国家统计信息
        stats_df = pd.DataFrame(country_stats)
        stats_excel = os.path.join(merged_dir, "PISA2018_国家样本统计.xlsx")
        stats_df.to_excel(stats_excel, index=False)
        print(f"   ✓ 国家统计文件: {os.path.basename(stats_excel)}")
        saved_files.append(stats_excel)

    # 6. 生成处理报告
    print(f"\n" + "=" * 60)
    print("处理完成报告:")
    print("=" * 60)

    print(f"成功处理的国家/地区 ({len(country_stats)}个):")
    for stat in country_stats:
        print(f"  • {stat['国家']}(代码{stat['代码']}): {stat['样本量']:,}行")

    print(f"\n生成文件总数: {len(saved_files)}个")
    print(f"输出目录: {os.path.abspath(output_dir)}")

    # 显示总耗时
    total_time = time.time() - start_time
    print(f"\n总处理时间: {total_time:.2f}秒")

    return saved_files, country_stats


def main():
    """主函数"""

    # ================ 配置区域 ================
    # 请根据你的实际情况修改这些路径

    # 1. 国家列表txt文件路径
    COUNTRY_TXT_FILE = r"D:\Python\Python\李荆广老师课题\data\01 按国家筛选出的数据-python处理\国家列表.txt"

    # 2. PISA原始数据文件路径
    PISA_INPUT_FILE = r"D:\Python\Python\李荆广老师课题\data\原始数据\数据\SPSS_STU_FLT\FLT\CY07_MSU_FLT_TTM.SAV"

    # 3. 输出目录
    OUTPUT_DIRECTORY = r"D:\Python\Python\李荆广老师课题\data\01 按国家筛选出的数据-python处理\SPSS_STU_FLT\CY07_MSU_FLT_TTM"

    # 4. 是否只处理部分国家（设为None则处理全部）
    SELECTED_CODES = None  # 处理全部

    # ================ 执行处理 ================

    print("=" * 70)
    print("PISA 2018 数据按国家批量筛选工具 - 三种格式保存")
    print("=" * 70)
    print("将生成: SAV(SPSS格式), CSV(文本格式), Excel(表格格式)")
    print("=" * 70)

    # 步骤1: 解析国家列表文件
    country_dict, all_codes = parse_country_file(COUNTRY_TXT_FILE)

    if not country_dict:
        print("错误: 无法解析国家列表文件，程序退出。")
        return

    # 步骤2: 确定要处理的国家代码
    if SELECTED_CODES is None:
        codes_to_process = all_codes
        print(f"\n将处理全部 {len(codes_to_process)} 个国家/地区")
    else:
        codes_to_process = [code for code in SELECTED_CODES if code in country_dict]
        print(f"\n将处理选中的 {len(codes_to_process)} 个国家/地区")

    # 步骤3: 执行数据筛选和保存
    saved_files, country_stats = filter_and_save_pisa_data(
        input_file_path=PISA_INPUT_FILE,
        output_dir=OUTPUT_DIRECTORY,
        country_dict=country_dict,
        country_codes=codes_to_process
    )

    # 步骤4: 显示完成信息
    if saved_files:
        print("\n" + "=" * 70)
        print("✅ 批量处理完成!")
        print(f"总共生成 {len(saved_files)} 个文件")

        # 统计各格式文件数量
        format_count = {'sav': 0, 'csv': 0, 'xlsx': 0}
        for file_path in saved_files:
            if file_path.endswith('.sav'):
                format_count['sav'] += 1
            elif file_path.endswith('.csv'):
                format_count['csv'] += 1
            elif file_path.endswith('.xlsx'):
                format_count['xlsx'] += 1

        print(f"\n按格式统计:")
        print(f"  • SAV格式 (SPSS): {format_count['sav']} 个文件")
        print(f"  • CSV格式 (文本): {format_count['csv']} 个文件")
        print(f"  • Excel格式: {format_count['xlsx']} 个文件")

        print(f"\n文件保存位置: {os.path.abspath(OUTPUT_DIRECTORY)}")

        print("\n📊 各格式用途说明:")
        print("1. SAV格式 (.sav): 用于SPSS统计分析，保留完整变量标签和值标签")
        print("2. CSV格式 (.csv): 通用文本格式，可用于Python/R/Excel等软件")
        print("3. Excel格式 (.xlsx): 便于数据查看和简单分析")

        # 显示总样本量
        total_samples = sum(stat['样本量'] for stat in country_stats)
        print(f"\n📈 数据统计:")
        print(f"  总样本量: {total_samples:,}行")
        print(f"  平均每个国家: {total_samples / len(country_stats):,.0f}行")

        # 显示样本量最大的国家
        if country_stats:
            max_country = max(country_stats, key=lambda x: x['样本量'])
            print(f"  样本量最大: {max_country['国家']} - {max_country['样本量']:,}行")
    else:
        print("\n❌ 处理完成，但未保存任何文件。")


if __name__ == "__main__":
    # 确保必要的库已安装
    try:
        import pyreadstat
        import pandas as pd

        print("✅ 所有依赖库已安装")
    except ImportError as e:
        print(f"❌ 缺少依赖库: {e}")
        print("请运行: pip install pyreadstat pandas openpyxl")
        exit(1)

    main()