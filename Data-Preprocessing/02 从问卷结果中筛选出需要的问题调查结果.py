import pyreadstat
import pandas as pd
import os
import shutil
import time
import warnings

warnings.filterwarnings('ignore')


def parse_variable_file(txt_file_path):
    """
    解析变量列表txt文件。
    适配新格式：每行仅包含一个变量名（无任何分隔符）
    """
    variable_list = []
    print(f"正在解析变量文件: {txt_file_path}")

    try:
        with open(txt_file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:  # 跳过空行
                    continue

                # ========== 核心修改：移除分隔符处理，直接读取单行变量 ==========
                # 新格式下每行仅一个变量，无需分割，直接加入列表
                vars_in_line = [line]  # 保持列表格式，兼容后续extend逻辑

                # ========== 原有逻辑保留 ==========
                variable_list.extend(vars_in_line)

                # 打印第一行的解析详情用于调试
                if line_num == 1:
                    print(f"  首行原始内容: {line}")
                    print(f"  首行解析出变量数: {len(vars_in_line)}")
                    print(f"  首行解析示例（前5个）: {vars_in_line[:5]}")

        # 去重并保持原始出现的顺序
        unique_vars = list(dict.fromkeys(variable_list))

        print(f"  总计找到 {len(variable_list)} 个变量名（去重后 {len(unique_vars)} 个）。")
        print(f"  去重后变量示例（前10个）: {unique_vars[:10]}")

        # 重要调试：检查列表中是否包含明确的变量
        check_vars = ['ST016Q01NA', 'AGE', 'ST001Q01TA', 'ST160Q02IA']
        found_vars = []
        not_found_vars = []
        for v in check_vars:
            if v in unique_vars:
                found_vars.append(v)
            else:
                not_found_vars.append(v)

        if found_vars:
            print(f"  ✓ 确认列表包含变量: {found_vars}")
        if not_found_vars:
            print(f"  ⚠ 警告: 列表未找到预期变量: {not_found_vars}")

        return unique_vars

    except Exception as e:
        print(f"✗ 解析变量文件失败: {e}")
        raise


def filter_dataframe_by_variables(df, meta, variables_to_keep, first_n_columns=14):
    """
    根据变量列表筛选DataFrame，并保留相应的元数据。

    参数:
        df: 原始数据的DataFrame
        meta: 原始数据的元数据对象
        variables_to_keep: 要保留的变量名列表（来自txt文件）
        first_n_columns: 必须保留的前N列（默认为14）

    返回:
        filtered_df: 筛选后的DataFrame
        filtered_meta_dict: 筛选后的元数据字典
    """
    # 1. 确定最终要保留的所有列
    all_columns = list(df.columns)

    # 前N列（确保存在）
    first_n_cols = all_columns[:min(first_n_columns, len(all_columns))]

    # 从变量列表中筛选出在数据中实际存在的列
    existing_vars_to_keep = [v for v in variables_to_keep if v in all_columns]

    # 合并列列表，并去重，保持原始顺序（前N列优先）
    columns_to_keep = list(dict.fromkeys(first_n_cols + existing_vars_to_keep))

    # 找出实际被过滤掉的列
    columns_removed = [col for col in all_columns if col not in columns_to_keep]

    print(f"    原始变量数: {len(all_columns)}")
    print(f"    保留变量数: {len(columns_to_keep)} (前{first_n_columns}列 + {len(existing_vars_to_keep)}个指定变量)")
    print(f"    过滤变量数: {len(columns_removed)}")

    # 显示找到的变量数量信息
    print(f"    变量列表匹配情况: {len(existing_vars_to_keep)}/{len(variables_to_keep)} 个变量在数据中找到")

    if columns_removed and len(columns_removed) < 10:
        print(f"    过滤掉的变量: {columns_removed}")

    # 2. 筛选DataFrame
    filtered_df = df[columns_to_keep].copy()

    # 3. 筛选元数据
    filtered_meta = {}

    # 定义需要筛选的元数据属性
    meta_attributes = [
        'column_labels',  # 变量标签
        'variable_value_labels',  # 值标签
        'variable_measure',  # 测量尺度
        'variable_display_width',  # 显示宽度
        'missing_ranges'  # 缺失值定义 (关键！)
    ]

    for attr in meta_attributes:
        if hasattr(meta, attr):
            original_dict = getattr(meta, attr)
            if isinstance(original_dict, dict):
                # 只保留 columns_to_keep 中存在的变量的元数据
                filtered_dict = {k: v for k, v in original_dict.items() if k in columns_to_keep}
                filtered_meta[attr] = filtered_dict

    # 保留文件标签等其他属性
    if hasattr(meta, 'file_label'):
        filtered_meta['file_label'] = getattr(meta, 'file_label', '') + " [变量筛选后]"

    return filtered_df, filtered_meta


def save_filtered_data(filtered_df, meta_dict, base_output_path, country_name=""):
    """
    将筛选后的数据保存为三种格式。
    """
    saved_files = []

    # 1. 保存为SAV格式 (SPSS) - 保留元数据
    sav_path = f"{base_output_path}.sav"
    print(f"      正在保存SAV文件...")
    try:
        # 准备写入参数
        sav_kwargs = {
            'file_label': meta_dict.get('file_label', f"筛选后数据 - {country_name}"),
            'column_labels': meta_dict.get('column_labels', {}),
            'variable_value_labels': meta_dict.get('variable_value_labels', {}),
            'variable_measure': meta_dict.get('variable_measure', {}),
            'variable_display_width': meta_dict.get('variable_display_width', {}),
        }

        # 关键：写入缺失值定义（如果存在）
        if 'missing_ranges' in meta_dict and meta_dict['missing_ranges']:
            sav_kwargs['missing_ranges'] = meta_dict['missing_ranges']
            print(f"        包含缺失值定义的变量: {len(meta_dict['missing_ranges'])} 个")

        pyreadstat.write_sav(filtered_df, sav_path, **sav_kwargs)
        sav_size = os.path.getsize(sav_path) / 1024 / 1024
        print(f"        ✓ SAV文件保存成功: {sav_size:.1f} MB")
        saved_files.append(sav_path)

    except Exception as e:
        print(f"        ✗ 保存SAV文件失败: {e}")
        # 尝试简化保存作为后备
        try:
            print("        尝试简化保存（不含复杂元数据）...")
            pyreadstat.write_sav(filtered_df, sav_path)
            print("        ✓ 简化SAV文件保存成功")
            saved_files.append(sav_path)
        except Exception as e2:
            print(f"        ✗ 简化保存也失败: {e2}")

    # 2. 保存为CSV格式
    csv_path = f"{base_output_path}.csv"
    print(f"      正在保存CSV文件...")
    try:
        filtered_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        csv_size = os.path.getsize(csv_path) / 1024 / 1024
        print(f"        ✓ CSV文件保存成功: {csv_size:.1f} MB")
        saved_files.append(csv_path)
    except Exception as e:
        print(f"        ✗ 保存CSV文件失败: {e}")

    # 3. 保存为Excel格式 - 优化内存使用
    excel_path = f"{base_output_path}.xlsx"
    print(f"      正在保存Excel文件...")
    try:
        # 检查数据大小，大文件使用分块写入
        if len(filtered_df) > 50000 or len(filtered_df.columns) > 100:
            print(f"        数据量较大({len(filtered_df):,}行×{len(filtered_df.columns)}列)，使用优化方式写入Excel...")
            with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
                # 写入数据
                filtered_df.to_excel(writer, sheet_name='Data', index=False)

                # 可选：添加一个元数据说明的工作表
                try:
                    # 创建一个简单的元数据说明工作表
                    meta_info = pd.DataFrame({
                        '信息类型': ['生成时间', '数据维度', '变量数量', '数据来源'],
                        '详细信息': [
                            time.strftime("%Y-%m-%d %H:%M:%S"),
                            f"{len(filtered_df)}行 × {len(filtered_df.columns)}列",
                            len(filtered_df.columns),
                            f"PISA 2018 - {country_name}"
                        ]
                    })
                    meta_info.to_excel(writer, sheet_name='Metadata', index=False)
                except:
                    pass  # 如果添加元数据失败，继续保存主数据
        else:
            # 小文件直接保存
            filtered_df.to_excel(excel_path, index=False)

        excel_size = os.path.getsize(excel_path) / 1024 / 1024
        print(f"        ✓ Excel文件保存成功: {excel_size:.1f} MB")
        saved_files.append(excel_path)

    except MemoryError:
        print(f"        ⚠ 内存不足，尝试分块保存Excel...")
        try:
            with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
                chunk_size = 10000
                for start_idx in range(0, len(filtered_df), chunk_size):
                    end_idx = min(start_idx + chunk_size, len(filtered_df))
                    sheet_num = start_idx // chunk_size + 1
                    chunk = filtered_df.iloc[start_idx:end_idx]
                    chunk.to_excel(writer, sheet_name=f'Part_{sheet_num}', index=False)
                    print(f"          已写入 {end_idx:,}/{len(filtered_df):,} 行")

            excel_size = os.path.getsize(excel_path) / 1024 / 1024
            print(f"        ✓ Excel文件分块保存成功: {excel_size:.1f} MB")
            saved_files.append(excel_path)
        except Exception as e2:
            print(f"        ✗ 分块保存Excel也失败: {e2}")
    except Exception as e:
        print(f"        ✗ 保存Excel文件失败: {e}")
        print(f"        建议: 可以只使用CSV格式或使用分块处理")

    return saved_files


def process_directory(input_root, output_root, variables_to_keep, file_extensions=['.sav']):
    """
    遍历输入目录，处理所有.sav文件，并在输出目录保持相同结构。
    """
    total_files_processed = 0
    total_files_saved = 0

    print(f"开始遍历目录: {input_root}")
    print(f"输出目录: {output_root}")

    # 统计处理进度
    all_sav_files = []
    for root, dirs, files in os.walk(input_root):
        for file in files:
            if file.lower().endswith('.sav'):
                all_sav_files.append(os.path.join(root, file))

    print(f"找到 {len(all_sav_files)} 个.sav文件需要处理")

    # 遍历输入目录
    for root, dirs, files in os.walk(input_root):
        # 计算相对于输入根目录的相对路径
        rel_path = os.path.relpath(root, input_root)
        if rel_path == '.':
            rel_path = ''

        # 创建对应的输出目录
        output_dir = os.path.join(output_root, rel_path)
        os.makedirs(output_dir, exist_ok=True)

        # 处理当前目录下的.sav文件
        sav_files = [f for f in files if f.lower().endswith('.sav')]

        for sav_file in sav_files:
            input_sav_path = os.path.join(root, sav_file)

            # 获取基本文件名（不含扩展名）
            base_name = os.path.splitext(sav_file)[0]

            # 构建输出文件的基础路径（不含扩展名）
            base_output_path = os.path.join(output_dir, base_name)

            print(f"\n{'=' * 60}")
            print(
                f"处理文件 [{total_files_processed + 1}/{len(all_sav_files)}]: {os.path.join(rel_path, sav_file) if rel_path else sav_file}")
            print(f"完整路径: {input_sav_path}")

            try:
                # 读取原始数据和元数据
                df, meta = pyreadstat.read_sav(input_sav_path)
                print(f"  ✓ 读取成功: {len(df):,}行 × {len(df.columns)}列")

                # 显示前20个列名，方便调试
                if total_files_processed == 0:
                    print(f"  数据列名示例（前20个）: {list(df.columns)[:20]}")

                # 按变量筛选数据
                filtered_df, filtered_meta = filter_dataframe_by_variables(
                    df, meta, variables_to_keep, first_n_columns=14
                )

                # 确定国家/地区名称（用于文件标签）
                country_name = os.path.basename(root) if root != input_root else "合并数据"

                # 保存筛选后的数据
                saved = save_filtered_data(
                    filtered_df, filtered_meta, base_output_path, country_name
                )

                total_files_processed += 1
                total_files_saved += len(saved)

                print(f"  ✓ 本文件处理完成，保存了 {len(saved)} 个输出文件。")

            except Exception as e:
                print(f"  ✗ 处理文件失败: {e}")
                import traceback
                traceback.print_exc()
                print(f"  跳过此文件，继续处理下一个...")

    return total_files_processed, total_files_saved


def main():
    """主函数"""
    print("=" * 80)
    print("PISA 数据 - 按变量筛选工具 (第二阶段处理)")
    print("目标: 在已按国家筛选的基础上，进一步筛选变量，保留前14列及指定变量")
    print("=" * 80)

    # ================ 配置区域 (请根据您的实际情况修改) ================

    # 1. 变量列表文件路径 (每行一个变量名)
    VARIABLE_LIST_PATH = r"D:\心理\李荆广老师课题\01处理数据\02 按问题筛选出的数据-python处理\SPSS_SCH_QQQ\变量列表1\变量列表1.txt"

    # 2. 输入目录 (包含按国家筛选结果的一级子文件夹)
    INPUT_ROOT_DIR = r"D:\心理\李荆广老师课题\01处理数据\01 按国家筛选出的数据-python处理\SPSS_SCH_QQQ\个体主义国家"

    # 3. 输出目录 (将创建与输入目录相同的结构)
    OUTPUT_ROOT_DIR = r"D:\心理\李荆广老师课题\01处理数据\02 按问题筛选出的数据-python处理\SPSS_SCH_QQQ\变量列表1\个体文化主义国家"

    # 可选: 如果只想处理特定扩展名的文件
    FILE_EXTENSIONS_TO_PROCESS = ['.sav']  # 目前只处理.sav，但会生成.csv和.xlsx

    # =================================================================

    start_time = time.time()

    # 步骤1: 解析变量列表文件
    print("\n步骤1: 解析变量列表文件")
    print("-" * 40)
    try:
        variables_to_keep = parse_variable_file(VARIABLE_LIST_PATH)
        if not variables_to_keep:
            print("错误: 变量列表为空，请检查文件内容。")
            return
    except Exception as e:
        print(f"程序终止: 无法解析变量列表文件。")
        return

    # 步骤2: 确保输出目录存在
    os.makedirs(OUTPUT_ROOT_DIR, exist_ok=True)

    # 步骤3: 遍历并处理所有目录
    print(f"\n步骤2: 开始处理目录结构")
    print("-" * 40)
    print(f"输入目录: {INPUT_ROOT_DIR}")
    print(f"输出目录: {OUTPUT_ROOT_DIR}")
    print(f"要保留的变量数: {len(variables_to_keep)}")
    print(f"同时自动保留每个文件的前14列。")

    total_processed, total_saved = process_directory(
        INPUT_ROOT_DIR,
        OUTPUT_ROOT_DIR,
        variables_to_keep,
        FILE_EXTENSIONS_TO_PROCESS
    )

    # 步骤4: 最终报告
    total_time = time.time() - start_time

    print(f"\n{'=' * 80}")
    print("处 理 完 成 报 告")
    print(f"{'=' * 80}")
    print(f"处理总时间: {total_time:.1f} 秒")
    print(f"成功处理的.sav文件数: {total_processed}")
    print(f"生成的输出文件总数: {total_saved} (每个输入文件对应最多3个输出文件)")

    if total_processed > 0:
        # 显示一些统计信息
        print(f"\n关键验证指标:")
        print(f"1. 每个输出文件应包含: 前14列 + 变量列表中实际存在的变量")
        print(f"2. 总变量数应显著大于14")
        print(f"3. .sav文件应保留原始元数据（标签、缺失值定义等）")

        print(f"\n输出目录结构已保持与输入目录一致:")
        print(f"  {OUTPUT_ROOT_DIR}/")
        print(f"  ├── [国家1]/")
        print(f"  │   ├── PISA2018_[国家1].sav")
        print(f"  │   ├── PISA2018_[国家1].csv")
        print(f"  │   └── PISA2018_[国家1].xlsx")
        print(f"  ├── [国家2]/")
        print(f"  │   └── ...")
        print(f"  └── ...")

    print(f"\n重 要 提 醒:")
    print(f"1. 请用SPSS打开任意输出的.sav文件，验证变量数量是否符合预期")
    print(f"2. 检查'变量视图'中的标签、缺失值定义等是否完整")
    print(f"3. 如发现问题，请检查变量列表文件格式和内容")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()