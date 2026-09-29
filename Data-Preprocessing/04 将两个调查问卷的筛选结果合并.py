import pandas as pd

# ===================== 请修改以下配置项 =====================
# a文件路径（单个CNTSCHID对应一行）
FILE_A_PATH = r"D:\心理\李荆广老师课题\00原始数据\05 国家层面变量处理\国家变量-原始数据\国家变量-集体主义国家.xlsx"  # 替换为你的a文件实际路径，比如 r"D:\data\a.xlsx"
# b文件路径（单个CNTSCHID对应多行）
FILE_B_PATH = r"D:\心理\李荆广老师课题\01处理数据\04 将两个调查问卷的筛选结果合并\集体文化主义国家.xlsx"  # 替换为你的b文件实际路径
# 输出文件路径（扩充后的b文件，建议不要直接覆盖原b文件）
OUTPUT_FILE_PATH = r"D:\心理\李荆广老师课题\01处理数据\05 加入国家变量\集体文化主义国家.xlsx"
# 匹配的关键字段（固定为CNTSCHID），加入国家变量则为CNTRYID
MATCH_COLUMN = "CNTRYID"


# ==========================================================

def merge_excel_by_id():
    try:
        # 1. 读取a、b两个Excel文件
        print(f"正在读取文件A: {FILE_A_PATH}")
        df_a = pd.read_excel(FILE_A_PATH, engine="openpyxl")
        print(f"文件A读取完成：{len(df_a)} 行 × {len(df_a.columns)} 列")

        print(f"\n正在读取文件B: {FILE_B_PATH}")
        df_b = pd.read_excel(FILE_B_PATH, engine="openpyxl")
        print(f"文件B读取完成：{len(df_b)} 行 × {len(df_b.columns)} 列")

        # 2. 校验关键字段是否存在
        if MATCH_COLUMN not in df_a.columns:
            raise ValueError(f"文件A中未找到 {MATCH_COLUMN} 列，请检查列名是否正确")
        if MATCH_COLUMN not in df_b.columns:
            raise ValueError(f"文件B中未找到 {MATCH_COLUMN} 列，请检查列名是否正确")

        # 3. 关键：按CNTSCHID左合并（保留b的所有行，匹配a的列）
        # how='left' 确保b的所有行都保留，on指定匹配列
        # 合并后a的列会追加到b的列后面
        df_merged = pd.merge(df_b, df_a, on=MATCH_COLUMN, how="left")

        # 4. 保存合并后的文件
        print(f"\n正在保存合并后的文件: {OUTPUT_FILE_PATH}")
        df_merged.to_excel(OUTPUT_FILE_PATH, index=False, engine="openpyxl")
        print(f"保存完成！合并后文件：{len(df_merged)} 行 × {len(df_merged.columns)} 列")

        # 5. 输出匹配统计信息
        # 统计b中有多少个CNTSCHID在a中能匹配到
        matched_count = df_merged[MATCH_COLUMN].isin(df_a[MATCH_COLUMN]).sum()
        total_count = len(df_merged)
        print(f"\n匹配统计：")
        print(f"- 文件B总行数：{total_count}")
        print(f"- 成功匹配到文件A数据的行数：{matched_count}")
        print(f"- 未匹配到的行数：{total_count - matched_count}")

    except FileNotFoundError as e:
        print(f"错误：找不到文件 - {e}")
    except ValueError as e:
        print(f"错误：{e}")
    except Exception as e:
        print(f"未知错误：{e}")


if __name__ == "__main__":
    print("===== 开始执行Excel数据合并 =====")
    merge_excel_by_id()
    print("\n===== 执行完成 =====")