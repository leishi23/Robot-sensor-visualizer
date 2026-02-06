#!/usr/bin/env python3
"""
密码哈希生成器
用于生成 Streamlit 应用的密码哈希值
"""

import hashlib
import sys

def hash_password(password):
    """对密码进行 SHA256 哈希"""
    return hashlib.sha256(password.encode()).hexdigest()

def main():
    print("=" * 60)
    print("🔐 Streamlit 密码哈希生成器")
    print("=" * 60)
    print()
    
    if len(sys.argv) > 1:
        # 从命令行参数获取密码
        password = sys.argv[1]
    else:
        # 交互式输入
        password = input("请输入要设置的密码: ")
    
    if not password:
        print("❌ 密码不能为空")
        return
    
    # 生成哈希
    password_hash = hash_password(password)
    
    print()
    print("✅ 密码哈希已生成！")
    print()
    print("请将以下内容添加到 Streamlit Cloud Secrets 中：")
    print("-" * 60)
    print(f'app_password_hash = "{password_hash}"')
    print("-" * 60)
    print()
    print("📋 完整的 Secrets 配置示例：")
    print()
    print("""[gcp_service_account]
type = "service_account"
project_id = "..."
# ... 其他字段 ...

gdrive_folder_id = "YOUR_FOLDER_ID"

# 密码保护
app_password_hash = "{}\"
""".format(password_hash))
    print()
    print("💡 提示：")
    print("   - 密码哈希是单向加密的，无法反推出原密码")
    print("   - 即使有人看到哈希值，也无法知道密码")
    print("   - 建议使用强密码（至少8位，包含字母和数字）")
    print()

if __name__ == "__main__":
    main()
