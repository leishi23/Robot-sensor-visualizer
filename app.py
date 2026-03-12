import streamlit as st
import json
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import time
from pathlib import Path
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
import io
import hashlib

# 页面配置
st.set_page_config(
    page_title="Robot Sensor Data Visualizer",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义样式
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================================
# 密码保护
# ============================================================================

def hash_password(password):
    """对密码进行 SHA256 哈希"""
    return hashlib.sha256(password.encode()).hexdigest()

def get_correct_password_hash():
    """获取正确的密码哈希"""
    if "gcp_service_account" in st.secrets and "app_password_hash" in st.secrets["gcp_service_account"]:
        return st.secrets["gcp_service_account"]["app_password_hash"]
    if "app_password_hash" in st.secrets:
        return st.secrets["app_password_hash"]
    return hash_password("robot2024")

def check_password():
    """
    检查密码，返回 True 表示验证通过。

    关键改动：
    - 用 st.session_state["authenticated"] 作为唯一认证标志
    - 登录成功后立即 st.rerun()，确保密码输入框不再渲染
    - 不使用 on_change 回调，改用显式 Login 按钮，避免 rerun 时状态混乱
    """

    # 已经验证通过，直接返回 True，不渲染任何登录 UI
    if st.session_state.get("authenticated", False):
        with st.sidebar:
            if st.button("🔓 Logout", use_container_width=True):
                st.session_state.clear()
                st.rerun()
        return True

    # ---------- 未验证，显示登录界面 ----------
    correct_password_hash = get_correct_password_hash()

    st.title("🔒 Robot Sensor Data Visualizer")
    st.markdown("---")

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("### 请输入密码访问")

        password = st.text_input(
            "Password",
            type="password",
            key="login_password",
            placeholder="输入密码..."
        )

        login_clicked = st.button("Login", use_container_width=True, type="primary")

        if login_clicked:
            if password:
                entered_hash = hash_password(password.strip())
                if entered_hash == correct_password_hash:
                    st.session_state["authenticated"] = True
                    # 关键：立即 rerun，下次循环走 authenticated=True 分支
                    # 密码输入框不会再被渲染
                    st.rerun()
                else:
                    st.error("❌ 密码错误，请重试")
            else:
                st.warning("请输入密码")

        st.info("💡 如果忘记密码，请联系管理员")

    return False

# ============================================================================
# Google Drive 相关函数
# ============================================================================

MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds


@st.cache_resource(ttl=1800)
def get_gdrive_service():
    """创建 Google Drive API 服务（每 30 分钟自动刷新）"""
    try:
        credentials = service_account.Credentials.from_service_account_info(
            st.secrets["gcp_service_account"],
            scopes=['https://www.googleapis.com/auth/drive.readonly']
        )
        service = build('drive', 'v3', credentials=credentials)
        return service
    except Exception as e:
        st.error(f"Failed to authenticate with Drive: {e}")
        return None


def _execute_with_retry(request_fn, description="API call"):
    """
    通用重试封装：对 Google API 调用进行最多 MAX_RETRIES 次重试，
    以处理 BrokenPipeError、ConnectionResetError 等瞬时网络异常。
    
    Parameters:
        request_fn: 一个可调用对象，执行后返回 API 结果
        description: 用于日志/警告的描述文字
    
    Returns:
        API 调用的返回结果
    
    Raises:
        最后一次重试仍失败时抛出原始异常
    """
    last_exception = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return request_fn()
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError) as e:
            last_exception = e
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY * attempt)  # 指数退避
            else:
                raise
        except Exception as e:
            # 检查嵌套的 BrokenPipe（有时被包在 google-api 的 HttpError 里）
            if "Broken pipe" in str(e) or "broken pipe" in str(e).lower():
                last_exception = e
                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_DELAY * attempt)
                else:
                    raise
            else:
                raise  # 非网络类异常直接抛出


@st.cache_data(ttl=3600)
def list_json_files_from_gdrive(_service, folder_id):
    """从 Google Drive 文件夹递归获取所有 JSON 文件（带重试）"""
    if _service is None:
        return []

    json_files = []

    def list_files_recursive(parent_id, parent_path=""):
        try:
            query = f"'{parent_id}' in parents and trashed=false"

            def do_list():
                return _service.files().list(
                    q=query,
                    fields="files(id, name, mimeType, parents)",
                    pageSize=1000
                ).execute()

            results = _execute_with_retry(do_list, f"list files in {parent_path or 'root'}")
            items = results.get('files', [])

            for item in items:
                file_name = item['name']
                file_id = item['id']
                mime_type = item['mimeType']

                current_path = f"{parent_path}/{file_name}" if parent_path else file_name

                if mime_type == 'application/vnd.google-apps.folder':
                    list_files_recursive(file_id, current_path)
                elif file_name.endswith('.json'):
                    json_files.append({
                        'id': file_id,
                        'name': file_name,
                        'path': current_path
                    })
        except Exception as e:
            st.warning(f"Error listing files in folder {parent_path}: {e}")

    list_files_recursive(folder_id)
    json_files.sort(key=lambda x: x['path'])

    return json_files


@st.cache_data(ttl=3600)
def build_folder_structure(json_files):
    """构建文件夹结构"""
    root = {'__subfolders__': {}, '__files__': []}

    for file_info in json_files:
        parts = file_info['path'].split('/')
        current = root

        for part in parts[:-1]:
            if part not in current['__subfolders__']:
                current['__subfolders__'][part] = {'__subfolders__': {}, '__files__': []}
            current = current['__subfolders__'][part]

        current['__files__'].append(file_info)

    return root


@st.cache_data(ttl=3600)
def download_file_from_gdrive(_service, file_id):
    """从 Google Drive 下载文件内容（带重试）"""
    if _service is None:
        return None

    try:
        def do_download():
            request = _service.files().get_media(fileId=file_id)
            file_content = io.BytesIO()
            downloader = MediaIoBaseDownload(file_content, request)

            done = False
            while not done:
                status, done = downloader.next_chunk()

            file_content.seek(0)
            return json.loads(file_content.read().decode('utf-8'))

        return _execute_with_retry(do_download, f"download file {file_id}")
    except Exception as e:
        st.error(f"Error downloading file: {e}")
        return None

# ============================================================================
# 可视化函数
# ============================================================================

def plot_wrist_pose(data, side, frame_idx=None):
    """绘制手腕位姿（位置和四元数）"""
    poses = np.array(data[f'{side}_wrist_pose'])

    if frame_idx is not None:
        pose = poses[frame_idx]
        st.write(f"**Frame {frame_idx}:**")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Position (x, y, z)", f"[{pose[0]:.3f}, {pose[1]:.3f}, {pose[2]:.3f}]")
        with col2:
            st.metric("Quaternion (w, x, y, z)", f"[{pose[3]:.3f}, {pose[4]:.3f}, {pose[5]:.3f}, {pose[6]:.3f}]")
    else:
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Position (x, y, z)', 'Orientation (quaternion w, x, y, z)'),
            vertical_spacing=0.15
        )

        for i, label in enumerate(['x', 'y', 'z']):
            fig.add_trace(
                go.Scatter(x=list(range(len(poses))), y=poses[:, i],
                          name=label, mode='lines'),
                row=1, col=1
            )

        for i, label in enumerate(['w', 'x', 'y', 'z']):
            fig.add_trace(
                go.Scatter(x=list(range(len(poses))), y=poses[:, i+3],
                          name=label, mode='lines', showlegend=True),
                row=2, col=1
            )

        fig.update_xaxes(title_text="Frame", row=2, col=1)
        fig.update_yaxes(title_text="Position (m)", row=1, col=1)
        fig.update_yaxes(title_text="Quaternion", row=2, col=1)
        fig.update_layout(height=600, title_text=f"{side.capitalize()} Wrist Pose")

        st.plotly_chart(fig, use_container_width=True)

def plot_joint_states(data, side, frame_idx=None):
    """绘制关节状态"""
    joints = np.array(data[f'{side}_joint_states'])

    if frame_idx is not None:
        joint = joints[frame_idx]
        st.write(f"**Frame {frame_idx} - {len(joint)} joints:**")
        cols = st.columns(min(6, len(joint)))
        for i, val in enumerate(joint):
            cols[i % len(cols)].metric(f"J{i}", f"{val:.3f}")
    else:
        fig = go.Figure()

        for i in range(joints.shape[1]):
            fig.add_trace(
                go.Scatter(x=list(range(len(joints))), y=joints[:, i],
                          name=f'Joint {i}', mode='lines')
            )

        fig.update_layout(
            title=f"{side.capitalize()} Joint States",
            xaxis_title="Frame",
            yaxis_title="Joint Angle (rad)",
            height=500
        )

        st.plotly_chart(fig, use_container_width=True)

def plot_tactile_data(data, side, sensor_type, frame_idx=None):
    """绘制触觉传感器数据"""
    sensor_key = f'{side}_{sensor_type}_tactile'
    tactile = np.array(data[sensor_key])

    if len(tactile) == 0 or len(tactile[0]) == 0:
        st.warning(f"No data available for {sensor_key}")
        return

    if frame_idx is not None:
        tactile_frame = np.array(tactile[frame_idx])

        fig = go.Figure(data=go.Heatmap(
            z=[tactile_frame],
            colorscale='Viridis',
            colorbar=dict(title="Force")
        ))

        fig.update_layout(
            title=f"{side.capitalize()} {sensor_type.capitalize()} Tactile - Frame {frame_idx}",
            xaxis_title="Sensor Index",
            yaxis_title="",
            height=200,
            yaxis=dict(showticklabels=False)
        )

        st.plotly_chart(fig, use_container_width=True)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Min", f"{tactile_frame.min():.3f}")
        col2.metric("Max", f"{tactile_frame.max():.3f}")
        col3.metric("Mean", f"{tactile_frame.mean():.3f}")
        col4.metric("Std", f"{tactile_frame.std():.3f}")
    else:
        tactile_array = np.array(tactile)

        fig = go.Figure(data=go.Heatmap(
            z=tactile_array.T,
            colorscale='Viridis',
            colorbar=dict(title="Force")
        ))

        fig.update_layout(
            title=f"{side.capitalize()} {sensor_type.capitalize()} Tactile Sensor",
            xaxis_title="Frame",
            yaxis_title="Sensor Index",
            height=500
        )

        st.plotly_chart(fig, use_container_width=True)

        col1, col2, col3 = st.columns(3)
        col1.metric("Num Frames", tactile_array.shape[0])
        col2.metric("Num Sensors", tactile_array.shape[1])
        col3.metric("Total Data Points", tactile_array.size)

def plot_all_tactile_comparison(data, side, frame_idx):
    """对比显示所有触觉传感器"""
    sensors = ['finger_0', 'finger_1', 'finger_2', 'palm']

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[f'{s.capitalize()}' for s in sensors],
        vertical_spacing=0.15,
        horizontal_spacing=0.1
    )

    positions = [(1, 1), (1, 2), (2, 1), (2, 2)]

    for idx, sensor in enumerate(sensors):
        sensor_key = f'{side}_{sensor}_tactile'
        tactile = np.array(data[sensor_key])

        if len(tactile) > 0 and len(tactile[0]) > 0:
            tactile_frame = np.array(tactile[frame_idx])

            row, col = positions[idx]
            fig.add_trace(
                go.Heatmap(
                    z=[tactile_frame],
                    colorscale='Viridis',
                    showscale=(idx == 3)
                ),
                row=row, col=col
            )

    fig.update_layout(
        title_text=f"{side.capitalize()} Hand - All Tactile Sensors (Frame {frame_idx})",
        height=500
    )

    st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# 主应用
# ============================================================================

def main():
    # 密码验证
    if not check_password():
        st.stop()

    st.title("🤖 Robot Sensor Data Visualizer")
    st.markdown("---")

    # 检查 secrets
    try:
        if "gcp_service_account" not in st.secrets:
            st.error("❌ Drive credentials not configured!")
            return

        folder_id = None
        if "gcp_service_account" in st.secrets and "gdrive_folder_id" in st.secrets["gcp_service_account"]:
            folder_id = st.secrets["gcp_service_account"]["gdrive_folder_id"]
        elif "gdrive_folder_id" in st.secrets:
            folder_id = st.secrets["gdrive_folder_id"]

        if folder_id is None:
            st.error("❌ gdrive_folder_id not configured!")
            return

    except Exception as e:
        st.error(f"❌ Error reading secrets: {e}")
        return

    service = get_gdrive_service()
    if service is None:
        return

    # 加载所有文件（带连接刷新按钮）
    with st.spinner("Loading files..."):
        all_files = list_json_files_from_gdrive(service, folder_id)

    if not all_files:
        st.error("No JSON files found in Google Drive folder.")
        st.markdown("**Possible causes:**")
        st.markdown(
            "- Network issue (Broken pipe / connection reset) — try refreshing below\n"
            "- The `gdrive_folder_id` is incorrect\n"
            "- The service account has no access to the folder"
        )
        if st.button("🔄 Refresh Connection & Retry", type="primary"):
            st.cache_resource.clear()
            st.cache_data.clear()
            st.rerun()
        return

    structure = build_folder_structure(all_files)

    # 初始化 session state
    if 'current_path' not in st.session_state:
        st.session_state.current_path = []
    if 'selected_file' not in st.session_state:
        st.session_state.selected_file = None

    # 侧边栏
    with st.sidebar:
        st.header("📁 File Browser")
        st.success(f"Total: {len(all_files)} files")

        # 刷新连接按钮
        if st.button("🔄 Refresh", use_container_width=True, help="Clear cache and reconnect to  Drive"):
            st.cache_resource.clear()
            st.cache_data.clear()
            st.rerun()

        # 面包屑导航
        breadcrumb = " / ".join(['Root'] + st.session_state.current_path)
        st.markdown(f"**📂 {breadcrumb}**")

        # 返回按钮
        if st.session_state.current_path:
            if st.button("⬆️ Back", use_container_width=True):
                st.session_state.current_path.pop()
                st.session_state.selected_file = None
                st.rerun()

        st.markdown("---")

        # 获取当前文件夹内容
        current = structure
        for folder_name in st.session_state.current_path:
            if folder_name in current['__subfolders__']:
                current = current['__subfolders__'][folder_name]
            else:
                st.error("Invalid path")
                st.session_state.current_path = []
                st.rerun()
                return

        # 显示子文件夹
        subfolders = sorted(current['__subfolders__'].keys())
        if subfolders:
            st.subheader(f"📂 Folders ({len(subfolders)})")
            for folder in subfolders:
                def count_files(node):
                    count = len(node.get('__files__', []))
                    for sub in node.get('__subfolders__', {}).values():
                        count += count_files(sub)
                    return count

                file_count = count_files(current['__subfolders__'][folder])

                if st.button(f"📁 {folder} ({file_count})", key=f"fold_{folder}", use_container_width=True):
                    st.session_state.current_path.append(folder)
                    st.session_state.selected_file = None
                    st.rerun()

        # 显示文件
        files = current.get('__files__', [])
        if files:
            st.markdown("---")
            st.subheader(f"📄 Files ({len(files)})")

            for idx, file_info in enumerate(sorted(files, key=lambda x: x['name'])):
                is_selected = (st.session_state.selected_file and
                             st.session_state.selected_file['id'] == file_info['id'])
                button_type = "primary" if is_selected else "secondary"
                icon = "✓ " if is_selected else ""

                if st.button(f"{icon}{file_info['name']}",
                           key=f"file_{file_info['id']}",
                           type=button_type,
                           use_container_width=True):
                    st.session_state.selected_file = file_info
                    st.rerun()

            # 文件导航
            if st.session_state.selected_file:
                st.markdown("---")
                current_idx = next((i for i, f in enumerate(files)
                                  if f['id'] == st.session_state.selected_file['id']), None)
                if current_idx is not None:
                    col1, col2 = st.columns(2)
                    if col1.button("⬅️", disabled=(current_idx == 0), use_container_width=True):
                        st.session_state.selected_file = files[current_idx - 1]
                        st.rerun()
                    if col2.button("➡️", disabled=(current_idx == len(files) - 1), use_container_width=True):
                        st.session_state.selected_file = files[current_idx + 1]
                        st.rerun()
                    st.caption(f"File {current_idx + 1} / {len(files)}")

        if not files and not subfolders:
            st.info("Empty folder")
            return

        if not st.session_state.selected_file:
            st.info("👆 Select a file")
            return

        st.markdown("---")
        st.header("⚙️ Options")

        side = st.radio("Arm", ["left", "right"], horizontal=True)
        viz_mode = st.radio("Mode", ["Time Series", "Single Frame"])

        frame_idx = None
        if viz_mode == "Single Frame":
            st.info("Loading...")

    # 主内容
    if not st.session_state.selected_file:
        st.info("Please select a file from the sidebar")
        return

    try:
        with st.spinner(f"Loading {st.session_state.selected_file['name']}..."):
            data = download_file_from_gdrive(service, st.session_state.selected_file['id'])

        if data is None:
            st.error("Failed to load file")
            if st.button("🔄 Retry", type="primary"):
                st.cache_data.clear()
                st.rerun()
            return

        if viz_mode == "Single Frame":
            with st.sidebar:
                num_frames = len(data[f'{side}_wrist_pose'])
                frame_idx = st.slider("Frame", 0, num_frames - 1, 0)

        with st.expander("📊 Data Summary"):
            col1, col2, col3 = st.columns(3)
            col1.metric("File", st.session_state.selected_file['name'])
            col2.metric("Frames", len(data[f'{side}_wrist_pose']))
            col3.metric("Keys", len(data.keys()))

        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "🎯 Wrist", "🦾 Joints", "👆 Fingers", "🖐️ Palm", "📊 All"
        ])

        with tab1:
            plot_wrist_pose(data, side, frame_idx)

        with tab2:
            plot_joint_states(data, side, frame_idx)

        with tab3:
            for finger in ['finger_0', 'finger_1', 'finger_2']:
                with st.expander(f"{finger.replace('_', ' ').title()}", expanded=(frame_idx is not None)):
                    plot_tactile_data(data, side, finger, frame_idx)

        with tab4:
            plot_tactile_data(data, side, 'palm', frame_idx)

        with tab5:
            if frame_idx is not None:
                plot_all_tactile_comparison(data, side, frame_idx)
            else:
                st.info("Switch to Single Frame mode")

    except Exception as e:
        st.error(f"Error: {e}")
        st.exception(e)
        if st.button("🔄 Clear Cache & Retry", type="primary"):
            st.cache_resource.clear()
            st.cache_data.clear()
            st.rerun()

if __name__ == "__main__":
    main()
