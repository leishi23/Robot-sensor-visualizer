import streamlit as st
import json
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
from pathlib import Path
import tempfile
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
import io

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
# Google Drive 相关函数
# ============================================================================

@st.cache_resource
def get_gdrive_service():
    """创建 Google Drive API 服务"""
    try:
        credentials = service_account.Credentials.from_service_account_info(
            st.secrets["gcp_service_account"],
            scopes=['https://www.googleapis.com/auth/drive.readonly']
        )
        service = build('drive', 'v3', credentials=credentials)
        return service
    except Exception as e:
        st.error(f"Failed to authenticate with Google Drive: {e}")
        return None

@st.cache_data(ttl=3600)  # 缓存1小时
def list_json_files_from_gdrive(_service, folder_id):
    """从 Google Drive 文件夹递归获取所有 JSON 文件"""
    if _service is None:
        return []
    
    json_files = []
    
    def list_files_recursive(parent_id, parent_path=""):
        """递归列出文件"""
        try:
            query = f"'{parent_id}' in parents and trashed=false"
            results = _service.files().list(
                q=query,
                fields="files(id, name, mimeType, parents)",
                pageSize=1000
            ).execute()
            
            items = results.get('files', [])
            
            for item in items:
                file_name = item['name']
                file_id = item['id']
                mime_type = item['mimeType']
                
                current_path = f"{parent_path}/{file_name}" if parent_path else file_name
                
                # 如果是文件夹，递归搜索
                if mime_type == 'application/vnd.google-apps.folder':
                    list_files_recursive(file_id, current_path)
                # 如果是 JSON 文件，添加到列表
                elif file_name.endswith('.json'):
                    json_files.append({
                        'id': file_id,
                        'name': file_name,
                        'path': current_path
                    })
        except Exception as e:
            st.warning(f"Error listing files in folder {parent_path}: {e}")
    
    # 开始递归搜索
    list_files_recursive(folder_id)
    
    # 按路径排序
    json_files.sort(key=lambda x: x['path'])
    
    return json_files

@st.cache_data(ttl=3600)
def download_file_from_gdrive(_service, file_id):
    """从 Google Drive 下载文件内容"""
    if _service is None:
        return None
    
    try:
        request = _service.files().get_media(fileId=file_id)
        file_content = io.BytesIO()
        downloader = MediaIoBaseDownload(file_content, request)
        
        done = False
        while not done:
            status, done = downloader.next_chunk()
        
        # 解析 JSON
        file_content.seek(0)
        data = json.loads(file_content.read().decode('utf-8'))
        return data
    except Exception as e:
        st.error(f"Error downloading file: {e}")
        return None

# ============================================================================
# 原有的可视化函数（保持不变）
# ============================================================================

def plot_wrist_pose(data, side, frame_idx=None):
    """绘制手腕位姿（位置和四元数）"""
    poses = np.array(data[f'{side}_wrist_pose'])
    
    if frame_idx is not None:
        # 显示单帧
        pose = poses[frame_idx]
        st.write(f"**Frame {frame_idx}:**")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Position (x, y, z)", f"[{pose[0]:.3f}, {pose[1]:.3f}, {pose[2]:.3f}]")
        with col2:
            st.metric("Quaternion (w, x, y, z)", f"[{pose[3]:.3f}, {pose[4]:.3f}, {pose[5]:.3f}, {pose[6]:.3f}]")
    else:
        # 绘制时间序列
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Position (x, y, z)', 'Orientation (quaternion w, x, y, z)'),
            vertical_spacing=0.15
        )
        
        # 位置
        for i, label in enumerate(['x', 'y', 'z']):
            fig.add_trace(
                go.Scatter(x=list(range(len(poses))), y=poses[:, i], 
                          name=label, mode='lines'),
                row=1, col=1
            )
        
        # 四元数
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
        # 显示单帧
        joint = joints[frame_idx]
        st.write(f"**Frame {frame_idx} - {len(joint)} joints:**")
        cols = st.columns(min(6, len(joint)))
        for i, val in enumerate(joint):
            cols[i % len(cols)].metric(f"J{i}", f"{val:.3f}")
    else:
        # 绘制时间序列
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
        # 显示单帧热图
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
        
        # 显示统计信息
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Min", f"{tactile_frame.min():.3f}")
        col2.metric("Max", f"{tactile_frame.max():.3f}")
        col3.metric("Mean", f"{tactile_frame.mean():.3f}")
        col4.metric("Std", f"{tactile_frame.std():.3f}")
    else:
        # 绘制时间序列热图
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
        
        # 显示整体统计
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
    st.title("🤖 Robot Sensor Data Visualizer (Google Drive)")
    st.markdown("---")
    
    # 检查是否配置了 secrets
    # if "gcp_service_account" not in st.secrets or "gdrive_folder_id" not in st.secrets:
    #     st.error("❌ Google Drive credentials not configured!")
    #     st.info("""
    #     Please configure your secrets in Streamlit Cloud:
    #     1. Go to your app settings
    #     2. Add secrets in TOML format:
    #     ```toml
    #     [gcp_service_account]
    #     type = "service_account"
    #     project_id = "..."
    #     # ... other fields from service_account.json
        
    #     gdrive_folder_id = "YOUR_FOLDER_ID"
    #     ```
    #     """)
    #     return
    
    # 获取 Google Drive 服务
    service = get_gdrive_service()
    if service is None:
        return
    
    folder_id = st.secrets["gdrive_folder_id"]
    
    # 侧边栏
    with st.sidebar:
        st.header("📁 File Selection (Google Drive)")
        
        # 显示文件夹信息
        st.info(f"📂 Folder ID: {folder_id[:20]}...")
        
        # 加载文件列表
        with st.spinner("Loading files from Google Drive..."):
            json_files = list_json_files_from_gdrive(service, folder_id)
        
        if json_files:
            st.success(f"Found {len(json_files)} JSON files")
            
            # 文件选择
            file_idx = st.selectbox(
                "Select File",
                range(len(json_files)),
                format_func=lambda x: json_files[x]['path']
            )
            
            selected_file = json_files[file_idx]
            
            # 显示当前文件信息
            st.info(f"📄 File {file_idx + 1}/{len(json_files)}")
            with st.expander("File Details", expanded=False):
                st.write(f"**Name:** {selected_file['name']}")
                st.write(f"**Path:** {selected_file['path']}")
                st.write(f"**ID:** {selected_file['id']}")
            
            # 导航按钮
            col1, col2 = st.columns(2)
            if col1.button("⬅️ Previous", disabled=(file_idx == 0)):
                st.rerun()
            if col2.button("➡️ Next", disabled=(file_idx == len(json_files) - 1)):
                st.rerun()
            
            # 显示文件夹结构统计
            if len(json_files) > 0:
                st.markdown("---")
                with st.expander("📊 Folder Distribution", expanded=False):
                    # 统计每个子文件夹的文件数量
                    folder_counts = {}
                    for file_info in json_files:
                        path = file_info['path']
                        folder = os.path.dirname(path) if os.path.dirname(path) else "root"
                        folder_counts[folder] = folder_counts.get(folder, 0) + 1
                    
                    # 显示统计
                    st.write(f"**Total folders: {len(folder_counts)}**")
                    for folder, count in sorted(folder_counts.items()):
                        st.text(f"📁 {folder}: {count} files")
        else:
            st.error("No JSON files found in Google Drive folder")
            st.info("💡 Make sure you've uploaded JSON files and the service account has access")
            return
        
        st.markdown("---")
        st.header("⚙️ Visualization Options")
        
        # 选择机械臂侧
        side = st.radio("Select Arm", ["left", "right"], horizontal=True)
        
        # 选择可视化模式
        viz_mode = st.radio(
            "Visualization Mode",
            ["Time Series", "Single Frame"],
            help="Time Series: 显示整个序列\nSingle Frame: 查看单个帧"
        )
        
        frame_idx = None
        if viz_mode == "Single Frame":
            # 显示帧选择提示
            st.info("⏳ Loading file to get frame count...")
    
    # 主内容区域
    try:
        # 下载并加载数据
        with st.spinner(f"Downloading {selected_file['name']} from Google Drive..."):
            data = download_file_from_gdrive(service, selected_file['id'])
        
        if data is None:
            st.error("Failed to load file from Google Drive")
            return
        
        # 如果是 Single Frame 模式，显示帧选择滑块
        if viz_mode == "Single Frame":
            with st.sidebar:
                num_frames = len(data[f'{side}_wrist_pose'])
                frame_idx = st.slider("Frame Index", 0, num_frames - 1, 0)
        
        # 显示文件信息
        with st.expander("📊 Data Summary", expanded=False):
            col1, col2, col3 = st.columns(3)
            col1.metric("File Name", selected_file['name'])
            col2.metric("Number of Frames", len(data[f'{side}_wrist_pose']))
            col3.metric("Data Keys", len(data.keys()))
            
            st.json({k: f"List[{len(v)}]" for k, v in data.items()})
        
        # 创建标签页
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "🎯 Wrist Pose", 
            "🦾 Joint States", 
            "👆 Finger Tactile",
            "🖐️ Palm Tactile",
            "📊 All Sensors"
        ])
        
        with tab1:
            st.subheader(f"{side.capitalize()} Wrist Pose")
            plot_wrist_pose(data, side, frame_idx)
        
        with tab2:
            st.subheader(f"{side.capitalize()} Joint States")
            plot_joint_states(data, side, frame_idx)
        
        with tab3:
            st.subheader(f"{side.capitalize()} Finger Tactile Sensors")
            for finger in ['finger_0', 'finger_1', 'finger_2']:
                with st.expander(f"📍 {finger.replace('_', ' ').title()}", expanded=(frame_idx is not None)):
                    plot_tactile_data(data, side, finger, frame_idx)
        
        with tab4:
            st.subheader(f"{side.capitalize()} Palm Tactile Sensor")
            plot_tactile_data(data, side, 'palm', frame_idx)
        
        with tab5:
            if frame_idx is not None:
                st.subheader(f"{side.capitalize()} Hand - All Sensors Comparison")
                plot_all_tactile_comparison(data, side, frame_idx)
            else:
                st.info("Switch to 'Single Frame' mode to view all sensors comparison")
                
    except Exception as e:
        st.error(f"Error loading or visualizing data: {e}")
        st.exception(e)

if __name__ == "__main__":
    main()
