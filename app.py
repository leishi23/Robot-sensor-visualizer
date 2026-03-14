#!/usr/bin/env python3
"""
Robot Sensor Data Visualizer (Unified)
=======================================
从 Google Drive 在线可视化机器人传感器数据。
自动根据文件类型（.json / .bag）切换可视化管线。

.json  → 原有 JSON 格式数据（wrist_pose, joint_states, tactile）
.bag   → ROS1 bag（observation, tactile, camera, human pose, quality）

依赖:
  pip install streamlit plotly numpy google-api-python-client google-auth
  pip install rosbags Pillow

运行:
  streamlit run app.py
"""

import streamlit as st
import json
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import io
import time
import tempfile
import hashlib
from pathlib import Path
from collections import defaultdict

from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

# ============================================================================
# 页面配置
# ============================================================================

st.set_page_config(
    page_title="Robot Sensor Data Visualizer",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main { padding: 0rem 1rem; }
    .stTabs [data-baseweb="tab-list"] { gap: 1rem; }
    </style>
""",
    unsafe_allow_html=True,
)

# ============================================================================
# 密码保护
# ============================================================================


def _hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()


def _get_correct_password_hash() -> str:
    if (
        "gcp_service_account" in st.secrets
        and "app_password_hash" in st.secrets["gcp_service_account"]
    ):
        return st.secrets["gcp_service_account"]["app_password_hash"]
    if "app_password_hash" in st.secrets:
        return st.secrets["app_password_hash"]
    return _hash_password("robot2024")


def check_password() -> bool:
    if st.session_state.get("authenticated", False):
        with st.sidebar:
            if st.button("🔓 Logout", use_container_width=True):
                st.session_state.clear()
                st.rerun()
        return True

    correct_hash = _get_correct_password_hash()
    st.title("🔒 Robot Sensor Data Visualizer")
    st.markdown("---")
    _, col2, _ = st.columns([1, 2, 1])
    with col2:
        st.markdown("### 请输入密码访问")
        password = st.text_input(
            "Password", type="password", key="login_password", placeholder="输入密码..."
        )
        if st.button("Login", use_container_width=True, type="primary"):
            if password:
                if _hash_password(password.strip()) == correct_hash:
                    st.session_state["authenticated"] = True
                    st.rerun()
                else:
                    st.error("❌ 密码错误，请重试")
            else:
                st.warning("请输入密码")
        st.info("💡 如果忘记密码，请联系管理员")
    return False


# ============================================================================
# Google Drive 通用函数
# ============================================================================

MAX_RETRIES = 3
RETRY_DELAY = 2


@st.cache_resource(ttl=1800)
def get_gdrive_service():
    try:
        credentials = service_account.Credentials.from_service_account_info(
            st.secrets["gcp_service_account"],
            scopes=["https://www.googleapis.com/auth/drive.readonly"],
        )
        return build("drive", "v3", credentials=credentials)
    except Exception as e:
        st.error(f"Failed to authenticate with Drive: {e}")
        return None


def _execute_with_retry(request_fn, description="API call"):
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return request_fn()
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError) as e:
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY * attempt)
            else:
                raise
        except Exception as e:
            if "broken pipe" in str(e).lower():
                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_DELAY * attempt)
                else:
                    raise
            else:
                raise


SUPPORTED_EXTENSIONS = (".json", ".bag")


@st.cache_data(ttl=3600)
def list_data_files_from_gdrive(_service, folder_id):
    """递归列出 Google Drive 文件夹中所有 .json 和 .bag 文件"""
    if _service is None:
        return []

    data_files = []

    def _recurse(parent_id, parent_path=""):
        try:
            query = f"'{parent_id}' in parents and trashed=false"

            def do_list():
                return (
                    _service.files()
                    .list(q=query, fields="files(id, name, mimeType, size)", pageSize=1000)
                    .execute()
                )

            results = _execute_with_retry(do_list, f"list {parent_path or 'root'}")
            for item in results.get("files", []):
                name = item["name"]
                fid = item["id"]
                current_path = f"{parent_path}/{name}" if parent_path else name

                if item["mimeType"] == "application/vnd.google-apps.folder":
                    _recurse(fid, current_path)
                elif any(name.lower().endswith(ext) for ext in SUPPORTED_EXTENSIONS):
                    data_files.append(
                        {
                            "id": fid,
                            "name": name,
                            "path": current_path,
                            "size": int(item.get("size", 0)),
                            "type": "bag" if name.lower().endswith(".bag") else "json",
                        }
                    )
        except Exception as e:
            st.warning(f"Error listing {parent_path}: {e}")

    _recurse(folder_id)
    data_files.sort(key=lambda x: x["path"])
    return data_files


@st.cache_data(ttl=3600)
def build_folder_structure(file_list):
    root = {"__subfolders__": {}, "__files__": []}
    for fi in file_list:
        parts = fi["path"].split("/")
        node = root
        for part in parts[:-1]:
            if part not in node["__subfolders__"]:
                node["__subfolders__"][part] = {"__subfolders__": {}, "__files__": []}
            node = node["__subfolders__"][part]
        node["__files__"].append(fi)
    return root


# ============================================================================
# 文件下载
# ============================================================================


@st.cache_data(ttl=3600)
def download_json_from_gdrive(_service, file_id):
    """下载 JSON 文件并解析"""
    if _service is None:
        return None
    try:

        def do_download():
            req = _service.files().get_media(fileId=file_id)
            buf = io.BytesIO()
            dl = MediaIoBaseDownload(buf, req)
            done = False
            while not done:
                _, done = dl.next_chunk()
            buf.seek(0)
            return json.loads(buf.read().decode("utf-8"))

        return _execute_with_retry(do_download, f"download json {file_id}")
    except Exception as e:
        st.error(f"Error downloading JSON: {e}")
        return None


def _get_temp_dir():
    """获取/创建临时目录用于缓存 bag 文件"""
    tmp = os.path.join(tempfile.gettempdir(), "rosbag_cache")
    os.makedirs(tmp, exist_ok=True)
    return tmp


def download_bag_to_temp(_service, file_id, file_name):
    """
    下载 .bag 文件到临时目录，返回本地路径。
    使用 file_id 作为缓存 key，避免重复下载。
    """
    if _service is None:
        return None

    tmp_dir = _get_temp_dir()
    local_path = os.path.join(tmp_dir, f"{file_id}_{file_name}")

    # 已缓存
    if os.path.isfile(local_path):
        return local_path

    try:

        def do_download():
            req = _service.files().get_media(fileId=file_id)
            buf = io.BytesIO()
            dl = MediaIoBaseDownload(buf, req)
            done = False
            while not done:
                status, done = dl.next_chunk()
            buf.seek(0)
            with open(local_path, "wb") as f:
                f.write(buf.read())
            return local_path

        return _execute_with_retry(do_download, f"download bag {file_name}")
    except Exception as e:
        st.error(f"Error downloading bag: {e}")
        if os.path.isfile(local_path):
            os.remove(local_path)
        return None


# ============================================================================
# ==================  JSON 可视化管线  =====================================
# ============================================================================


def json_plot_wrist_pose(data, side, frame_idx=None):
    poses = np.array(data[f"{side}_wrist_pose"])
    if frame_idx is not None:
        pose = poses[frame_idx]
        col1, col2 = st.columns(2)
        with col1:
            st.metric(
                "Position (x, y, z)",
                f"[{pose[0]:.3f}, {pose[1]:.3f}, {pose[2]:.3f}]",
            )
        with col2:
            st.metric(
                "Quaternion (w, x, y, z)",
                f"[{pose[3]:.3f}, {pose[4]:.3f}, {pose[5]:.3f}, {pose[6]:.3f}]",
            )
    else:
        fig = make_subplots(
            rows=2,
            cols=1,
            subplot_titles=("Position (x, y, z)", "Orientation (quaternion w, x, y, z)"),
            vertical_spacing=0.15,
        )
        for i, label in enumerate(["x", "y", "z"]):
            fig.add_trace(
                go.Scatter(x=list(range(len(poses))), y=poses[:, i], name=label, mode="lines"),
                row=1, col=1,
            )
        for i, label in enumerate(["w", "x", "y", "z"]):
            fig.add_trace(
                go.Scatter(x=list(range(len(poses))), y=poses[:, i + 3], name=label, mode="lines"),
                row=2, col=1,
            )
        fig.update_xaxes(title_text="Frame", row=2, col=1)
        fig.update_yaxes(title_text="Position (m)", row=1, col=1)
        fig.update_yaxes(title_text="Quaternion", row=2, col=1)
        fig.update_layout(height=600, title_text=f"{side.capitalize()} Wrist Pose")
        st.plotly_chart(fig, use_container_width=True)


def json_plot_joint_states(data, side, frame_idx=None):
    joints = np.array(data[f"{side}_joint_states"])
    if frame_idx is not None:
        jvals = joints[frame_idx]
        st.write(f"**Frame {frame_idx} — {len(jvals)} joints:**")
        cols = st.columns(min(6, len(jvals)))
        for i, val in enumerate(jvals):
            cols[i % len(cols)].metric(f"J{i}", f"{val:.3f}")
    else:
        fig = go.Figure()
        for i in range(joints.shape[1]):
            fig.add_trace(go.Scatter(x=list(range(len(joints))), y=joints[:, i], name=f"Joint {i}", mode="lines"))
        fig.update_layout(
            title=f"{side.capitalize()} Joint States",
            xaxis_title="Frame",
            yaxis_title="Joint Angle (rad)",
            height=500,
        )
        st.plotly_chart(fig, use_container_width=True)


def json_plot_tactile_data(data, side, sensor_type, frame_idx=None):
    sensor_key = f"{side}_{sensor_type}_tactile"
    tactile = np.array(data[sensor_key])
    if len(tactile) == 0 or len(tactile[0]) == 0:
        st.warning(f"No data for {sensor_key}")
        return
    if frame_idx is not None:
        frame = np.array(tactile[frame_idx])
        fig = go.Figure(data=go.Heatmap(z=[frame], colorscale="Viridis", colorbar=dict(title="Force")))
        fig.update_layout(
            title=f"{side.capitalize()} {sensor_type.capitalize()} Tactile — Frame {frame_idx}",
            xaxis_title="Sensor Index",
            yaxis=dict(showticklabels=False),
            height=200,
        )
        st.plotly_chart(fig, use_container_width=True)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Min", f"{frame.min():.3f}")
        c2.metric("Max", f"{frame.max():.3f}")
        c3.metric("Mean", f"{frame.mean():.3f}")
        c4.metric("Std", f"{frame.std():.3f}")
    else:
        arr = np.array(tactile)
        fig = go.Figure(data=go.Heatmap(z=arr.T, colorscale="Viridis", colorbar=dict(title="Force")))
        fig.update_layout(
            title=f"{side.capitalize()} {sensor_type.capitalize()} Tactile Sensor",
            xaxis_title="Frame",
            yaxis_title="Sensor Index",
            height=500,
        )
        st.plotly_chart(fig, use_container_width=True)
        c1, c2, c3 = st.columns(3)
        c1.metric("Frames", arr.shape[0])
        c2.metric("Sensors", arr.shape[1])
        c3.metric("Total", arr.size)


def json_plot_all_tactile_comparison(data, side, frame_idx):
    sensors = ["finger_0", "finger_1", "finger_2", "palm"]
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=[s.replace("_", " ").title() for s in sensors],
        vertical_spacing=0.15,
        horizontal_spacing=0.1,
    )
    positions = [(1, 1), (1, 2), (2, 1), (2, 2)]
    for idx, sensor in enumerate(sensors):
        key = f"{side}_{sensor}_tactile"
        tactile = np.array(data[key])
        if len(tactile) > 0 and len(tactile[0]) > 0:
            r, c = positions[idx]
            fig.add_trace(
                go.Heatmap(z=[np.array(tactile[frame_idx])], colorscale="Viridis", showscale=(idx == 3)),
                row=r, col=c,
            )
    fig.update_layout(
        title_text=f"{side.capitalize()} — All Tactile (Frame {frame_idx})",
        height=500,
    )
    st.plotly_chart(fig, use_container_width=True)


def render_json_visualizer(data, side, viz_mode, frame_idx):
    """JSON 文件的完整可视化 UI"""
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["🎯 Wrist", "🦾 Joints", "👆 Fingers", "🖐️ Palm", "📊 All Tactile"]
    )
    with tab1:
        json_plot_wrist_pose(data, side, frame_idx)
    with tab2:
        json_plot_joint_states(data, side, frame_idx)
    with tab3:
        for finger in ["finger_0", "finger_1", "finger_2"]:
            with st.expander(f"{finger.replace('_', ' ').title()}", expanded=(frame_idx is not None)):
                json_plot_tactile_data(data, side, finger, frame_idx)
    with tab4:
        json_plot_tactile_data(data, side, "palm", frame_idx)
    with tab5:
        if frame_idx is not None:
            json_plot_all_tactile_comparison(data, side, frame_idx)
        else:
            st.info("Switch to Single Frame mode to see comparison")


# ============================================================================
# ==================  ROS Bag 可视化管线  ===================================
# ============================================================================

# --- Topic 定义 ---
OBSERVATION_TOPICS = {
    "left": "/robot/data/glove_left/observation",
    "right": "/robot/data/glove_right/observation",
}
HUMAN_POSE_TOPICS = {
    "pose_a": "/robot/data/human_pose_a/observation",
    "pose_b": "/robot/data/human_pose_b/observation",
}
TACTILE_TOPICS = {
    "left": {
        "finger_0": "/robot/data/glove_left_finger_0_tactile/tactile",
        "finger_1": "/robot/data/glove_left_finger_1_tactile/tactile",
        "finger_2": "/robot/data/glove_left_finger_2_tactile/tactile",
        "palm": "/robot/data/glove_left_palm_tactile/tactile",
    },
    "right": {
        "finger_0": "/robot/data/glove_right_finger_0_tactile/tactile",
        "finger_1": "/robot/data/glove_right_finger_1_tactile/tactile",
        "finger_2": "/robot/data/glove_right_finger_2_tactile/tactile",
        "palm": "/robot/data/glove_right_palm_tactile/tactile",
    },
}
IMAGE_TOPICS = {
    "color": "/robot/data/head_camera/color_image",
    "depth": "/robot/data/head_camera/depth_image",
}
CONFIG_TOPIC = "/robot/data/exoskeleton_glove/config"


# --- 自定义消息类型注册 ---
def _get_registered_typestore():
    from rosbags.typesys import Stores, get_typestore, get_types_from_msg

    typestore = get_typestore(Stores.ROS1_NOETIC)

    component_obs_msgdef = """std_msgs/Header header
geometry_msgs/PoseStamped[] track_poses
geometry_msgs/PoseStamped multibody_pose
geometry_msgs/WrenchStamped[] force_torques
multibody_msgs/MultibodyState multibody_state
multibody_msgs/MultibodyCommand multibody_command

================================================================================
MSG: multibody_msgs/MultibodyState
Header header
multibody_msgs/JointState[] states

================================================================================
MSG: multibody_msgs/JointState
float64 q
float64 qdot
float64 tau
float64 q_out
float64 qd_out
float64 tau_out
float64 temperature
float64 homed
float64 locked

================================================================================
MSG: multibody_msgs/MultibodyCommand
Header header
multibody_msgs/JointCommand[] commands

================================================================================
MSG: multibody_msgs/JointCommand
int8 mode
float64[] values
"""
    typestore.register(
        get_types_from_msg(component_obs_msgdef, "data_msgs/msg/ComponentObservation")
    )

    robot_config_msgdef = """std_msgs/Header header
multibody_msgs/MultibodyConfig[] configs

================================================================================
MSG: multibody_msgs/MultibodyConfig
string[] operation_mode_names
multibody_msgs/JointConfig[] configs

================================================================================
MSG: multibody_msgs/JointConfig
string name
float64 lower_position
float64 upper_position
int8[] supported_operation_modes
"""
    typestore.register(
        get_types_from_msg(robot_config_msgdef, "data_msgs/msg/RobotConfig")
    )

    event_msgdef = """std_msgs/Header header
string event_type
string event_detail
"""
    typestore.register(get_types_from_msg(event_msgdef, "data_msgs/msg/Event"))
    return typestore


# --- 数据加载函数 ---


@st.cache_data(show_spinner="Loading bag metadata...")
def bag_load_metadata(bag_path):
    from rosbags.rosbag1 import Reader as Rosbag1Reader

    _get_registered_typestore()
    metadata = {}
    topic_timestamps = defaultdict(list)
    with Rosbag1Reader(bag_path) as reader:
        metadata["duration_sec"] = reader.duration / 1e9
        metadata["start_time"] = reader.start_time / 1e9
        metadata["end_time"] = reader.end_time / 1e9
        metadata["message_count"] = reader.message_count
        metadata["topics"] = {}
        for topic_name, topic in reader.topics.items():
            metadata["topics"][topic_name] = {"type": topic.msgtype, "count": topic.msgcount}
        for conn, timestamp, _ in reader.messages():
            topic_timestamps[conn.topic].append(timestamp / 1e9)
    for k in topic_timestamps:
        topic_timestamps[k] = sorted(topic_timestamps[k])
    return metadata, dict(topic_timestamps)


@st.cache_data(show_spinner="Loading observation data...")
def bag_load_observation_data(bag_path):
    from rosbags.rosbag1 import Reader as Rosbag1Reader

    typestore = _get_registered_typestore()
    all_obs_topics = set(OBSERVATION_TOPICS.values()) | set(HUMAN_POSE_TOPICS.values())
    data = defaultdict(
        lambda: {
            "timestamps": [], "wrist_pose": [], "joint_q": [], "joint_qdot": [],
            "joint_tau": [], "track_poses": [], "track_names": None,
            "force_torques": [], "ft_names": None,
        }
    )
    with Rosbag1Reader(bag_path) as reader:
        connections = [c for c in reader.connections if c.topic in all_obs_topics]
        for conn, timestamp, rawdata in reader.messages(connections=connections):
            try:
                msg = typestore.deserialize_ros1(rawdata, conn.msgtype)
            except Exception:
                continue
            d = data[conn.topic]
            d["timestamps"].append(timestamp / 1e9)
            pose = msg.multibody_pose.pose
            d["wrist_pose"].append([
                pose.position.x, pose.position.y, pose.position.z,
                pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w,
            ])
            states = msg.multibody_state.states
            d["joint_q"].append([s.q for s in states])
            d["joint_qdot"].append([s.qdot for s in states])
            d["joint_tau"].append([s.tau for s in states])
            if len(msg.track_poses) > 0:
                if d["track_names"] is None:
                    d["track_names"] = [tp.header.frame_id for tp in msg.track_poses]
                frame_poses = {}
                for tp in msg.track_poses:
                    p = tp.pose
                    frame_poses[tp.header.frame_id] = [
                        p.position.x, p.position.y, p.position.z,
                        p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w,
                    ]
                d["track_poses"].append(frame_poses)
            if len(msg.force_torques) > 0:
                if d["ft_names"] is None:
                    d["ft_names"] = [ft.header.frame_id for ft in msg.force_torques]
                frame_ft = {}
                for ft in msg.force_torques:
                    w = ft.wrench
                    frame_ft[ft.header.frame_id] = [
                        w.force.x, w.force.y, w.force.z,
                        w.torque.x, w.torque.y, w.torque.z,
                    ]
                d["force_torques"].append(frame_ft)

    result = {}
    for topic, d in data.items():
        result[topic] = {
            "timestamps": np.array(d["timestamps"]),
            "wrist_pose": np.array(d["wrist_pose"]) if d["wrist_pose"] else np.array([]),
            "joint_q": np.array(d["joint_q"]) if d["joint_q"] else np.array([]),
            "joint_qdot": np.array(d["joint_qdot"]) if d["joint_qdot"] else np.array([]),
            "joint_tau": np.array(d["joint_tau"]) if d["joint_tau"] else np.array([]),
            "track_names": d["track_names"],
            "track_poses": d["track_poses"],
            "ft_names": d["ft_names"],
            "force_torques": d["force_torques"],
        }
    return result


@st.cache_data(show_spinner="Loading tactile data...")
def bag_load_tactile_data(bag_path):
    from rosbags.rosbag1 import Reader as Rosbag1Reader

    typestore = _get_registered_typestore()
    all_tactile = set()
    for side_topics in TACTILE_TOPICS.values():
        all_tactile.update(side_topics.values())
    tdata = defaultdict(list)
    tts = defaultdict(list)
    with Rosbag1Reader(bag_path) as reader:
        connections = [c for c in reader.connections if c.topic in all_tactile]
        for conn, timestamp, rawdata in reader.messages(connections=connections):
            msg = typestore.deserialize_ros1(rawdata, conn.msgtype)
            tdata[conn.topic].append(msg.data.copy())
            tts[conn.topic].append(timestamp / 1e9)
    result, ts_result = {}, {}
    for topic in all_tactile:
        if topic in tdata and len(tdata[topic]) > 0:
            result[topic] = np.array(tdata[topic])
            ts_result[topic] = np.array(tts[topic])
    return result, ts_result


@st.cache_data(show_spinner="Loading image index...")
def bag_load_image_index(bag_path):
    from rosbags.rosbag1 import Reader as Rosbag1Reader

    timestamps = {"color": [], "depth": []}
    with Rosbag1Reader(bag_path) as reader:
        for conn, timestamp, _ in reader.messages():
            if conn.topic == IMAGE_TOPICS["color"]:
                timestamps["color"].append(timestamp / 1e9)
            elif conn.topic == IMAGE_TOPICS["depth"]:
                timestamps["depth"].append(timestamp / 1e9)
    return timestamps


def bag_load_single_image(bag_path, topic, target_idx):
    from rosbags.rosbag1 import Reader as Rosbag1Reader

    typestore = _get_registered_typestore()
    with Rosbag1Reader(bag_path) as reader:
        connections = [c for c in reader.connections if c.topic == topic]
        idx = 0
        for conn, timestamp, rawdata in reader.messages(connections=connections):
            if idx == target_idx:
                msg = typestore.deserialize_ros1(rawdata, conn.msgtype)
                return msg.data, msg.format, timestamp / 1e9
            idx += 1
    return None, None, None


@st.cache_data(show_spinner="Loading config...")
def bag_load_config(bag_path):
    from rosbags.rosbag1 import Reader as Rosbag1Reader

    typestore = _get_registered_typestore()
    with Rosbag1Reader(bag_path) as reader:
        connections = [c for c in reader.connections if c.topic == CONFIG_TOPIC]
        for conn, timestamp, rawdata in reader.messages(connections=connections):
            try:
                msg = typestore.deserialize_ros1(rawdata, conn.msgtype)
                configs = []
                for cfg in msg.configs:
                    joints = [
                        {"name": jc.name, "lower": jc.lower_position, "upper": jc.upper_position}
                        for jc in cfg.configs
                    ]
                    configs.append({"operation_modes": list(cfg.operation_mode_names), "joints": joints})
                return configs
            except Exception:
                return None
    return None


# --- Bag 可视化函数 ---


def bag_plot_wrist_from_track_poses(obs_data, topic, part_name, frame_idx=None):
    d = obs_data[topic]
    track_poses = d["track_poses"]
    ts = d["timestamps"]
    if not track_poses:
        st.warning("No track poses data")
        return
    poses = []
    for frame in track_poses:
        poses.append(frame.get(part_name, [np.nan] * 7))
    poses = np.array(poses)
    rel_time = ts - ts[0]

    if frame_idx is not None:
        pose = poses[frame_idx]
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**Position** (t={rel_time[frame_idx]:.3f}s)")
            c1, c2, c3 = st.columns(3)
            c1.metric("X", f"{pose[0]:.4f}")
            c2.metric("Y", f"{pose[1]:.4f}")
            c3.metric("Z", f"{pose[2]:.4f}")
        with col2:
            st.markdown("**Orientation (quaternion)**")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("qx", f"{pose[3]:.4f}")
            c2.metric("qy", f"{pose[4]:.4f}")
            c3.metric("qz", f"{pose[5]:.4f}")
            c4.metric("qw", f"{pose[6]:.4f}")
    else:
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=("Position (x, y, z)", "Orientation (qx, qy, qz, qw)"),
            vertical_spacing=0.12,
        )
        colors_pos = ["#ff6b6b", "#4ecdc4", "#ffe66d"]
        for i, label in enumerate(["x", "y", "z"]):
            fig.add_trace(
                go.Scatter(x=rel_time, y=poses[:, i], name=label, mode="lines", line=dict(color=colors_pos[i])),
                row=1, col=1,
            )
        colors_quat = ["#a29bfe", "#fd79a8", "#00cec9", "#636e72"]
        for i, label in enumerate(["qx", "qy", "qz", "qw"]):
            fig.add_trace(
                go.Scatter(x=rel_time, y=poses[:, i + 3], name=label, mode="lines", line=dict(color=colors_quat[i])),
                row=2, col=1,
            )
        fig.update_xaxes(title_text="Time (s)", row=2, col=1)
        fig.update_yaxes(title_text="Position (m)", row=1, col=1)
        fig.update_yaxes(title_text="Quaternion", row=2, col=1)
        fig.update_layout(height=550, margin=dict(l=40, r=40, t=40, b=40))
        st.plotly_chart(fig, use_container_width=True)


def bag_plot_joint_states(obs_data, topic, field="joint_q", frame_idx=None):
    d = obs_data[topic]
    joints = d[field]
    ts = d["timestamps"]
    if len(joints) == 0:
        st.warning("No joint data")
        return
    rel_time = ts - ts[0]
    labels = {"joint_q": "q (rad)", "joint_qdot": "qdot (rad/s)", "joint_tau": "tau (Nm)"}
    if frame_idx is not None:
        jvals = joints[frame_idx]
        st.markdown(f"**{labels.get(field, field)}** — Frame {frame_idx} — {len(jvals)} joints")
        cols = st.columns(min(8, len(jvals)))
        for i, val in enumerate(jvals):
            cols[i % len(cols)].metric(f"J{i}", f"{val:.4f}")
    else:
        fig = go.Figure()
        for i in range(joints.shape[1]):
            fig.add_trace(go.Scatter(x=rel_time, y=joints[:, i], name=f"J{i}", mode="lines"))
        fig.update_layout(
            title=labels.get(field, field),
            xaxis_title="Time (s)",
            yaxis_title=labels.get(field, field),
            height=450,
            margin=dict(l=40, r=40, t=40, b=40),
        )
        st.plotly_chart(fig, use_container_width=True)


def bag_plot_track_poses(obs_data, topic, frame_idx=None):
    d = obs_data[topic]
    track_poses = d["track_poses"]
    track_names = d["track_names"]
    ts = d["timestamps"]
    if not track_poses or not track_names:
        st.info("No track poses")
        return
    rel_time = ts - ts[0]

    if frame_idx is not None:
        frame = track_poses[frame_idx]
        st.markdown(f"**Frame {frame_idx}** — {len(track_names)} parts")
        for name in track_names:
            if name in frame:
                p = frame[name]
                with st.expander(f"📍 {name}", expanded=False):
                    c1, c2, c3 = st.columns(3)
                    c1.metric("X", f"{p[0]:.4f}")
                    c2.metric("Y", f"{p[1]:.4f}")
                    c3.metric("Z", f"{p[2]:.4f}")
                    c4, c5, c6, c7 = st.columns(4)
                    c4.metric("qx", f"{p[3]:.4f}")
                    c5.metric("qy", f"{p[4]:.4f}")
                    c6.metric("qz", f"{p[5]:.4f}")
                    c7.metric("qw", f"{p[6]:.4f}")
    else:
        st.markdown("**Position**")
        for comp_idx, comp_label in enumerate(["X", "Y", "Z"]):
            fig = go.Figure()
            for name in track_names:
                vals = [fr.get(name, [np.nan] * 7)[comp_idx] for fr in track_poses]
                fig.add_trace(go.Scatter(x=rel_time, y=vals, name=name, mode="lines"))
            fig.update_layout(
                title=f"Position — {comp_label}",
                xaxis_title="Time (s)",
                yaxis_title=f"{comp_label} (m)",
                height=350,
                margin=dict(l=40, r=40, t=40, b=40),
            )
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        st.markdown("**Orientation (quaternion)**")
        for comp_idx, comp_label in enumerate(["qx", "qy", "qz", "qw"]):
            fig = go.Figure()
            for name in track_names:
                vals = [fr.get(name, [np.nan] * 7)[comp_idx + 3] for fr in track_poses]
                fig.add_trace(go.Scatter(x=rel_time, y=vals, name=name, mode="lines"))
            fig.update_layout(
                title=f"Orientation — {comp_label}",
                xaxis_title="Time (s)",
                yaxis_title=comp_label,
                height=350,
                margin=dict(l=40, r=40, t=40, b=40),
            )
            st.plotly_chart(fig, use_container_width=True)


def bag_plot_tactile_heatmap_single(data_array, timestamps, topic, frame_idx):
    frame = data_array[frame_idx]
    short = topic.split("/")[-2].replace("_tactile", "")
    fig = go.Figure(data=go.Heatmap(z=[frame], colorscale="Viridis", colorbar=dict(title="Force")))
    fig.update_layout(
        title=f"{short} — Frame {frame_idx}",
        xaxis_title="Sensor Index",
        yaxis=dict(showticklabels=False),
        height=200,
        margin=dict(l=40, r=40, t=40, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Min", f"{frame.min():.2f}")
    c2.metric("Max", f"{frame.max():.2f}")
    c3.metric("Mean", f"{frame.mean():.2f}")
    c4.metric("Std", f"{frame.std():.2f}")


def bag_plot_tactile_timeseries(data_array, timestamps, topic):
    rel = timestamps - timestamps[0]
    short = topic.split("/")[-2].replace("_tactile", "")
    fig = go.Figure(data=go.Heatmap(z=data_array.T, x=rel, colorscale="Viridis", colorbar=dict(title="Force")))
    fig.update_layout(
        title=f"{short}",
        xaxis_title="Time (s)",
        yaxis_title="Sensor",
        height=350,
        margin=dict(l=40, r=40, t=40, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)
    c1, c2, c3 = st.columns(3)
    c1.metric("Frames", data_array.shape[0])
    c2.metric("Sensors", data_array.shape[1])
    c3.metric("Max", f"{data_array.max():.2f}")


def bag_plot_tactile_stats(data_array, timestamps, topic):
    rel = timestamps - timestamps[0]
    short = topic.split("/")[-2].replace("_tactile", "")
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        subplot_titles=("Max", "Mean", "Std"), vertical_spacing=0.08,
    )
    fig.add_trace(go.Scatter(x=rel, y=data_array.max(axis=1), mode="lines", line=dict(color="#ff6b6b")), row=1, col=1)
    fig.add_trace(go.Scatter(x=rel, y=data_array.mean(axis=1), mode="lines", line=dict(color="#4ecdc4")), row=2, col=1)
    fig.add_trace(go.Scatter(x=rel, y=data_array.std(axis=1), mode="lines", line=dict(color="#ffe66d")), row=3, col=1)
    fig.update_xaxes(title_text="Time (s)", row=3, col=1)
    fig.update_layout(title=f"{short} Stats", height=500, showlegend=False, margin=dict(l=40, r=40, t=60, b=40))
    st.plotly_chart(fig, use_container_width=True)


def bag_plot_all_tactile_comparison(tactile_data, side, frame_idx):
    sensors = ["finger_0", "finger_1", "finger_2", "palm"]
    topics = TACTILE_TOPICS[side]
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[s.replace("_", " ").title() for s in sensors],
        vertical_spacing=0.15, horizontal_spacing=0.1,
    )
    positions = [(1, 1), (1, 2), (2, 1), (2, 2)]
    gmax = max(
        (tactile_data[topics[s]][frame_idx].max() for s in sensors if topics[s] in tactile_data),
        default=1,
    )
    for idx, sensor in enumerate(sensors):
        tp = topics[sensor]
        if tp in tactile_data:
            r, c = positions[idx]
            fig.add_trace(
                go.Heatmap(
                    z=[tactile_data[tp][frame_idx]],
                    colorscale="Viridis",
                    showscale=(idx == 3),
                    zmin=0,
                    zmax=max(gmax, 1),
                ),
                row=r, col=c,
            )
    fig.update_layout(
        title_text=f"{side.capitalize()} Tactile (Frame {frame_idx})",
        height=450,
        margin=dict(l=40, r=40, t=60, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)


def bag_plot_frequency_analysis(topic_timestamps, topic_name):
    ts = np.array(topic_timestamps[topic_name])
    if len(ts) < 2:
        st.warning("Not enough messages")
        return
    intervals = np.diff(ts) * 1000
    rel = ts[1:] - ts[0]
    avg = np.mean(intervals)
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=("Interval (ms)", "Distribution"), vertical_spacing=0.15,
    )
    fig.add_trace(go.Scatter(x=rel, y=intervals, mode="lines", line=dict(width=0.5)), row=1, col=1)
    fig.add_hline(y=avg, line_dash="dash", line_color="red", annotation_text=f"Mean: {avg:.2f}ms", row=1, col=1)
    fig.add_trace(go.Histogram(x=intervals, nbinsx=100), row=2, col=1)
    fig.update_xaxes(title_text="Time (s)", row=1, col=1)
    fig.update_xaxes(title_text="ms", row=2, col=1)
    short = "/".join(topic_name.split("/")[-2:])
    fig.update_layout(title=f"Frequency: {short}", height=500, showlegend=False, margin=dict(l=40, r=40, t=60, b=40))
    st.plotly_chart(fig, use_container_width=True)
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Hz", f"{1000 / avg:.1f}")
    c2.metric("Mean", f"{avg:.2f}ms")
    c3.metric("Std", f"{np.std(intervals):.2f}ms")
    c4.metric("Min", f"{np.min(intervals):.2f}ms")
    c5.metric("Max", f"{np.max(intervals):.2f}ms")
    lg = intervals[intervals > avg * 3]
    if len(lg) > 0:
        st.warning(f"⚠️ {len(lg)} large gaps (>{avg * 3:.1f}ms)")
    if np.any(intervals < 0):
        st.error(f"🚨 {np.sum(intervals < 0)} out-of-order")
    if np.any(intervals == 0):
        st.warning(f"⚠️ {np.sum(intervals == 0)} duplicates")


def bag_plot_cross_topic_sync(topic_timestamps):
    all_120hz = []
    for side_topics in TACTILE_TOPICS.values():
        all_120hz.extend(side_topics.values())
    all_120hz.extend(OBSERVATION_TOPICS.values())
    all_120hz.extend(HUMAN_POSE_TOPICS.values())
    sync = [t for t in all_120hz if t in topic_timestamps]
    if len(sync) < 2:
        st.warning("Not enough topics")
        return
    ref = sync[0]
    ref_ts = np.array(topic_timestamps[ref])
    fig = go.Figure()
    for t in sync[1:]:
        ots = np.array(topic_timestamps[t])
        ml = min(len(ref_ts), len(ots))
        fig.add_trace(
            go.Box(y=(ots[:ml] - ref_ts[:ml]) * 1000, name=t.split("/robot/data/")[-1], boxpoints=False)
        )
    fig.update_layout(
        title=f"Offset vs {ref.split('/robot/data/')[-1]} (ms)",
        yaxis_title="ms",
        height=500,
        margin=dict(l=40, r=40, t=60, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)


# --- Bag 主渲染 ---


def render_bag_visualizer(bag_path, side, viz_mode, frame_idx):
    """ROS bag 文件的完整可视化 UI"""
    metadata, topic_timestamps = bag_load_metadata(bag_path)

    # 单帧 slider
    if viz_mode == "Single Frame":
        with st.sidebar:
            obs_topic = OBSERVATION_TOPICS[side]
            if obs_topic in topic_timestamps:
                num_frames = len(topic_timestamps[obs_topic])
                frame_idx = st.slider("Frame", 0, max(num_frames - 1, 0), 0)
                rel_t = topic_timestamps[obs_topic][frame_idx] - topic_timestamps[obs_topic][0]
                st.caption(f"t = {rel_t:.3f}s | {frame_idx}/{num_frames}")

    # 概览
    file_size_mb = os.path.getsize(bag_path) / 1024 / 1024
    with st.expander("📋 Bag Overview", expanded=False):
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Duration", f"{metadata['duration_sec']:.1f}s")
        c2.metric("Messages", f"{metadata['message_count']:,}")
        c3.metric("Topics", len(metadata["topics"]))
        c4.metric("Size", f"{file_size_mb:.0f} MB")
        config = bag_load_config(bag_path)
        if config:
            for i, cfg in enumerate(config):
                st.markdown(f"**Config {i}** — Modes: {cfg['operation_modes']}")
                if cfg["joints"]:
                    st.text(
                        "  Joints: "
                        + ", ".join(
                            [f"{j['name']}[{j['lower']:.1f},{j['upper']:.1f}]" for j in cfg["joints"]]
                        )
                    )

    # Tabs
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(
        ["🎯 Wrist Pose", "🦾 Joints", "👆 Tactile", "🧍 Human Pose", "📷 Camera", "📊 Quality", "📋 Overview"]
    )

    with tab1:
        st.subheader(f"{side.capitalize()} Wrist Pose")
        st.caption("Source: human_pose_b → LEFT_HAND / RIGHT_HAND")
        obs_data = bag_load_observation_data(bag_path)
        pose_b_topic = HUMAN_POSE_TOPICS["pose_b"]
        if pose_b_topic in obs_data:
            hand_name_map = {"left": "LEFT_HAND", "right": "RIGHT_HAND"}
            part_name = hand_name_map[side]
            track_names = obs_data[pose_b_topic].get("track_names")
            if track_names and part_name in track_names:
                bag_plot_wrist_from_track_poses(obs_data, pose_b_topic, part_name, frame_idx)
            elif track_names:
                matched = [n for n in track_names if "hand" in n.lower() and side in n.lower()]
                if matched:
                    st.info(f"Using '{matched[0]}' (exact '{part_name}' not found)")
                    bag_plot_wrist_from_track_poses(obs_data, pose_b_topic, matched[0], frame_idx)
                else:
                    st.warning(f"'{part_name}' not found. Available: {track_names}")
            else:
                st.warning("No track_poses in human_pose_b")
        else:
            st.warning("human_pose_b not available")

    with tab2:
        st.subheader(f"{side.capitalize()} Joint States")
        obs_data = bag_load_observation_data(bag_path)
        ot = OBSERVATION_TOPICS[side]
        if ot in obs_data:
            jf = st.selectbox(
                "Field",
                ["joint_q", "joint_qdot", "joint_tau"],
                format_func=lambda x: {
                    "joint_q": "Position (q)",
                    "joint_qdot": "Velocity (qdot)",
                    "joint_tau": "Torque (tau)",
                }[x],
            )
            bag_plot_joint_states(obs_data, ot, field=jf, frame_idx=frame_idx)

    with tab3:
        st.subheader(f"{side.capitalize()} Tactile")
        tdata, tts = bag_load_tactile_data(bag_path)
        tmode = st.radio("View", ["Time Series", "Single Frame", "Statistics"], horizontal=True, key="bag_tac_mode")
        tf = frame_idx
        if tmode == "Single Frame" and tf is not None:
            bag_plot_all_tactile_comparison(tdata, side, tf)
            st.markdown("---")
            for sn, tp in TACTILE_TOPICS[side].items():
                if tp in tdata:
                    with st.expander(f"📍 {sn.replace('_', ' ').title()}", expanded=False):
                        bag_plot_tactile_heatmap_single(tdata[tp], tts[tp], tp, tf)
        elif tmode == "Statistics":
            for sn, tp in TACTILE_TOPICS[side].items():
                if tp in tdata:
                    with st.expander(f"📍 {sn.replace('_', ' ').title()}", expanded=True):
                        bag_plot_tactile_stats(tdata[tp], tts[tp], tp)
        else:
            for sn, tp in TACTILE_TOPICS[side].items():
                if tp in tdata:
                    with st.expander(f"📍 {sn.replace('_', ' ').title()}", expanded=True):
                        bag_plot_tactile_timeseries(tdata[tp], tts[tp], tp)

    with tab4:
        st.subheader("Human Body Pose")
        obs_data = bag_load_observation_data(bag_path)
        psrc = st.selectbox(
            "Source",
            list(HUMAN_POSE_TOPICS.keys()),
            format_func=lambda x: x.replace("_", " ").title(),
        )
        pt = HUMAN_POSE_TOPICS[psrc]
        if pt in obs_data:
            st.markdown("**Track Poses (body parts)**")
            bag_plot_track_poses(obs_data, pt, frame_idx)
        else:
            st.warning("No data")

    with tab5:
        st.subheader("Head Camera")
        img_ts = bag_load_image_index(bag_path)
        cc, cd = st.columns(2)
        with cc:
            st.markdown("**Color (jpg)**")
            if img_ts["color"]:
                nc = len(img_ts["color"])
                ci = st.slider("Color Frame", 0, nc - 1, 0, key="bag_cs")
                st.caption(f"{ci}/{nc} | t={img_ts['color'][ci] - img_ts['color'][0]:.3f}s")
                idata, ifmt, _ = bag_load_single_image(bag_path, IMAGE_TOPICS["color"], ci)
                if idata is not None:
                    from PIL import Image

                    im = Image.open(io.BytesIO(bytes(idata)))
                    st.image(im, use_container_width=True)
                    st.caption(f"{im.size[0]}x{im.size[1]} {ifmt}")
            else:
                st.info("No color images")
        with cd:
            st.markdown("**Depth (png)**")
            if img_ts["depth"]:
                nd = len(img_ts["depth"])
                di = st.slider("Depth Frame", 0, nd - 1, 0, key="bag_ds")
                st.caption(f"{di}/{nd} | t={img_ts['depth'][di] - img_ts['depth'][0]:.3f}s")
                idata, ifmt, _ = bag_load_single_image(bag_path, IMAGE_TOPICS["depth"], di)
                if idata is not None:
                    from PIL import Image

                    im = Image.open(io.BytesIO(bytes(idata)))
                    depth_arr = np.array(im, dtype=np.float32)
                    valid = depth_arr[depth_arr > 0]
                    dmin = np.min(valid) if len(valid) > 0 else 0
                    dmax = np.max(depth_arr)
                    if dmax > dmin:
                        depth_norm = np.clip((depth_arr - dmin) / (dmax - dmin) * 255, 0, 255).astype(np.uint8)
                    else:
                        depth_norm = np.zeros_like(depth_arr, dtype=np.uint8)
                    fig_d = go.Figure(
                        data=go.Heatmap(
                            z=depth_norm[::-1],
                            colorscale="Turbo",
                            showscale=True,
                            colorbar=dict(title="Depth"),
                        )
                    )
                    fig_d.update_layout(
                        height=400,
                        margin=dict(l=0, r=0, t=0, b=0),
                        xaxis=dict(showticklabels=False),
                        yaxis=dict(showticklabels=False),
                    )
                    st.plotly_chart(fig_d, use_container_width=True)
                    st.caption(f"{im.size[0]}x{im.size[1]} {ifmt} | [{dmin:.0f}, {dmax:.0f}]")
            else:
                st.info("No depth images")

    with tab6:
        st.subheader("Data Quality")
        sel = st.selectbox(
            "Topic",
            sorted(topic_timestamps.keys()),
            format_func=lambda x: f"{x.split('/robot/data/')[-1]} ({len(topic_timestamps[x])})",
        )
        if sel:
            bag_plot_frequency_analysis(topic_timestamps, sel)
        st.markdown("---")
        st.subheader("Cross-Topic Sync (120 Hz)")
        bag_plot_cross_topic_sync(topic_timestamps)
        if IMAGE_TOPICS["color"] in topic_timestamps and IMAGE_TOPICS["depth"] in topic_timestamps:
            st.markdown("---")
            st.subheader("Camera Sync")
            cts = np.array(topic_timestamps[IMAGE_TOPICS["color"]])
            dts = np.array(topic_timestamps[IMAGE_TOPICS["depth"]])
            ml = min(len(cts), len(dts))
            offs = (dts[:ml] - cts[:ml]) * 1000
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=np.arange(ml), y=offs, mode="lines"))
            fig.update_layout(title="Depth-Color Offset", xaxis_title="Frame", yaxis_title="ms", height=300)
            st.plotly_chart(fig, use_container_width=True)
            c1, c2, c3 = st.columns(3)
            c1.metric("Mean", f"{np.mean(offs):.2f}ms")
            c2.metric("Std", f"{np.std(offs):.2f}ms")
            c3.metric("Max|off|", f"{np.max(np.abs(offs)):.2f}ms")

    with tab7:
        st.subheader("All Topics")
        import pandas as pd

        rows = []
        for tn in sorted(topic_timestamps.keys()):
            ts_arr = topic_timestamps[tn]
            cnt = len(ts_arr)
            if cnt >= 2:
                ivs = np.diff(ts_arr) * 1000
                af = 1000 / np.mean(ivs)
                sd = np.std(ivs)
                mg = np.max(ivs)
                lg = int(np.sum(ivs > np.mean(ivs) * 3))
                s = "✅" if lg == 0 and sd < np.mean(ivs) * 0.5 else "⚠️"
            else:
                af = sd = mg = 0
                lg = 0
                s = "ℹ️"
            rows.append(
                {
                    "": s,
                    "Topic": tn.split("/robot/data/")[-1],
                    "Count": cnt,
                    "Hz": round(af, 1),
                    "Std(ms)": round(sd, 2),
                    "MaxGap": round(mg, 1),
                    "Gaps": lg,
                }
            )
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        st.markdown("---")
        st.markdown("**Timeline**")
        fig = go.Figure()
        for tn in sorted(topic_timestamps.keys()):
            ts_arr = np.array(topic_timestamps[tn])
            rel = ts_arr - metadata["start_time"]
            short = tn.split("/robot/data/")[-1]
            step = max(1, len(rel) // 2000)
            fig.add_trace(
                go.Scatter(
                    x=rel[::step],
                    y=[short] * len(rel[::step]),
                    mode="markers",
                    marker=dict(size=2),
                    name=short,
                    showlegend=False,
                )
            )
        fig.update_layout(xaxis_title="Time (s)", height=500, margin=dict(l=200, r=40, t=40, b=40))
        st.plotly_chart(fig, use_container_width=True)


# ============================================================================
# 主应用
# ============================================================================


def main():
    if not check_password():
        st.stop()

    st.title("🤖 Robot Sensor Data Visualizer")
    st.markdown("---")

    # --- Drive 连接 ---
    try:
        if "gcp_service_account" not in st.secrets:
            st.error("❌ Drive credentials not configured!")
            return
        folder_id = (
            st.secrets["gcp_service_account"].get("gdrive_folder_id")
            or st.secrets.get("gdrive_folder_id")
        )
        if not folder_id:
            st.error("❌ gdrive_folder_id not configured!")
            return
    except Exception as e:
        st.error(f"❌ Error reading secrets: {e}")
        return

    service = get_gdrive_service()
    if service is None:
        return

    # --- 加载文件列表 ---
    with st.spinner("Loading file list from Google Drive..."):
        all_files = list_data_files_from_gdrive(service, folder_id)

    if not all_files:
        st.error("No .json or .bag files found in Google Drive folder.")
        if st.button("🔄 Refresh Connection & Retry", type="primary"):
            st.cache_resource.clear()
            st.cache_data.clear()
            st.rerun()
        return

    structure = build_folder_structure(all_files)

    # --- 初始化 session state ---
    if "current_path" not in st.session_state:
        st.session_state.current_path = []
    if "selected_file" not in st.session_state:
        st.session_state.selected_file = None

    # --- 侧边栏文件浏览器 ---
    with st.sidebar:
        st.header("📁 File Browser")

        n_json = sum(1 for f in all_files if f["type"] == "json")
        n_bag = sum(1 for f in all_files if f["type"] == "bag")
        st.success(f"Total: {len(all_files)} files ({n_json} JSON, {n_bag} BAG)")

        if st.button("🔄 Refresh", use_container_width=True, help="Clear cache and reconnect"):
            st.cache_resource.clear()
            st.cache_data.clear()
            st.rerun()

        # 面包屑
        breadcrumb = " / ".join(["Root"] + st.session_state.current_path)
        st.markdown(f"**📂 {breadcrumb}**")

        if st.session_state.current_path:
            if st.button("⬆️ Back", use_container_width=True):
                st.session_state.current_path.pop()
                st.session_state.selected_file = None
                st.rerun()

        st.markdown("---")

        # 导航到当前节点
        current = structure
        for folder_name in st.session_state.current_path:
            if folder_name in current["__subfolders__"]:
                current = current["__subfolders__"][folder_name]
            else:
                st.session_state.current_path = []
                st.rerun()
                return

        # 子文件夹
        subfolders = sorted(current["__subfolders__"].keys())
        if subfolders:
            st.subheader(f"📂 Folders ({len(subfolders)})")
            for folder in subfolders:

                def count_files(node):
                    c = len(node.get("__files__", []))
                    for sub in node.get("__subfolders__", {}).values():
                        c += count_files(sub)
                    return c

                fc = count_files(current["__subfolders__"][folder])
                if st.button(f"📁 {folder} ({fc})", key=f"fold_{folder}", use_container_width=True):
                    st.session_state.current_path.append(folder)
                    st.session_state.selected_file = None
                    st.rerun()

        # 文件列表
        files = sorted(current.get("__files__", []), key=lambda x: x["name"])
        if files:
            st.markdown("---")
            st.subheader(f"📄 Files ({len(files)})")
            for fi in files:
                is_sel = st.session_state.selected_file and st.session_state.selected_file["id"] == fi["id"]
                btn_type = "primary" if is_sel else "secondary"
                icon = "✓ " if is_sel else ""
                size_mb = fi["size"] / 1024 / 1024
                type_tag = "🟢" if fi["type"] == "json" else "🔵"
                label = f"{icon}{type_tag} {fi['name']}"
                if size_mb >= 1:
                    label += f" ({size_mb:.0f}MB)"

                if st.button(label, key=f"file_{fi['id']}", type=btn_type, use_container_width=True):
                    st.session_state.selected_file = fi
                    st.rerun()

            # 前后翻页
            if st.session_state.selected_file:
                st.markdown("---")
                cidx = next(
                    (i for i, f in enumerate(files) if f["id"] == st.session_state.selected_file["id"]),
                    None,
                )
                if cidx is not None:
                    c1, c2 = st.columns(2)
                    if c1.button("⬅️", disabled=(cidx == 0), use_container_width=True):
                        st.session_state.selected_file = files[cidx - 1]
                        st.rerun()
                    if c2.button("➡️", disabled=(cidx == len(files) - 1), use_container_width=True):
                        st.session_state.selected_file = files[cidx + 1]
                        st.rerun()
                    st.caption(f"File {cidx + 1} / {len(files)}")

        if not files and not subfolders:
            st.info("Empty folder")
            return

        if not st.session_state.selected_file:
            st.info("👆 Select a file")
            return

    # --- 选中文件后 ---
    selected = st.session_state.selected_file
    if not selected:
        st.info("Please select a file from the sidebar")
        return

    file_type = selected["type"]
    size_mb = selected["size"] / 1024 / 1024

    # 侧边栏公共选项
    with st.sidebar:
        st.markdown("---")
        st.header("⚙️ Options")
        side = st.radio("Arm / Hand", ["left", "right"], horizontal=True)
        viz_mode = st.radio("Mode", ["Time Series", "Single Frame"])

    frame_idx = None

    # === JSON 管线 ===
    if file_type == "json":
        st.caption(f"📄 JSON — {selected['name']} ({size_mb:.1f} MB)")

        try:
            with st.spinner(f"Loading {selected['name']}..."):
                data = download_json_from_gdrive(service, selected["id"])
            if data is None:
                st.error("Failed to load file")
                if st.button("🔄 Retry", type="primary"):
                    st.cache_data.clear()
                    st.rerun()
                return

            if viz_mode == "Single Frame":
                with st.sidebar:
                    num_frames = len(data[f"{side}_wrist_pose"])
                    frame_idx = st.slider("Frame", 0, num_frames - 1, 0)

            with st.expander("📊 Data Summary"):
                c1, c2, c3 = st.columns(3)
                c1.metric("File", selected["name"])
                c2.metric("Frames", len(data[f"{side}_wrist_pose"]))
                c3.metric("Keys", len(data.keys()))

            render_json_visualizer(data, side, viz_mode, frame_idx)

        except Exception as e:
            st.error(f"Error: {e}")
            st.exception(e)
            if st.button("🔄 Clear Cache & Retry", type="primary"):
                st.cache_resource.clear()
                st.cache_data.clear()
                st.rerun()

    # === BAG 管线 ===
    elif file_type == "bag":
        st.caption(f"🔵 ROS Bag — {selected['name']} ({size_mb:.1f} MB)")

        if size_mb > 400:
            st.warning(
                f"⚠️ This bag file is {size_mb:.0f} MB. "
                "Download and parsing may take a while and could exceed memory limits on Streamlit Cloud free tier."
            )

        try:
            with st.spinner(f"Downloading {selected['name']} ({size_mb:.0f} MB)..."):
                bag_path = download_bag_to_temp(service, selected["id"], selected["name"])

            if bag_path is None:
                st.error("Failed to download bag file")
                if st.button("🔄 Retry", type="primary"):
                    st.cache_data.clear()
                    st.rerun()
                return

            render_bag_visualizer(bag_path, side, viz_mode, frame_idx)

        except Exception as e:
            st.error(f"Error: {e}")
            st.exception(e)
            if st.button("🔄 Clear Cache & Retry", type="primary"):
                st.cache_resource.clear()
                st.cache_data.clear()
                st.rerun()


if __name__ == "__main__":
    main()
