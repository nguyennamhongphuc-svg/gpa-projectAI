import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
import os

# 1. CẤU HÌNH TRANG CHỦ ĐỀ CÔNG NGHỆ
st.set_page_config(
    page_title="AI STUDENT CORE v2.0",
    page_icon="🤖",
    layout="wide"
)

# Tùy chỉnh giao diện Cyberpunk/Robot
st.markdown("""
    <style>
    /* Nền tối chủ đạo công nghệ */
    .stApp {
        background-color: #0e1117;
        color: #e0e0e0;
    }
    
    /* Hiệu ứng khối kính mờ (Glassmorphism) */
    .stMetric, .css-1r6slb0, .css-k7v80w {
        background: rgba(23, 28, 35, 0.8);
        border: 1px solid #00f2ff;
        border-radius: 15px;
        box-shadow: 0 0 15px rgba(0, 242, 255, 0.2);
    }

    /* Tiêu đề phong cách Robot */
    .robot-title {
        font-family: 'Courier New', Courier, monospace;
        color: #00f2ff;
        text-transform: uppercase;
        letter-spacing: 3px;
        text-shadow: 2px 2px 10px rgba(0, 242, 255, 0.5);
        border-bottom: 2px solid #00f2ff;
        padding-bottom: 10px;
    }

    /* Khối dự đoán GPA */
    .gpa-container {
        background: linear-gradient(180deg, #16213e 0%, #0f3460 100%);
        border: 2px solid #00f2ff;
        padding: 30px;
        border-radius: 20px;
        text-align: center;
        margin-bottom: 25px;
    }

    /* Khối lời khuyên AI */
    .ai-advice-box {
        background: rgba(0, 255, 128, 0.05);
        border-left: 5px solid #00ff80;
        padding: 20px;
        border-radius: 5px 15px 15px 5px;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    /* Nút bấm Neon */
    .stButton>button {
        background-color: transparent;
        color: #00f2ff;
        border: 2px solid #00f2ff;
        border-radius: 30px;
        transition: 0.3s;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #00f2ff;
        color: #000;
        box-shadow: 0 0 20px #00f2ff;
    }
    </style>
    """, unsafe_allow_html=True)

# 2. HÀM XỬ LÝ DỮ LIỆU (Fix lỗi ValueError)
@st.cache_resource
def load_and_train(file_path):
    if not os.path.exists(file_path):
        return None, None, None
    
    df = pd.read_csv(file_path).dropna()
    df_ml = df.copy()
    
    # Mã hóa dữ liệu
    mapping = {'Có': 1, 'Không': 0, 'Tự học': 0, 'Học nhóm': 1}
    for col in ['Làm_Thêm', 'Tham_Gia_CLB', 'Hình_Thức_Học']:
        if col in df_ml.columns:
            df_ml[col] = df_ml[col].map(mapping)
    
    X = df_ml.drop('GPA', axis=1)
    y = df_ml['GPA']
    feature_names = X.columns.tolist()
    
    model = RandomForestRegressor(n_estimators=150, random_state=42)
    model.fit(X, y)
    
    return model, df, feature_names

model, original_df, feature_cols = load_and_train('Dataset_GPA_Thuc_Te_250_Responses.csv')

if model is None:
    st.error("🤖 HỆ THỐNG: Thiếu file dữ liệu nguồn. Vui lòng nạp 'Dataset_GPA_Thuc_Te_250_Responses.csv'.")
    st.stop()

# 3. SIDEBAR (CONTROL PANEL)
st.sidebar.markdown("<h2 style='color:#00f2ff;'>⚙️ CORE SETTINGS</h2>", unsafe_allow_html=True)

input_data = {
    'Số_Giờ_Học_Tuần': st.sidebar.slider("📚 Tổng giờ tự học/tuần", 0, 60, 20),
    'Số_Môn_Đang_Học': st.sidebar.number_input("📝 Số môn đăng ký", 1, 12, 5),
    'Làm_Thêm': st.sidebar.selectbox("💼 Công việc bên ngoài", ["Không", "Có"]),
    'Thời_Gian_Ngủ': st.sidebar.slider("😴 Thời gian phục hồi (giờ)", 3, 12, 7),
    'Tham_Gia_CLB': st.sidebar.selectbox("🤝 Hoạt động ngoại khóa", ["Không", "Có"]),
    'Điểm_Danh_%': st.sidebar.slider("📍 Tần suất lên lớp (%)", 0, 100, 95),
    'Hình_Thức_Học': st.sidebar.selectbox("📖 Phương pháp học", ["Tự học", "Học nhóm"]),
    'Thời_Gian_Mạng_Xã_Hội': st.sidebar.slider("📱 Mạng xã hội/Giải trí", 0, 15, 2)
}

st.sidebar.markdown("---")
btn_predict = st.sidebar.button("⚡ EXECUTE ANALYSIS", use_container_width=True)

# 4. GIAO DIỆN CHÍNH
st.markdown("<h1 class='robot-title'>🤖 AI STUDENT PERFORMANCE ANALYZER</h1>", unsafe_allow_html=True)
st.write("Dữ liệu được xử lý bởi nhân trung tâm Robot v2.0 - Dự đoán chính xác dựa trên Machine Learning.")

if btn_predict:
    # --- XỬ LÝ INPUT (ĐẢM BẢO KHỚP CỘT) ---
    input_df = pd.DataFrame([input_data])
    input_df['Làm_Thêm'] = input_df['Làm_Thêm'].map({'Có': 1, 'Không': 0})
    input_df['Tham_Gia_CLB'] = input_df['Tham_Gia_CLB'].map({'Có': 1, 'Không': 0})
    input_df['Hình_Thức_Học'] = input_df['Hình_Thức_Học'].map({'Tự học': 0, 'Học nhóm': 1})
    
    # Sắp xếp đúng thứ tự cột để tránh lỗi ValueError
    input_df = input_df[feature_cols]

    # --- DỰ ĐOÁN ---
    prediction = model.predict(input_df)[0]
    
    # --- HIỂN THỊ KẾT QUẢ ---
    col1, col2 = st.columns([1, 1.5], gap="large")
    
    with col1:
        st.markdown(f"""
            <div class="gpa-container">
                <p style="color: #00f2ff; font-size: 1.2rem; margin-bottom:0;">GPA DỰ TOÁN</p>
                <h1 style="color: #ffffff; font-size: 5rem; margin: 10px 0;">{prediction:.2f}</h1>
                <p style="color: #00ff80;">HỆ THỐNG: ỔN ĐỊNH</p>
            </div>
        """, unsafe_allow_html=True)
        
        # Thanh trạng thái
        st.progress(prediction / 4.0)
        if prediction >= 3.2: st.success("🎯 Rank: S (Xuất sắc)")
        elif prediction >= 2.5: st.info("🎯 Rank: A (Khá)")
        else: st.warning("🎯 Rank: C (Cần nâng cấp)")

    with col2:
        st.markdown("### 🛠️ LỜI KHUYÊN TỪ AI CORE:")
        
        # Logic phân tích thông minh
        advices = []
        if input_data['Điểm_Danh_%'] < 90:
            advices.append("❌ **Cảnh báo Lên lớp:** Tỷ lệ điểm danh thấp là rủi ro lớn nhất. Robot đề xuất bạn tăng tỷ lệ này lên >90% để tối ưu điểm quá trình.")
        if input_data['Số_Giờ_Học_Tuần'] < 15:
            advices.append("⚠️ **Tối ưu Học tập:** Thời gian tự học hiện tại quá thấp. Hãy nâng lên 20 giờ/tuần để thấy sự khác biệt.")
        if input_data['Thời_Gian_Ngủ'] < 6:
            advices.append("🔋 **Năng lượng thấp:** Bạn đang 'overclock' cơ thể. Ngủ dưới 6 tiếng làm giảm 30% hiệu suất xử lý thông tin.")
        if input_data['Thời_Gian_Mạng_Xã_Hội'] > 4:
            advices.append("🚫 **Nhiễu tín hiệu:** Thời gian dùng MXH đang lấn át thời gian học. Hãy giảm xuống dưới 2h/ngày.")
        
        if not advices:
            advices.append("✅ **Trạng thái Hoàn hảo:** Mọi chỉ số đều đang ở mức tối ưu. Hãy duy trì thuật toán sinh hoạt này!")

        # Hiển thị lời khuyên
        advice_html = "".join([f"<p style='margin-bottom:10px;'>{a}</p>" for a in advices])
        st.markdown(f"""
            <div class="ai-advice-box">
                {advice_html}
            </div>
        """, unsafe_allow_html=True)

    # --- BIỂU ĐỒ CÔNG NGHỆ ---
    st.markdown("---")
    t1, t2 = st.tabs(["📉 XU HƯỚNG DỮ LIỆU", "🧬 PHÂN TÍCH GEN HỌC TẬP"])
    
    with t1:
        fig = px.scatter(original_df, x="Số_Giờ_Học_Tuần", y="GPA", color="GPA",
                         title="Bản đồ mật độ: Giờ học vs GPA", template="plotly_dark")
        fig.update_traces(marker=dict(size=12, line=dict(width=2, color='DarkSlateGrey')))
        st.plotly_chart(fig, use_container_width=True)

    with t2:
        # Radar chart phong cách Robot
        categories = ['Số_Giờ_Học_Tuần', 'Thời_Gian_Ngủ', 'Điểm_Danh_%', 'Thời_Gian_Mạng_Xã_Hội', 'Số_Môn_Đang_Học']
        avg_vals = original_df[categories].mean()
        user_vals = [input_data[c] for c in categories]
        
        # Chuẩn hóa để vẽ radar đẹp
        max_vals = [60, 12, 100, 10, 12]
        user_norm = [u/m for u, m in zip(user_vals, max_vals)]
        avg_norm = [a/m for a, m in zip(avg_vals, max_vals)]

        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(r=user_norm, theta=categories, fill='toself', name='Bạn (You)', line_color='#00f2ff'))
        fig_radar.add_trace(go.Scatterpolar(r=avg_norm, theta=categories, fill='toself', name='Trung bình (Global)', line_color='#ff00ff'))
        fig_radar.update_layout(polar=dict(radialaxis=dict(visible=False)), template="plotly_dark")
        st.plotly_chart(fig_radar, use_container_width=True)

else:
    # Màn hình chờ Robot
    col_intro1, col_intro2 = st.columns([1, 1])
    with col_intro1:
        st.markdown("""
            ### 🤖 HỆ THỐNG ĐANG TRỰC
            Vui lòng nhập các tham số sinh học và học thuật của bạn vào bảng điều khiển bên trái. 
            Nhấn **EXECUTE ANALYSIS** để AI Core bắt đầu quét dữ liệu và dự đoán tương lai học tập của bạn.
        """)
    with col_intro2:
        st.image("https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExNHJueXF6Znd6Znd6Znd6Znd6Znd6Znd6Znd6Znd6Znd6Znd6Znd6JmVwPXYxX2ludGVybmFsX2dpZl9ieV9pZCZjdD1n/3o7TKMGpxVfclE56mI/giphy.gif")
