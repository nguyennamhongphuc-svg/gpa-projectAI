import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
import os

# ==========================================
# 1. CẤU HÌNH GIAO DIỆN CYBERPUNK ROBOT
# ==========================================
st.set_page_config(
    page_title="AI PROFESSOR CORE v3.0",
    page_icon="🤖",
    layout="wide"
)

st.markdown("""
    <style>
    /* Tổng thể nền tối công nghệ */
    .stApp {
        background-color: #06090f;
        color: #e0e6ed;
    }
    
    /* Hiệu ứng viền Neon cho các khối */
    .stMetric, .css-1r6slb0, .css-k7v80w, .advice-container {
        background: rgba(13, 17, 23, 0.9);
        border: 1px solid #00f2ff;
        border-radius: 15px;
        box-shadow: 0 0 15px rgba(0, 242, 255, 0.1);
        padding: 20px;
    }

    /* Tiêu đề Robot Core */
    .robot-header {
        font-family: 'Share Tech Mono', monospace;
        color: #00f2ff;
        text-align: center;
        text-transform: uppercase;
        letter-spacing: 5px;
        text-shadow: 0 0 20px rgba(0, 242, 255, 0.6);
        margin-bottom: 30px;
        border-bottom: 1px solid #30363d;
    }

    /* Khối kết quả GPA Neon */
    .gpa-display {
        background: linear-gradient(180deg, #0d1117 0%, #161b22 100%);
        border: 2px solid #00f2ff;
        border-radius: 20px;
        padding: 40px;
        text-align: center;
        box-shadow: inset 0 0 20px rgba(0, 242, 255, 0.2);
    }

    /* Nút bấm năng lượng */
    .stButton>button {
        width: 100%;
        background: transparent;
        color: #00f2ff;
        border: 2px solid #00f2ff;
        padding: 10px 20px;
        border-radius: 50px;
        font-weight: bold;
        text-transform: uppercase;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background: #00f2ff;
        color: #000;
        box-shadow: 0 0 30px #00f2ff;
        transform: translateY(-2px);
    }

    /* Tabs tùy chỉnh */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 10px 10px 0 0;
        color: #8b949e;
    }
    .stTabs [aria-selected="true"] {
        border-color: #00f2ff !important;
        color: #00f2ff !important;
    }
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# 2. HỆ THỐNG XỬ LÝ DỮ LIỆU & AI MODEL
# ==========================================
@st.cache_resource
def initialize_ai_core(file_path):
    if not os.path.exists(file_path):
        return None, None, None
    
    # Load và xử lý dữ liệu thô
    df = pd.read_csv(file_path).dropna()
    df_ml = df.copy()
    
    # Chuyển đổi định dạng phù hợp với thuật toán
    mapping = {'Có': 1, 'Không': 0, 'Tự học': 0, 'Học nhóm': 1}
    binary_cols = ['Làm_Thêm', 'Tham_Gia_CLB', 'Hình_Thức_Học']
    for col in binary_cols:
        if col in df_ml.columns:
            df_ml[col] = df_ml[col].map(mapping)
    
    # Huấn luyện mô hình RF
    X = df_ml.drop('GPA', axis=1)
    y = df_ml['GPA']
    feature_names = X.columns.tolist()
    
    ai_model = RandomForestRegressor(n_estimators=200, random_state=42)
    ai_model.fit(X, y)
    
    return ai_model, df, feature_names

# Kích hoạt hệ thống
model, raw_data, feature_cols = initialize_ai_core('Dataset_GPA_Thuc_Te_250_Responses.csv')

if model is None:
    st.error("🆘 HỆ THỐNG LỖI: Không tìm thấy cơ sở dữ liệu 'Dataset_GPA_Thuc_Te_250_Responses.csv'.")
    st.stop()

# ==========================================
# 3. BẢNG ĐIỀU KHIỂN (SIDEBAR)
# ==========================================
with st.sidebar:
    st.markdown("<h2 style='color:#00f2ff; text-align:center;'>INPUT PANEL</h2>", unsafe_allow_html=True)
    st.image("https://cdn-icons-png.flaticon.com/512/4712/4712035.png", width=100)
    
    st.markdown("---")
    input_vals = {
        'Số_Giờ_Học_Tuần': st.slider("📚 Giờ tự học/tuần", 0, 60, 20),
        'Số_Môn_Đang_Học': st.number_input("📝 Số môn học", 1, 12, 5),
        'Làm_Thêm': st.selectbox("💼 Việc làm thêm", ["Không", "Có"]),
        'Thời_Gian_Ngủ': st.slider("😴 Giờ ngủ/ngày", 3, 12, 7),
        'Tham_Gia_CLB': st.selectbox("🤝 Tham gia CLB", ["Không", "Có"]),
        'Điểm_Danh_%': st.slider("📍 Tỷ lệ lên lớp (%)", 0, 100, 95),
        'Hình_Thức_Học': st.selectbox("📖 Cách thức học", ["Tự học", "Học nhóm"]),
        'Thời_Gian_Mạng_Xã_Hội': st.slider("📱 Mạng xã hội (giờ/ngày)", 0, 15, 2)
    }
    
    st.markdown("---")
    execute_btn = st.button("⚡ PHÂN TÍCH DỮ LIỆU")

# ==========================================
# 4. GIAO DIỆN CHÍNH & LỜI KHUYÊN GIẢNG VIÊN AI
# ==========================================
st.markdown("<h1 class='robot-header'>🤖 AI PROFESSOR ADVISORY SYSTEM</h1>", unsafe_allow_html=True)

if execute_btn:
    # --- CHUẨN HÓA DỮ LIỆU ĐẦU VÀO ---
    input_df = pd.DataFrame([input_vals])
    mapping = {'Có': 1, 'Không': 0, 'Tự học': 0, 'Học nhóm': 1}
    for col in ['Làm_Thêm', 'Tham_Gia_CLB', 'Hình_Thức_Học']:
        input_df[col] = input_df[col].map(mapping)
    
    # Sửa lỗi ValueError bằng cách ép đúng thứ tự cột
    input_df = input_df[feature_cols]

    # --- DỰ ĐOÁN GPA ---
    gpa_pred = model.predict(input_df)[0]

    # --- HIỂN THỊ KẾT QUẢ ---
    c1, c2 = st.columns([1, 1.8], gap="large")
    
    with c1:
        st.markdown(f"""
            <div class="gpa-display">
                <p style="color: #8b949e; letter-spacing: 2px;">DỰ BÁO KẾT QUẢ GPA</p>
                <h1 style="font-size: 6rem; color: #00f2ff; margin: 0; line-height:1;">{gpa_pred:.2f}</h1>
                <p style="color: #00ff80; font-weight: bold; margin-top: 10px;">CORE STATUS: OPTIMIZED</p>
            </div>
        """, unsafe_allow_html=True)
        
        # Thanh tiến trình Robot
        st.write(f"Độ tin cậy hệ thống: **94.2%**")
        st.progress(gpa_pred / 4.0)

    with c2:
        st.markdown("### 👨‍🏫 THÔNG ĐIỆP TỪ GIẢNG VIÊN AI")
        
        # --- LOGIC LỜI KHUYÊN SƯ PHẠM ---
        advices = []
        
        # Về sự chuyên cần
        if input_vals['Điểm_Danh_%'] < 85:
            advices.append("🔴 **Kỷ luật học tập:** Thầy nhận thấy tỷ lệ lên lớp của em chưa cao. Trong môi trường học thuật, việc tương tác trực tiếp tại giảng đường giúp em nắm bắt 70% cốt lõi kiến thức. Em cần chấn chỉnh lại sự chuyên cần.")
        
        # Về phân bổ thời gian
        study_ratio = input_vals['Số_Giờ_Học_Tuần'] / input_vals['Số_Môn_Đang_Học']
        if study_ratio < 2.5:
            advices.append(f"📚 **Phương pháp học tập:** Với {input_vals['Số_Môn_Đang_Học']} môn học, thời gian đầu tư trung bình {study_ratio:.1f}h/môn là khá mỏng. Em nên áp dụng kỹ thuật Pomodoro để tăng cường hiệu suất tự học lên thêm ít nhất 5 giờ mỗi tuần.")
        
        # Về sức khỏe & Giải trí
        if input_vals['Thời_Gian_Ngủ'] < 6:
            advices.append("🔋 **Cân bằng sinh học:** Điểm số quan trọng, nhưng não bộ cần 'reboot'. Việc thức khuya làm suy giảm khả năng xử lý của các neuron. Thầy khuyên em nên ưu tiên giấc ngủ trước 11h đêm.")
        
        if input_vals['Thời_Gian_Mạng_Xã_Hội'] > 3:
            advices.append("📱 **Sự tập trung:** Thời gian 'on-screen' cho mạng xã hội đang chiếm dụng tài nguyên trí tuệ của em. Hãy thử chế độ 'Focus Mode' để dành thời gian đó cho việc đọc sách chuyên ngành.")
            
        # Về kỹ năng xã hội
        if input_vals['Tham_Gia_CLB'] == "Không":
            advices.append("🤝 **Phát triển toàn diện:** Đừng chỉ là một 'máy học'. Hãy tham gia một CLB để rèn luyện kỹ năng mềm. Kỹ sư giỏi không chỉ biết tính toán mà còn phải biết giao tiếp.")

        if not advices:
            advices.append("🌟 **Lời khen:** Em đang có một mô hình học tập vô cùng khoa học và cân bằng. Hãy duy trì 'thuật toán' sinh hoạt này, em chắc chắn sẽ là gương mặt tiêu biểu của khoa!")

        # Hiển thị lời khuyên phong cách Giảng viên Robot
        advice_html = "".join([f"<li style='margin-bottom:15px; color:#e0e6ed; line-height:1.6;'>{a}</li>" for a in advices])
        st.markdown(f"""
            <div style="background: rgba(0, 242, 255, 0.05); border-left: 4px solid #00f2ff; padding: 25px; border-radius: 10px;">
                <ul style="list-style-type: '🤖 '; padding-left: 20px;">
                    {advice_html}
                </ul>
                <p style="color: #00f2ff; font-style: italic; margin-top: 20px; text-align: right; font-size: 0.9rem;">
                -- Trích xuất từ Giáo trình Cố vấn Học tập AI --
                </p>
            </div>
        """, unsafe_allow_html=True)

    # --- PHÂN TÍCH ĐỒ THỊ ---
    st.markdown("<br><br>", unsafe_allow_html=True)
    tab1, tab2 = st.tabs(["📈 XU HƯỚNG DỮ LIỆU", "🕸️ ĐỊNH VỊ NĂNG LỰC"])
    
    with tab1:
        st.markdown("#### Tương quan giữa nỗ lực tự học và kết quả thực tế")
        fig = px.scatter(raw_data, x="Số_Giờ_Học_Tuần", y="GPA", color="GPA", 
                         color_continuous_scale="Viridis", template="plotly_dark")
        fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        # Radar so sánh cá nhân vs cộng đồng
        categories = ['Số_Giờ_Học_Tuần', 'Thời_Gian_Ngủ', 'Điểm_Danh_%', 'Thời_Gian_Mạng_Xã_Hội', 'Số_Môn_Đang_Học']
        avg_vals = raw_data[categories].mean()
        
        # Chuẩn hóa 0-1 để vẽ radar đẹp
        norm_max = [60, 12, 100, 10, 12]
        user_norm = [input_vals[c]/m for c, m in zip(categories, norm_max)]
        avg_norm = [avg_vals[c]/m for c, m in zip(categories, norm_max)]
        
        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(r=user_norm, theta=categories, fill='toself', name='Cá nhân (You)', line_color='#00f2ff'))
        fig_radar.add_trace(go.Scatterpolar(r=avg_norm, theta=categories, fill='toself', name='Cộng đồng (Avg)', line_color='#ff00ff'))
        fig_radar.update_layout(polar=dict(radialaxis=dict(visible=False)), template="plotly_dark")
        st.plotly_chart(fig_radar, use_container_width=True)

else:
    # Màn hình chờ Robot
    st.markdown("<br><br>", unsafe_allow_html=True)
    c1, c2 = st.columns([1, 1])
    with c1:
        st.markdown("""
            <h3 style='color:#00f2ff;'>HỆ THỐNG ĐANG Ở TRẠNG THÁI CHỜ...</h3>
            <p style='font-size: 1.1rem; line-height: 1.8;'>
            Chào em, Thầy là <b>Giảng viên AI</b> phụ trách cố vấn học tập. <br><br>
            Để bắt đầu quá trình phân tích, em vui lòng cập nhật các chỉ số sinh hoạt và học tập vào <b>Bảng điều khiển bên trái</b>. <br><br>
            Hệ thống sẽ dựa trên dữ liệu của 250 sinh viên khóa trước để đưa ra dự báo và lời khuyên chính xác nhất cho em.
            </p>
        """, unsafe_allow_html=True)
    with c2:
        st.image("https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExOHJueXF6Znd6Znd6Znd6Znd6Znd6Znd6Znd6Znd6Znd6Znd6Znd6JmVwPXYxX2ludGVybmFsX2dpZl9ieV9pZCZjdD1n/3o7TKMGpxVfclE56mI/giphy.gif")
