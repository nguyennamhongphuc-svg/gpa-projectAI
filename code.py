import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
import os

# 1. CẤU HÌNH TRANG & GIAO DIỆN CHUYÊN NGHIỆP
st.set_page_config(
    page_title="AI GPA Insights | Hệ Thống Dự Đoán & Phân Tích Kỷ Số Học Tập",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Tùy chỉnh giao diện bằng CSS nâng cao (Modern Glassmorphism & Clean UI)
st.markdown("""
    <style>
    /* Nền và font chữ */
    .main { background-color: #f8fafc; color: #1e293b; }
    
    /* Thiết kế thẻ Card cho các khối thông tin */
    .metric-card {
        background-color: #ffffff;
        padding: 24px;
        border-radius: 16px;
        box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.05), 0 2px 4px -2px rgb(0 0 0 / 0.05);
        border: 1px solid #e2e8f0;
        margin-bottom: 20px;
    }
    
    /* Tùy chỉnh tiêu đề */
    .main-title {
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(90deg, #1e3a8a, #3b82f6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .subtitle { color: #64748b; font-size: 1.1rem; margin-bottom: 2rem; }
    
    /* Khối kết quả GPA nổi bật */
    .gpa-box {
        background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
        color: white;
        padding: 30px;
        border-radius: 16px;
        text-align: center;
        box-shadow: 0 10px 15px -3px rgba(59, 130, 246, 0.3);
    }
    .gpa-value { font-size: 3.5rem; font-weight: 800; line-height: 1; margin: 10px 0; }
    
    /* Thẻ lời khuyên từ AI */
    .ai-box {
        background-color: #f0fdf4;
        border-left: 5px solid #22c55e;
        padding: 20px;
        border-radius: 4px 16px 16px 4px;
    }
    .ai-box-warning {
        background-color: #fef2f2;
        border-left: 5px solid #ef4444;
        padding: 20px;
        border-radius: 4px 16px 16px 4px;
    }
    </style>
    """, unsafe_allow_html=True)

# 2. HÀM XỬ LÝ DỮ LIỆU & HUẤN LUYỆN MODEL
@st.cache_resource
def train_model(file_path):
    if not os.path.exists(file_path):
        return None, None, None
    
    df = pd.read_csv(file_path)
    df_ml = df.copy()
    df_ml['Làm_Thêm'] = df_ml['Làm_Thêm'].map({'Có': 1, 'Không': 0})
    df_ml['Tham_Gia_CLB'] = df_ml['Tham_Gia_CLB'].map({'Có': 1, 'Không': 0})
    df_ml['Hình_Thức_Học'] = df_ml['Hình_Thức_Học'].map({'Tự học': 0, 'Học nhóm': 1})
    
    X = df_ml.drop('GPA', axis=1)
    y = df_ml['GPA']
    
    model = RandomForestRegressor(n_estimators=150, random_state=42) # Tăng estimators để ổn định hơn
    model.fit(X, y)
    
    return model, df, X.columns.tolist()

model, original_df, feature_cols = train_model('Dataset_GPA_Thuc_Te_250_Responses.csv')

if model is None:
    st.error("❌ Không tìm thấy file 'Dataset_GPA_Thuc_Te_250_Responses.csv'. Vui lòng kiểm tra lại đường dẫn.")
    st.stop()

# 3. SIDEBAR NHẬP LIỆU (Thiết kế gọn gàng, trực quan)
st.sidebar.markdown("### 🛠️ Cấu Hình Chỉ Số Cá Nhân")
st.sidebar.markdown("Thay đổi các thông số dưới đây để AI phân tích mô hình học tập của bạn.")

input_data = {}
input_data['Số_Giờ_Học_Tuần'] = st.sidebar.slider("📚 Số giờ tự học / tuần", 0, 60, 20, help="Tổng số giờ bạn tự học ngoài giờ lên lớp")
input_data['Điểm_Danh_%'] = st.sidebar.slider("📍 Tỷ lệ điểm danh lớp (%)", 0, 100, 90)
input_data['Thời_Gian_Ngủ'] = st.sidebar.slider("😴 Thời gian ngủ (giờ/ngày)", 3, 12, 7)
input_data['Thời_Gian_Mạng_Xã_Hội'] = st.sidebar.slider("📱 Dùng MXH & Giải trí (giờ/ngày)", 0, 10, 2)
input_data['Số_Môn_Đang_Học'] = st.sidebar.number_input("📝 Số môn học học kỳ này", 1, 12, 5)

st.sidebar.markdown("---")
input_data['Làm_Thêm'] = st.sidebar.selectbox("💼 Công việc làm thêm?", ["Không", "Có"])
input_data['Tham_Gia_CLB'] = st.sidebar.selectbox("🤝 Hoạt động ngoại khóa / CLB?", ["Không", "Có"])
input_data['Hình_Thức_Học'] = st.sidebar.selectbox("📖 Phương pháp học chủ đạo", ["Tự học", "Học nhóm"])

st.sidebar.markdown("<br>", unsafe_allow_html=True)
btn_predict = st.sidebar.button("🚀 PHÂN TÍCH & DỰ ĐOÁN", type="primary", use_container_width=True)

# 4. GIAO DIỆN CHÍNH (MAIN DASHBOARD)
st.markdown('<p class="main-title">🎓 Hệ Thống AI Dự Đoán & Tối Ưu GPA</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Nền tảng phân tích học thuật dựa trên dữ liệu thực tế của 250 sinh viên</p>', unsafe_allow_html=True)

# Khởi tạo trạng thái session state để giữ kết quả hiển thị mượt mà
if "predicted" not in st.session_state:
    st.session_state.predicted = False

if btn_predict:
    st.session_state.predicted = True

# Phân luồng hiển thị giao diện
if st.session_state.predicted:
    # --- XỬ LÝ DỮ LIỆU INPUT ---
    input_df = pd.DataFrame([input_data])
    input_df['Làm_Thêm'] = input_df['Làm_Thêm'].map({'Có': 1, 'Không': 0})
    input_df['Tham_Gia_CLB'] = input_df['Tham_Gia_CLB'].map({'Có': 1, 'Không': 0})
    input_df['Hình_Thức_Học'] = input_df['Hình_Thức_Học'].map({'Tự học': 0, 'Học nhóm': 1})
    
    # Thực hiện dự đoán bằng Machine Learning
    prediction = model.predict(input_df)[0]
    prediction = min(4.00, max(0.00, prediction)) # Giới hạn dải điểm hệ 4

    # BỐ CỤC KẾT QUẢ DỰ ĐOÁN LÊN ĐẦU TRANG
    col_res1, col_res2 = st.columns([1, 2], gap="large")
    
    with col_res1:
        # Thẻ điểm số nổi bật phong cách Dashboard
        st.markdown(f"""
        <div class="gpa-box">
            <span style="font-size: 1.1rem; font-weight: 500; opacity: 0.9;">KẾT QUẢ DỰ ĐOÁN GPA</span>
            <div class="gpa-value">{prediction:.2f}</div>
            <span style="font-size: 1rem; opacity: 0.8;">Thang điểm hệ 4.0</span>
        </div>
        """, unsafe_allow_html=True)
        
        # Đánh giá xếp loại nhanh
        st.markdown("<div style='margin-top: 12px;'></div>", unsafe_allow_html=True)
        if prediction >= 3.6:
            st.success("🏆 Xếp loại dự kiến: **Xuất Sắc**")
            st.balloons()
        elif prediction >= 3.2:
            st.success("🌟 Xếp loại dự kiến: **Giỏi**")
        elif prediction >= 2.5:
            st.info("✅ Xếp loại dự kiến: **Khá**")
        else:
            st.warning("⚠️ Xếp loại dự kiến: **Trung Bình / Cần Cải Thiện**")
            
    with col_res2:
        # Lời khuyên động từ AI sinh ra dựa trên Input chi tiết của người dùng
        st.markdown("### 🧠 AI Học Thuật Đánh Giá & Gợi Ý:")
        
        # Thuật toán sinh nhận xét động dựa trên dữ liệu đầu vào
        ai_advices = []
        is_warning = False
        
        if input_data['Điểm_Danh_%'] < 80:
            ai_advices.append("🔴 **Tỷ lệ đi học quá thấp:** Việc vắng mặt trên 20% số tiết đang kéo mạnh điểm số của bạn xuống. Hãy ưu tiên việc lên lớp học trực tiếp.")
            is_warning = True
        if input_data['Số_Giờ_Học_Tuần'] < 12:
            ai_advices.append("📚 **Thiếu thời gian tự học:** Bạn đang dành ít hơn 12h/tuần để tự nghiên cứu. Hãy nâng dần mục tiêu lên thêm 2 tiếng mỗi tuần.")
        if input_data['Thời_Gian_Mạng_Xã_Hội'] >= 5:
            ai_advices.append("📱 **Quá tải MXH:** Sử dụng mạng xã hội ≥ 5 tiếng/ngày đang lấn chiếm nghiêm trọng quỹ thời gian phục hồi của não bộ.")
        if input_data['Thời_Gian_Ngủ'] < 6:
            ai_advices.append("😴 **Thiếu ngủ nghiêm trọng:** Ngủ dưới 6 tiếng sẽ làm giảm khả năng ghi nhớ dài hạn trong mùa thi cử.")
            
        if not ai_advices: # Nếu các chỉ số đều rất tốt
            ai_advices.append("🎉 **Mô hình cân bằng tuyệt vời!** Các chỉ số sinh hoạt và học tập của bạn đang đạt trạng thái tối ưu lý tưởng. Hãy tiếp tục duy trì kỷ luật này để đạt học bổng.")

        # Chọn màu sắc hiển thị phù hợp trạng thái
        box_class = "ai-box-warning" if is_warning or prediction < 2.5 else "ai-box"
        advice_html = "".join([f"<li style='margin-bottom:8px;'>{adv}</li>" for adv in ai_advices])
        
        st.markdown(f"""
        <div class="{box_class}">
            <h4 style='margin-top:0; color:inherit;'>💡 Chiến lược tối ưu dành riêng cho bạn:</h4>
            <ul style='margin-bottom:0; padding-left:20px;'>
                {advice_html}
            </ul>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    
    # PHẦN 5. CÁC TABS PHÂN TÍCH ĐỒ THỊ CHUYÊN SÂU
    tab_radar, tab_eda = st.tabs(["📊 So Sánh Định Lượng (Radar)", "📈 Xu Hướng & Tầm Quan Trọng"])
    
    with tab_radar:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.subheader("Bản đồ định vị năng lực (So với trung bình 250 Sinh viên)")
        
        avg_values = original_df.mean(numeric_only=True)
        categories = ['Số_Giờ_Học_Tuần', 'Thời_Gian_Ngủ', 'Điểm_Danh_%', 'Thời_Gian_Mạng_Xã_Hội', 'Số_Môn_Đang_Học']
        
        # Chuẩn hóa dữ liệu chia cho max để biểu đồ Radar cân đối, đẹp mắt
        max_vals = [60, 12, 100, 10, 12] 
        user_vals_norm = [input_data[c]/m for c, m in zip(categories, max_vals)]
        avg_vals_norm = [avg_values[c]/m for c, m in zip(categories, max_vals)]
        
        display_categories = ['Giờ học/Tuần', 'Thời gian ngủ', 'Tỷ lệ Điểm danh', 'Thời gian MXH', 'Số môn học']
        
        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=user_vals_norm, theta=display_categories, 
            fill='toself', name='Chỉ số của bạn',
            fillcolor='rgba(59, 130, 246, 0.2)', line=dict(color='#3b82f6', width=2)
        ))
        fig_radar.add_trace(go.Scatterpolar(
            r=avg_vals_norm, theta=display_categories, 
            fill='toself', name='Trung bình mẫu sinh viên',
            fillcolor='rgba(148, 163, 184, 0.2)', line=dict(color='#94a3b8', width=2, dash='dash')
        ))
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1], showticklabels=False)),
            showlegend=True,
            margin=dict(t=20, b=20, l=20, r=20),
            height=380
        )
        st.plotly_chart(fig_radar, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with tab_eda:
        col_eda1, col_eda2 = st.columns(2, gap="medium")
        
        with col_eda1:
            st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
            fig_scatter = px.scatter(
                original_df, x="Số_Giờ_Học_Tuần", y="GPA", 
                title="<b>Mật độ tương quan: Thời gian học vs GPA thực tế</b>",
                trendline="ols", 
                color_discrete_sequence=['#3b82f6'],
                labels={"Số_Giờ_Học_Tuần": "Số giờ tự học/tuần", "GPA": "Điểm số GPA"}
            )
            fig_scatter.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", height=320)
            st.plotly_chart(fig_scatter, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
            
        with col_eda2:
            st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
            importances = model.feature_importances_
            feat_importances = pd.Series(importances, index=feature_cols).sort_values(ascending=True)
            
            # Đổi tên nhãn hiển thị sang tiếng Việt thân thiện trên biểu đồ
            friendly_labels = {
                'Số_Giờ_Học_Tuần': 'Số giờ học/tuần', 'Điểm_Danh_%': 'Tỷ lệ điểm danh',
                'Thời_Gian_Ngủ': 'Thời gian ngủ', 'Thời_Gian_Mạng_Xã_Hội': 'Thời gian dùng MXH',
                'Số_Môn_Đang_Học': 'Số môn đang học', 'Làm_Thêm': 'Đi làm thêm',
                'Tham_Gia_CLB': 'Tham gia CLB', 'Hình_Thức_Học': 'Hình thức học'
            }
            feat_importances.index = [friendly_labels.get(x, x) for x in feat_importances.index]

            fig_importance = px.bar(
                feat_importances, orientation='h',
                title="<b>Mức độ tác động lên GPA (Trọng số AI phân tích)</b>",
                color=feat_importances.values,
                color_continuous_scale="Blues"
            )
            fig_importance.update_layout(
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", 
                showlegend=False, coloraxis_showscale=False, height=320,
                margin=dict(l=10, r=10, t=40, b=10)
            )
            st.plotly_chart(fig_importance, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

else:
    # HIỂN THỊ BAN ĐẦU KHI CHƯA BẤM NÚT (Trang giới thiệu chuyên nghiệp)
    col_intro1, col_intro2 = st.columns([4, 3], gap="large")
    with col_intro1:
        st.markdown("""
        ### Chào mừng bạn đến với Hệ thống AI Phân Tích Học Thuật
        Ứng dụng sử dụng thuật toán Học máy **Random Forest Regressor** giúp bạn định vị bản thân và dự báo chính xác điểm số GPA mục tiêu dựa trên dữ liệu thói quen sinh hoạt.
        
        #### Đang chờ dữ liệu đầu vào...
        👉 **Hãy thực hiện các bước sau để xem kết quả:**
        1. Sử dụng thanh công cụ bên **bên trái (Sidebar)** để thiết lập các thông số thực tế của bạn.
        2. Nhấn nút màu xanh **"🚀 PHÂN TÍCH & DỰ ĐOÁN"** ở góc dưới thanh Sidebar.
        3. Hệ thống sẽ ngay lập tức tính toán và đưa ra biểu đồ báo cáo chi tiết.
        """)
    with col_intro2:
        st.image("https://img.freepik.com/free-vector/data-extraction-concept-illustration_114360-4766.jpg", use_container_width=True)
