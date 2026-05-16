import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
import os

# 1. Cấu hình trang
st.set_page_config(
    page_title="Dự Đoán GPA Sinh Viên",
    page_icon="🎓",
    layout="wide"
)

# Tùy chỉnh CSS để giao diện đẹp hơn
st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .stMetric {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    </style>
    """, unsafe_allow_html=True)

# 2. Hàm xử lý dữ liệu và huấn luyện
@st.cache_resource
def train_model(file_path):
    if not os.path.exists(file_path):
        return None, None, None
    
    df = pd.read_csv(file_path)
    
    # Tiền xử lý
    df_ml = df.copy()
    df_ml['Làm_Thêm'] = df_ml['Làm_Thêm'].map({'Có': 1, 'Không': 0})
    df_ml['Tham_Gia_CLB'] = df_ml['Tham_Gia_CLB'].map({'Có': 1, 'Không': 0})
    df_ml['Hình_Thức_Học'] = df_ml['Hình_Thức_Học'].map({'Tự học': 0, 'Học nhóm': 1})
    
    X = df_ml.drop('GPA', axis=1)
    y = df_ml['GPA']
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    return model, df, X.columns.tolist()

# Tải dữ liệu và mô hình
model, original_df, feature_cols = train_model('Dataset_GPA_Thuc_Te_250_Responses.csv')

if model is None:
    st.error("❌ Không tìm thấy file 'Dataset_GPA_Thuc_Te_250_Responses.csv'. Vui lòng kiểm tra lại nguồn dữ liệu.")
    st.stop()

# 3. Sidebar - Nhập liệu
st.sidebar.header("📊 Thông Số Cá Nhân")
st.sidebar.markdown("Nhập thông tin của bạn để dự đoán")

input_data = {}
input_data['Số_Giờ_Học_Tuần'] = st.sidebar.slider("📚 Số giờ học/tuần", 0, 60, 20)
input_data['Số_Môn_Đang_Học'] = st.sidebar.number_input("📝 Số môn đang học", 1, 12, 5)
input_data['Làm_Thêm'] = st.sidebar.selectbox("💼 Có đi làm thêm không?", ["Không", "Có"])
input_data['Thời_Gian_Ngủ'] = st.sidebar.slider("😴 Thời gian ngủ (giờ/ngày)", 3, 12, 7)
input_data['Tham_Gia_CLB'] = st.sidebar.selectbox("🤝 Tham gia CLB?", ["Không", "Có"])
input_data['Điểm_Danh_%'] = st.sidebar.slider("📍 Tỷ lệ điểm danh (%)", 0, 100, 90)
input_data['Hình_Thức_Học'] = st.sidebar.selectbox("📖 Hình thức học chính", ["Tự học", "Học nhóm"])
input_data['Thời_Gian_Mạng_Xã_Hội'] = st.sidebar.slider("📱 TG dùng MXH (giờ/ngày)", 0, 10, 2)

# Chuyển đổi input sang dạng số cho model
input_df = pd.DataFrame([input_data])
input_df['Làm_Thêm'] = input_df['Làm_Thêm'].map({'Có': 1, 'Không': 0})
input_df['Tham_Gia_CLB'] = input_df['Tham_Gia_CLB'].map({'Có': 1, 'Không': 0})
input_df['Hình_Thức_Học'] = input_df['Hình_Thức_Học'].map({'Tự học': 0, 'Học nhóm': 1})

# 4. Giao diện chính
st.title("🎓 Hệ Thống Dự Đoán & Phân Tích GPA Sinh Viên")
st.markdown("---")

tab1, tab2, tab3 = st.tabs(["🎯 Dự Đoán Kết Quả", "📈 Phân Tích Dữ Liệu (EDA)", "⚖️ So Sánh Chỉ Số"])

with tab1:
    st.subheader("Kết quả dự đoán của bạn")
    prediction = model.predict(input_df)[0]
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.metric(label="GPA Dự Đoán", value=f"{prediction:.2f}/4.0")
        
        if prediction >= 3.2:
            st.success("🌟 Loại: Giỏi/Xuất sắc")
            st.balloons()
        elif prediction >= 2.5:
            st.info("✅ Loại: Khá")
        else:
            st.warning("⚠️ Loại: Trung bình/Cần cố gắng")
            
    with col2:
        st.markdown("### 💡 Lời khuyên chuyên gia:")
        if prediction < 2.5:
            st.write("Cảnh báo: Bạn nên cân đối lại thời gian học tập và tăng tỷ lệ điểm danh trên lớp để cải thiện kết quả.")
        elif prediction > 3.6:
            st.write("Tuyệt vời! Bạn đang duy trì phong độ rất tốt. Hãy tiếp tục phát huy và chia sẻ kinh nghiệm học tập nhé.")
        else:
            st.write("Kết quả của bạn ở mức ổn định. Hãy thử tối ưu hóa phương pháp học hoặc giảm bớt thời gian mạng xã hội để bứt phá lên mức Giỏi.")

with tab2:
    st.subheader("Phân tích các yếu tố ảnh hưởng")
    col_eda1, col_eda2 = st.columns(2)
    
    with col_eda1:
        fig_scatter = px.scatter(original_df, x="Số_Giờ_Học_Tuần", y="GPA", 
                                 title="Tương quan giữa Giờ học và GPA",
                                 trendline="ols", color_discrete_sequence=['#1f77b4'])
        st.plotly_chart(fig_scatter, use_container_width=True)
        
    with col_eda2:
        importances = model.feature_importances_
        feat_importances = pd.Series(importances, index=feature_cols).sort_values()
        fig_importance = px.bar(feat_importances, orientation='h', 
                                title="Các yếu tố quan trọng nhất (Feature Importance)",
                                labels={'value': 'Độ quan trọng', 'index': 'Tính năng'})
        st.plotly_chart(fig_importance, use_container_width=True)

with tab3:
    st.subheader("So sánh chỉ số cá nhân với trung bình hệ thống")
    
    # Tính toán trung bình
    avg_values = original_df.mean(numeric_only=True)
    
    # Các tiêu chí so sánh
    categories = ['Số_Giờ_Học_Tuần', 'Thời_Gian_Ngủ', 'Điểm_Danh_%', 'Thời_Gian_Mạng_Xã_Hội', 'Số_Môn_Đang_Học']
    
    user_vals = [input_data[c] for c in categories]
    avg_vals = [avg_values[c] for c in categories]
    
    fig_radar = go.Figure()
    
    fig_radar.add_trace(go.Scatterpolar(
        r=user_vals,
        theta=categories,
        fill='toself',
        name='Của Bạn'
    ))
    fig_radar.add_trace(go.Scatterpolar(
        r=avg_vals,
        theta=categories,
        fill='toself',
        name='Trung bình (250 SV)'
    ))
    
    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True)),
        showlegend=True,
        title="Biểu đồ Radar so sánh chỉ số (Giá trị thực tế)"
    )
    st.plotly_chart(fig_radar, use_container_width=True)

