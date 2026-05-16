import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
import os

# ==========================================
# 1. CẤU HÌNH TRANG & GIAO DIỆN
# ==========================================
st.set_page_config(
    page_title="AI GPA Predictor Pro",
    page_icon="🎓",
    layout="wide"
)

# Custom CSS cho giao diện hiện đại
st.markdown("""
    <style>
    .main { background-color: #f8fafc; }
    .stMetric {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        border: 1px solid #e2e8f0;
    }
    .predict-box {
        background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin-bottom: 2rem;
    }
    .advice-card {
        background-color: #ffffff;
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 5px solid #3b82f6;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# 2. HÀM XỬ LÝ DỮ LIỆU & HUẤN LUYỆN (SỬA LỖI TRANG)
# ==========================================
@st.cache_resource
def train_model(file_path):
    if not os.path.exists(file_path):
        return None, None, None
    
    df = pd.read_csv(file_path)
    # Làm sạch dữ liệu nếu có giá trị trống
    df = df.dropna()
    
    df_ml = df.copy()
    # Chuyển đổi dữ liệu text sang số
    df_ml['Làm_Thêm'] = df_ml['Làm_Thêm'].map({'Có': 1, 'Không': 0})
    df_ml['Tham_Gia_CLB'] = df_ml['Tham_Gia_CLB'].map({'Có': 1, 'Không': 0})
    df_ml['Hình_Thức_Học'] = df_ml['Hình_Thức_Học'].map({'Tự học': 0, 'Học nhóm': 1})
    
    # Tách X và y
    X = df_ml.drop('GPA', axis=1)
    y = df_ml['GPA']
    
    # Lưu danh sách cột chuẩn để đối chiếu lúc predict
    feature_names = X.columns.tolist()
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    return model, df, feature_names

# Tải model
model, original_df, feature_cols = train_model('Dataset_GPA_Thuc_Te_250_Responses.csv')

if model is None:
    st.error("❌ Không tìm thấy file dữ liệu 'Dataset_GPA_Thuc_Te_250_Responses.csv'. Vui lòng kiểm tra lại file trong thư mục dự án.")
    st.stop()

# ==========================================
# 3. SIDEBAR - NHẬP LIỆU
# ==========================================
st.sidebar.header("📊 Thông Số Cá Nhân")
st.sidebar.markdown("Điều chỉnh các chỉ số thực tế của bạn bên dưới:")

input_data = {}
input_data['Số_Giờ_Học_Tuần'] = st.sidebar.slider("📚 Số giờ học/tuần", 0, 60, 20)
input_data['Số_Môn_Đang_Học'] = st.sidebar.number_input("📝 Số môn đang học", 1, 12, 5)
input_data['Làm_Thêm'] = st.sidebar.selectbox("💼 Có đi làm thêm không?", ["Không", "Có"])
input_data['Thời_Gian_Ngủ'] = st.sidebar.slider("😴 Thời gian ngủ (giờ/ngày)", 3, 12, 7)
input_data['Tham_Gia_CLB'] = st.sidebar.selectbox("🤝 Tham gia CLB?", ["Không", "Có"])
input_data['Điểm_Danh_%'] = st.sidebar.slider("📍 Tỷ lệ điểm danh (%)", 0, 100, 90)
input_data['Hình_Thức_Học'] = st.sidebar.selectbox("📖 Hình thức học chính", ["Tự học", "Học nhóm"])
input_data['Thời_Gian_Mạng_Xã_Hội'] = st.sidebar.slider("📱 TG dùng MXH (giờ/ngày)", 0, 10, 2)

st.sidebar.markdown("---")
btn_predict = st.sidebar.button("🚀 DỰ ĐOÁN NGAY", type="primary", use_container_width=True)

# ==========================================
# 4. GIAO DIỆN CHÍNH
# ==========================================
st.title("🎓 Hệ Thống AI Dự Đoán & Phân Tích GPA")

if btn_predict:
    # --- XỬ LÝ DỮ LIỆU INPUT (KHẮC PHỤC LỖI VALUEERROR) ---
    input_df = pd.DataFrame([input_data])
    input_df['Làm_Thêm'] = input_df['Làm_Thêm'].map({'Có': 1, 'Không': 0})
    input_df['Tham_Gia_CLB'] = input_df['Tham_Gia_CLB'].map({'Có': 1, 'Không': 0})
    input_df['Hình_Thức_Học'] = input_df['Hình_Thức_Học'].map({'Tự học': 0, 'Học nhóm': 1})
    
    # QUAN TRỌNG: Sắp xếp lại các cột cho đúng thứ tự lúc train model
    input_df = input_df[feature_cols]

    # --- DỰ ĐOÁN ---
    prediction = model.predict(input_df)[0]
    
    # --- HIỂN THỊ KẾT QUẢ ---
    col_left, col_right = st.columns([1, 1.5], gap="large")
    
    with col_left:
        st.markdown(f"""
            <div class="predict-box">
                <h3>GPA DỰ ĐOÁN</h3>
                <h1 style='font-size: 4rem; margin: 0;'>{prediction:.2f}</h1>
                <p>/ 4.0</p>
            </div>
        """, unsafe_allow_html=True)
        
        if prediction >= 3.2:
            st.success("🌟 Xếp loại: Giỏi/Xuất sắc")
            st.balloons()
        elif prediction >= 2.5:
            st.info("✅ Xếp loại: Khá")
        else:
            st.warning("⚠️ Xếp loại: Trung bình")

    with col_right:
        st.subheader("💡 Phân tích từ AI")
        advice_text = ""
        if input_data['Điểm_Danh_%'] < 85:
            advice_text += "• **Cải thiện tỷ lệ điểm danh:** Bạn đang vắng mặt khá nhiều, điều này ảnh hưởng trực tiếp đến điểm quá trình.<br>"
        if input_data['Số_Giờ_Học_Tuần'] < 15:
            advice_text += "• **Tăng thời gian tự học:** Hãy cố gắng dành ít nhất 3h/ngày để ôn tập kiến thức.<br>"
        if input_data['Thời_Gian_Ngủ'] < 6:
            advice_text += "• **Chú ý sức khỏe:** Ngủ quá ít sẽ làm giảm khả năng tập trung trong giờ học.<br>"
        
        if not advice_text:
            advice_text = "• Mọi chỉ số của bạn đều rất tốt! Hãy duy trì phong độ hiện tại để đạt kết quả cao nhất."
            
        st.markdown(f"""
            <div class="advice-card">
                {advice_text}
            </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    
    # --- TABS PHÂN TÍCH ---
    tab1, tab2 = st.tabs(["📊 Phân tích chuyên sâu", "⚖️ So sánh với cộng đồng"])
    
    with tab1:
        col_eda1, col_eda2 = st.columns(2)
        with col_eda1:
            fig_scatter = px.scatter(original_df, x="Số_Giờ_Học_Tuần", y="GPA", 
                                   title="Tương quan: Giờ học vs GPA",
                                   trendline="ols", color_discrete_sequence=['#3b82f6'])
            st.plotly_chart(fig_scatter, use_container_width=True)
        with col_eda2:
            importances = model.feature_importances_
            feat_importances = pd.Series(importances, index=feature_cols).sort_values()
            fig_importance = px.bar(feat_importances, orientation='h', 
                                   title="Các yếu tố ảnh hưởng nhiều nhất",
                                   color_continuous_scale='Blues', color=feat_importances.values)
            st.plotly_chart(fig_importance, use_container_width=True)

    with tab2:
        st.subheader("Vị thế của bạn so với trung bình sinh viên")
        avg_values = original_df.mean(numeric_only=True)
        categories = ['Số_Giờ_Học_Tuần', 'Thời_Gian_Ngủ', 'Điểm_Danh_%', 'Thời_Gian_Mạng_Xã_Hội', 'Số_Môn_Đang_Học']
        
        # Radar chart
        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(r=[input_data[c] for c in categories], theta=categories, fill='toself', name='Bạn'))
        fig_radar.add_trace(go.Scatterpolar(r=[avg_values[c] for c in categories], theta=categories, fill='toself', name='Trung bình'))
        fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True)), showlegend=True)
        st.plotly_chart(fig_radar, use_container_width=True)

else:
    # Màn hình chờ
    st.info("👈 Hãy điền thông số ở bên trái và nhấn 'DỰ ĐOÁN NGAY' để AI bắt đầu phân tích.")
    st.image("https://img.freepik.com/free-vector/data-extraction-concept-illustration_114360-4766.jpg", width=700)
