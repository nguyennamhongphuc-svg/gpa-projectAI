# =========================================================
# AI GPA PREDICTOR - STREAMLIT APP
# Author: ChatGPT (Fixed Version)
# =========================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

# =========================================================
# CẤU HÌNH TRANG
# =========================================================
st.set_page_config(
    page_title="AI GPA Predictor",
    page_icon="🎓",
    layout="wide"
)

# CSS GIAO DIỆN
st.markdown("""
<style>
.main { background-color: #071421; color: white; }
h1, h2, h3 { color: #FFD700; }
[data-testid="stSidebar"] { background: linear-gradient(180deg, #0B1F33, #102A43); }
.stMetric { background-color: #102A43; padding: 20px; border-radius: 15px; border: 1px solid #00D4FF; }
</style>
""", unsafe_allow_html=True)

st.title("🎓 AI GPA Predictor Dashboard")
st.markdown("### Dự đoán GPA sinh viên bằng Machine Learning")

# =========================================================
# ĐỌC DỮ LIỆU (Sửa lỗi: Đọc đúng file CSV bạn đã tải lên)
# =========================================================
try:
    # Sử dụng đúng tên file thực tế trong thư mục của bạn
    df = pd.read_csv("Dataset_GPA_Thuc_Te_250_Responses.xlsx - Dataset_GPA.csv")
    
    # Xử lý khoảng trắng thừa để tránh lỗi khi map dữ liệu
    for col in ["Làm_Thêm", "Tham_Gia_CLB", "Hình_Thức_Học"]:
        df[col] = df[col].astype(str).str.strip()
except FileNotFoundError:
    st.error("❌ Không tìm thấy file Dataset_GPA_Thuc_Te_250_Responses.xlsx - Dataset_GPA.csv")
    st.stop()

# =========================================================
# XỬ LÝ DỮ LIỆU & HUẤN LUYỆN MODEL
# =========================================================
data = df.copy()

# Encode dữ liệu phân loại
data["Làm_Thêm"] = data["Làm_Thêm"].map({"Có": 1, "Không": 0})
data["Tham_Gia_CLB"] = data["Tham_Gia_CLB"].map({"Có": 1, "Không": 0})
data["Hình_Thức_Học"] = data["Hình_Thức_Học"].map({"Học nhóm": 1, "Tự học": 0})

# Loại bỏ các hàng bị lỗi (nếu có) sau khi map
data = data.dropna()

X = data.drop(["GPA", "Mã_Sinh_Viên"], axis=1)
y = data["GPA"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# Đánh giá độ chính xác
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)

# =========================================================
# SIDEBAR - INPUT
# =========================================================
st.sidebar.title("⚙️ Nhập thông tin sinh viên")
study_hours = st.sidebar.slider("📚 Số giờ học mỗi tuần", 0, 50, 20)
subjects = st.sidebar.slider("📖 Số môn đang học", 1, 12, 5)
part_time = st.sidebar.selectbox("💼 Có làm thêm không?", ["Có", "Không"])
sleep_time = st.sidebar.slider("😴 Thời gian ngủ mỗi đêm", 4, 12, 7)
club = st.sidebar.selectbox("🎯 Có tham gia CLB?", ["Có", "Không"])
attendance = st.sidebar.slider("🏫 Tỉ lệ đi học (%)", 0, 100, 80)
study_method = st.sidebar.selectbox("🧠 Hình thức học", ["Tự học", "Học nhóm"])
social_media = st.sidebar.slider("📱 Thời gian dùng MXH", 0, 15, 3)

predict_button = st.sidebar.button("🚀 Dự đoán GPA")

# =========================================================
# TABS
# =========================================================
tab1, tab2, tab3 = st.tabs(["📈 Dự đoán kết quả", "📊 Phân tích dữ liệu (EDA)", "🕸️ So sánh"])

with tab1:
    st.subheader("🎯 Kết quả dự đoán GPA")
    if predict_button:
        # Encode input dựa trên lựa chọn người dùng
        input_data = pd.DataFrame({
            "Số_Giờ_Học_Tuần": [study_hours],
            "Số_Môn_Đang_Học": [subjects],
            "Làm_Thêm": [1 if part_time == "Có" else 0],
            "Thời_Gian_Ngủ": [sleep_time],
            "Tham_Gia_CLB": [1 if club == "Có" else 0],
            "Điểm_Danh_%": [attendance],
            "Hình_Thức_Học": [1 if study_method == "Học nhóm" else 0],
            "Thời_Gian_Mạng_Xã_Hội": [social_media]
        })

        predicted_gpa = round(model.predict(input_data)[0], 2)
        st.metric(label="📌 GPA Dự Đoán", value=f"{predicted_gpa}/4.0")

        # TƯ VẤN
        if predicted_gpa < 2.5:
            st.error("⚠️ GPA mức thấp. Hãy tăng giờ học, giảm dùng MXH và ngủ đủ giấc.")
        elif predicted_gpa < 3.2:
            st.warning("📚 GPA mức khá. Bạn có thể cải thiện bằng cách tăng giờ tự học.")
        else:
            st.success("🏆 GPA rất tốt! Hãy duy trì phong độ này.")
            st.balloons()
            
        st.info(f"🤖 Model: Random Forest | Accuracy R²: {round(r2, 3)}")
    else:
        st.warning("⬅️ Hãy nhập thông tin ở Sidebar để bắt đầu.")

with tab2:
    st.subheader("📊 Phân tích tương quan")
    col1, col2 = st.columns(2)
    with col1:
        fig_scatter = px.scatter(df, x="Số_Giờ_Học_Tuần", y="GPA", color="GPA", template="plotly_dark")
        st.plotly_chart(fig_scatter, use_container_width=True)
    with col2:
        importance_df = pd.DataFrame({"Feature": X.columns, "Importance": model.feature_importances_}).sort_values(by="Importance")
        fig_imp = px.bar(importance_df, x="Importance", y="Feature", orientation="h", template="plotly_dark")
        st.plotly_chart(fig_imp, use_container_width=True)

with tab3:
    st.subheader("🕸️ Radar Chart So sánh")
    # Chuẩn hóa giá trị để hiển thị Radar (Scale 0-100)
    user_vals = [study_hours*2, attendance, sleep_time*8.3, (15-social_media)*6.6]
    avg_vals = [data["Số_Giờ_Học_Tuần"].mean()*2, data["Điểm_Danh_%"].mean(), 
                data["Thời_Gian_Ngủ"].mean()*8.3, (15-data["Thời_Gian_Mạng_Xã_Hội"].mean())*6.6]
    
    categories = ["Giờ học", "Điểm danh", "Giấc ngủ", "Hạn chế MXH"]
    fig_radar = go.Figure()
    fig_radar.add_trace(go.Scatterpolar(r=user_vals, theta=categories, fill='toself', name='Bạn'))
    fig_radar.add_trace(go.Scatterpolar(r=avg_vals, theta=categories, fill='toself', name='Trung bình'))
    fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])), template="plotly_dark")
    st.plotly_chart(fig_radar, use_container_width=True)

st.markdown("---")
st.caption("© 2026 AI GPA Predictor Dashboard | Research Project")
