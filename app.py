import streamlit as st
from streamlit_gsheets import GSheetsConnection

# Tiêu đề
st.title("Kết nối Google Sheets thành công! ✅")

# URL file Sheet bạn vừa tạo
sheet_url = "DÁN_LINK_VÀO_ĐÂY"

try:
    # Kết nối
    conn = st.connection("gsheets", type=GSheetsConnection)
    # Đọc dữ liệu từ cột 'Measurements'
    df = conn.read(spreadsheet=sheet_url)
    
    st.write("Dữ liệu từ Sheet của bạn:")
    st.dataframe(df)
    
    # Lấy dữ liệu để tính toán
    data = df['Measurements'].dropna().tolist()
    st.success(f"Đã tìm thấy {len(data)} giá trị đo lường.")
    
except Exception as e:
    st.error(f"Lỗi kết nối: {e}")
    st.info("Hãy đảm bảo bạn đã chỉnh quyền 'Anyone with the link' trên file Sheet.")
