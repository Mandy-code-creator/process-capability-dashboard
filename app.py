# --- BỐ CỤC BIỂU ĐỒ NÂNG CAO ---
        col_left, col_right = st.columns([2, 1])

        with col_left:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            # --- CONTROL CHART VỚI CẢNH BÁO MÀU SẮC ---
            x_axis = df_clean[time_col] if time_col else list(range(1, n + 1))
            
            # Xác định màu sắc cho từng điểm: Đỏ nếu ngoài Spec, Xanh nếu trong Spec
            point_colors = ['#D83B01' if (val < lsl or val > usl) else '#0078D4' for val in data]
            point_sizes = [12 if (val < lsl or val > usl) else 8 for val in data] # Điểm lỗi to hơn

            fig_ctrl = go.Figure()
            
            # Vẽ đường nối
            fig_ctrl.add_trace(go.Scatter(
                x=x_axis, y=data, 
                mode='lines+markers',
                marker=dict(size=point_sizes, color=point_colors, line=dict(width=1, color='white')),
                line=dict(width=2, color='#0078D4'),
                name="Measurement"
            ))
            
            # Thêm các đường giới hạn
            fig_ctrl.add_hline(y=usl, line_dash="dash", line_color="#D83B01", line_width=2, annotation_text="USL")
            fig_ctrl.add_hline(y=lsl, line_dash="dash", line_color="#D83B01", line_width=2, annotation_text="LSL")
            fig_ctrl.add_hline(y=mean, line_color="#107C10", line_width=1, annotation_text="Mean")

            fig_ctrl.update_layout(
                height=450, template="plotly_white", 
                title="Process Trend & Outlier Detection",
                margin=dict(l=40, r=40, t=40, b=40),
                xaxis=dict(showline=True, linecolor='#605E5C', mirror=True, tickangle=45),
                yaxis=dict(showline=True, linecolor='#605E5C', mirror=True)
            )
            st.plotly_chart(fig_ctrl, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with col_right:
            # --- BOXPLOT VỚI CẢNH BÁO OUTLIERS ---
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            fig_box = go.Figure()
            fig_box.add_trace(go.Box(
                y=data, 
                marker_color='#0078D4',
                boxpoints='all', # Hiển thị tất cả các điểm bên cạnh box
                jitter=0.3,
                pointpos=-1.8,
                name="Distribution"
            ))
            # Đường giới hạn trên Boxplot
            fig_box.add_hline(y=usl, line_dash="dot", line_color="#D83B01")
            fig_box.add_hline(y=lsl, line_dash="dot", line_color="#D83B01")
            
            fig_box.update_layout(height=210, margin=dict(l=10, r=10, t=30, b=10), title="Boxplot Analysis")
            st.plotly_chart(fig_box, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

            # --- HISTOGRAM (TÔ MÀU CỘT VI PHẠM) ---
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            counts, bins = np.histogram(data, bins=10)
            bin_centers = 0.5 * (bins[:-1] + bins[1:])
            bar_colors = ['#D83B01' if (x < lsl or x > usl) else '#0078D4' for x in bin_centers]
            
            fig_hist = go.Figure()
            fig_hist.add_trace(go.Bar(x=bin_centers, y=counts, marker_color=bar_colors))
            fig_hist.update_layout(height=210, margin=dict(l=10, r=10, t=30, b=10), title="Frequency")
            st.plotly_chart(fig_hist, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
