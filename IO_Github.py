import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="IO analysis program", layout="wide")
st.title("IO analysis program")

if "step" not in st.session_state:
    st.session_state.step = "1. 데이터 로드 및 설정"
if "pm" not in st.session_state:
    st.session_state.pm = None
if "results" not in st.session_state:
    st.session_state.results = {}
if "config_df" not in st.session_state:
    st.session_state.config_df = None

@st.cache_data
def load_data():
    # 경로 설정(사용자 환경에 맞게 수정)
    base_path = "./"
    # 파일명 정확하게 작성
    file_Tts = base_path + "총거래표.csv"
    file_Ttsd = base_path + "국산거래표.csv"
    file_Ttse = base_path + "고용표.csv"
    file_pmb = base_path + "행통합행렬.csv"
    file_industry_list = base_path + "소분류(상품).csv"
    base_sector_names = ["농림수산품", "광산품", "음식료품", "섬유 및 가죽제품", "목재 및 종이, 인쇄", "석탄 및 석유제품", "화학제품", "비금속광물제품", "1차 금속제품", "금속가공제품",
                         "컴퓨터, 전자 및 광학기기", "전기장비", "기계 및 장비", "운송장비", "기타 제조업 제품", "제조임가공 및 산업용 장비 수리", "전력, 가스 및 증기",
                         "수도, 폐기물처리 및 재활용서비스", "건설", "도소매 및 상품중개서비스", "운송서비스", "음식점 및 숙박서비스", "정보통신 및 방송서비스", "금융 및 보험서비스",
                         "부동산서비스", "전문, 과학 및 기술서비스", "사업지원서비스", "공공행정, 국방 및 사회보장", "교육서비스", "보건 및 사회복지서비스", "예술, 스포츠 및 여가 관련 서비스", "기타서비스", "기타"]

    try:
        Tts = pd.read_csv(file_Tts, encoding='utf-8')
        Ttsd = pd.read_csv(file_Ttsd, encoding='utf-8')
        Ttse = pd.read_csv(file_Ttse, encoding='utf-8')
        pmb = pd.read_csv(file_pmb, encoding='utf-8',header=None)
        industry_list = pd.read_csv(file_industry_list, encoding='utf-8', header=None)
        
        Tts = Tts.iloc[5:,2:].apply(lambda x: pd.to_numeric(x.astype(str).str.replace(',', ''), errors='coerce'))
        Ttsd = Ttsd.iloc[5:,2:].apply(lambda x: pd.to_numeric(x.astype(str).str.replace(',', ''), errors='coerce'))
        Ttse = Ttse.iloc[5:,4:].apply(lambda x: pd.to_numeric(x.astype(str).str.replace(',', ''), errors='coerce'))
        pmb = pmb.iloc[1:,1:].apply(lambda x: pd.to_numeric(x.astype(str).str.replace(',', ''), errors='coerce'))
        
        Tts_m = Tts.to_numpy().astype(float)
        Ttsd_m = Ttsd.to_numpy().astype(float)
        Ttse_m = Ttse.to_numpy().astype(float)
        pmb_m = pmb.to_numpy().astype(float)
        industry_list = industry_list.iloc[:,0].tolist()
        
        if len(base_sector_names) < pmb_m.shape[0]:
            base_sector_names += [f"기타_{i}" for i in range(pmb_m.shape[0] - len(base_sector_names))]
        else:
            base_sector_names = base_sector_names[:pmb_m.shape[0]]
        return Tts_m, Ttsd_m, Ttse_m, pmb_m, industry_list, base_sector_names

    except Exception as e:
        st.error(f"데이터 로드 중 오류 발생: {e}")
        return None, None, None, None, None, None

Tts_m, Ttsd_m, Ttse_m, pmb_m, industry_list, base_sector_names = load_data()

st.sidebar.header("IO Analysis Pro")
menu = ["1. 산업 재분류 설정", "2. 통합행렬(pm) 생성", "3. 산업연관분석 실행", "4. 분석 결과 및 시각화"]
choice = st.sidebar.radio("단계 선택", menu)

#1. 산업 재분류 설정(비율 설정 기능 추가)
if choice == "1. 산업 재분류 설정":
    st.title("Step 1. 산업 재분류 설정")
    st.markdown("""**논문 방법론 적용:** 기존 소분류 산업에서 일정 **비율(%)**만큼 떼어내어 새로운 대분류 생성
                   (예: 조경시공/관리업: 7.9%, 조경계획/설계업: 21% 등)""")
    col1, col2 = st.columns([1,2])
    if "new_sector_name" not in st.session_state:
        st.session_state.new_sector_name = ""
    with col1:
        with st.form("name_input_form"):
            st.caption("1. 신설할 산업의 명칭을 입력하고 '명칭 확정'을 누르세요.")
            name_input = st.text_input("신설할 대분류 명칭", value=st.session_state.new_sector_name)
            name_submitted = st.form_submit_button("명칭 확정")

            if name_submitted:
                if name_input.strip():
                    st.session_state.new_sector_name = name_input.strip()
                    st.success(f"명칭이 '{st.session_state.new_sector_name}'(으)로 설정되었습니다.")
                else:
                    st.warning("산업 명칭을 입력해주세요.")

    st.markdown("---")
        
    st.subheader("소분류 산업 선택")

    if st.session_state.config_df is None:
        st.session_state.config_df = pd.DataFrame({"소분류명": industry_list, "선택": False})

    with st.form("industry_select_form"):
        st.caption("원하는 산업을 연속으로 체크한 뒤 아래 '선택 확정' 버튼을 눌러주세요.")

        edited_df = st.data_editor(st.session_state.config_df,
                                   column_config={"선택": st.column_config.CheckboxColumn("포함 여부", help="체크하면 신설 산업에 포함됨")},
                                   hide_index=True, use_container_width=True, height=350)
        submitted = st.form_submit_button("선택한 소분류 확정")

        if submitted:
            st.session_state.config_df = edited_df
            st.session_state.selected_df = edited_df[edited_df["선택"] == True].copy()
            st.success("선택이 확정되었습니다! 아래 비율 설정을 진행하세요.")
            
    selected_df = st.session_state.get("selected_df", pd.DataFrame())

    if not selected_df.empty:
        st.info(f"현재 **{len(selected_df)}개**의 소분류가 선택되었습니다.")
    else:
        st.info("아직 선택된 산업이 없습니다.")

    st.subheader("소분류 비율 설정")
    
    if selected_df.empty:
        st.warning("위에서 소분류를 선택하고 '선택 확정' 버튼을 누르면, 여기에서 비율을 설정할 수 있습니다.")
    else:
        if "ratio_df" not in st.session_state:
            selected_df["적용비율(%)"] = 100.0
            st.session_state.ratio_df = selected_df
        else:
            try:
                old = st.session_state.ratio_df.set_index("소분류명")["적용비율(%)"]
                selected_df["적용비율(%)"] = selected_df["소분류명"].map(old["적용비율(%)"]).fillna(100.0)
            except Exception:
                selected_df["적용비율(%)"] = 100.0

            st.session_state.ratio_df = selected_df
            
        with st.form("ratio_setting_form"):
            st.caption("각 산업별 적용 비율을 입력한 뒤, '비율 설정 확정' 버튼을 눌러주세요.")

            edited_ratio_df = st.data_editor(st.session_state.ratio_df,
                                             column_config={"적용비율(%)": st.column_config.NumberColumn("비율(%)", min_value=0.0, max_value=100.0, step=0.1, format="%1f")},
                                             hide_index=True, use_container_width=True)

            submit_ratio = st.form_submit_button("비율 설정 확정")

            if submit_ratio:
                st.session_state.ratio_df = edited_ratio_df
                st.success("비율 설정이 확정되었습니다! 이제 Step 2로 넘어가셔도 됩니다.")

#2. 통합행렬(pm) 생성
elif choice == "2. 통합행렬(pm) 생성":
    st.title("Step 2. 통합행렬(pm) 생성")

    if st.session_state.config_df is None or "ratio_df" not in st.session_state:
        st.error("Step 1에서 산업 선택 및 비율 설정을 먼저 진행해주세요.")
        st.stop()

    selected_rows = st.session_state.config_df[st.session_state.config_df["선택"] == True]
    st.write(f"**{len(selected_rows)}개**의 소분류 기반 통합행렬 생성")

    if selected_rows.empty:
        st.warning("선택된 소분류가 없습니다.")
        st.stop()

    if st.button("통합행렬(pm) 생성하기"):
        #기존 33개 + n개 신설 = 33+n개 행
        n_original = pmb_m.shape[0]
        n_new = n_original + 1
        n_sub = pmb_m.shape[1]
        
        pm_new = np.zeros((n_new, n_sub))
        #기존 pmb 구조 복사(33*165)
        pm_new[:n_original, :] = pmb_m

        ratio_map = dict(zip(st.session_state.ratio_df["소분류명"], st.session_state.ratio_df["적용비율(%)"] / 100.0))
        
        #선택된 소분류에 대한 비율 적용(재분류)
        #선택된 소분류 j에 대해 신설 산업(행 33)으로 r% 이동, 원래 부모 산업(행 i)엔 (100-r)% 남김
        for idx, row in selected_rows.iterrows():
            col_idx = idx
            ratio = ratio_map.get(industry_list[col_idx], 0.0)
            
            # 원래 이 소분류가 속해 있던 대분류 인덱스 찾기(값이 1인 곳)
            original_parent_idx = np.argmax(pmb_m[:, col_idx])

            #로직 적용: 신설 산업 행에 ratio 할당
            pm_new[n_original, col_idx] = ratio

            #원래 산업 행 수정(1-ratio), 만약 100% 이동이면 0으로 표기
            pm_new[original_parent_idx, col_idx] = 1.0 - ratio
            
        st.session_state.pm = pm_new
        
        #대분류 이름 리스트 업데이트
        new_sector_list = base_sector_names + ["(신설) " + st.session_state.new_sector_name]
        st.session_state.sector_names = new_sector_list
        
        st.success("통합행렬(pm) 생성 완료")
        st.write(f"행렬 크기: {pm_new.shape} (34개 대분류*165개 소분류)")
        
        #검증: 열합이 1인지 확인
        col_sums = pm_new.sum(axis=0)
        if np.allclose(col_sums, 1.0):
            st.info("검증 성공: 모든 소분류의 배분 합계가 1.0입니다.")
        else:
            st.error("검증 실패: 배분 합계가 1.0이 아닌 열이 존재합니다.")
            st.write(col_sums)
            
        #데이터 프레임으로 보여주기(히트맵 스타일)
        pm_df = pd.DataFrame(pm_new, index=[f"{i+1}.{name}" for i, name in enumerate(new_sector_list)], columns=industry_list)
        st.dataframe(pm_df.style.background_gradient(cmap="Blues", axis=None))

#3. 산업연관분석 실행
elif choice == "3. 산업연관분석 실행":
    st.title("Step 3. 산업연관분석 실행")

    if st.session_state.pm is None:
        st.error("통합행렬(pm)이 없습니다. Step 2를 먼저 진행하세요.")
    else:
        st.write("생성된 통합행렬(pm)을 기반으로 생산/부가가치/고용 유발계수를 계산합니다.")
        if st.button("분석 실행(Run Analysis)"):
            with st.spinner('산업연관분석 계산 중...'):
                pm = st.session_state.pm
                n = pm.shape[0]
                Ct = Tts_m[-1, 0:165]@pm.T
                Z = pm@Ttsd_m[:165, :165]@pm.T
                A = Z/np.tile(Ct, reps=[n,1])
                I = np.eye(n)
                Lf = np.linalg.inv(I-A)
                Lf_rw_sm = Lf.sum(axis=0)
                Lf_cl_sm = Lf.sum(axis=1)
                Total_sm = Lf_cl_sm.sum()
                LF_h = np.hstack((Lf, Lf_rw_sm.reshape(n,1)))
                bottom_row = np.hstack((Lf_cl_sm.reshape(1,n), np.array([[Total_sm]])))
                LF_A = np.vstack((LF_h, bottom_row.reshape(1,n+1)))
                Lf_bwd = (Lf_rw_sm/Lf_rw_sm.mean()).astype(float)
                Lf_fwd = (Lf_cl_sm/Lf_cl_sm.mean()).astype(float)
                Va = Tts_m[-2,:165]@pm.T
                V_ratio = Va/np.tile(Ct, reps=[n,1])
                V = np.diag(V_ratio)
                V_m = np.diagflat(V)
                Lv = V_m@Lf
                Ee = (Ttse_m[0:165,0].reshape(1,165).astype(float)@pm.T)
                Ee_ratio = ((Ee*1000/Ct.reshape(1,n))).astype(float)
                Ew = (Ttse_m[0:165,1].reshape(1,165).astype(float)@pm.T)
                Ew_ratio = ((Ew*1000/Ct.reshape(1,n))).astype(float)
                Ee_m = np.diagflat(Ee_ratio)
                Ew_m = np.diagflat(Ew_ratio)
                Le = Ee_m@Lf
                Lw = Ew_m@Lf
                
                st.session_state["Lf"] = Lf
                st.session_state["Lv"] = Lv
                st.session_state["Le"] = Le
                st.session_state["Lw"] = Lw
                st.session_state["V_ratio"] = V_ratio
                st.session_state["Ee_ratio"] = Ee_ratio
                st.session_state["Ew_ratio"] = Ew_ratio
                st.session_state["Lf_bwd"] = Lf_bwd
                st.session_state["Lf_fwd"] = Lf_fwd
                
                st.success("분석 완료!, '4. 분석결과 및 시각화' 탭에서 확인하세요.")

                st.markdown("### 분석 결과 요약(신설 산업)")
                new_idx = -1
                new_name = st.session_state.sector_names[new_idx]

                col1, col2, col3, col4 = st.columns(4)
                col1.metric("생산 유발효과 (Total)", f"{Lf.sum(axis=0)[new_idx]:.4f}")
                col2.metric("부가가치 유발효과 (Total)", f"{Lv.sum(axis=0)[new_idx]:.4f}")
                col3.metric("취업 유발효과 (명/10억 원)", f"{Le.sum(axis=0)[new_idx]:.2f}")
                col4.metric("고용 유발효과 (명/10억 원)", f"{Lw.sum(axis=0)[new_idx]:.2f}")

                st.markdown("### 생산유발계수 행렬($L_f$) 미리보기")
                df_preview = pd.DataFrame(Lf, index=st.session_state.sector_names, columns=st.session_state.sector_names)
                st.dataframe(df_preview.iloc[-5:, -5:].style.background_gradient(cmap="Blues").format("{:.3f}"))
                st.caption("전체 결과는 **Step 4**에서 자세히 확인하세요.")

#4. 분석 결과 및 시각화
elif choice == "4. 분석 결과 및 시각화":
    st.title("Step 4. 분석 결과 및 시각화")

    if "Lf" not in st.session_state:
        st.warning("분석 결과가 없습니다. Step 3에서 분석을 실행해주세요.")
        st.stop()
    else:
        Lf = st.session_state["Lf"]
        Lv = st.session_state["Lv"]
        Le = st.session_state["Le"]
        Lw = st.session_state["Lw"]
        V_ratio = st.session_state["V_ratio"]
        Ee_ratio = st.session_state["Ee_ratio"]
        Ew_ratio = st.session_state["Ew_ratio"]

        names = np.array(st.session_state["sector_names"])
        n = len(names)
        idx_new = n - 1
        name_new = names[idx_new]

        Lf_bwd = st.session_state["Lf"].sum(axis=0) / st.session_state["Lf"].sum(axis=0).mean()
        Lf_fwd = st.session_state["Lf"].sum(axis=1) / st.session_state["Lf"].sum(axis=1).mean()
    
        st.header("산업연관분석 결과")
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["종합 결과", "생산 유발계수", "부가가치 유발계수", "취업 유발계수", "고용 유발계수", "다운로드"])

        def show_detail_tab(effect_name, matrix, direct_vec=None, is_production=False):
                st.subheader(f"{name_new}의 {effect_name} 상세 분석")

                # 요약 매트릭스
                total_vec = matrix.sum(axis=0)
                total_val = total_vec[idx_new]

                if is_production:
                    direct_val = matrix[idx_new, idx_new]
                else:
                    direct_val = matrix[idx_new] if direct_vec is not None else 0

                indirect_val = total_val - direct_val

                cl, c2, c3 = st.columns(3)
                cl.metric("1. 총 유발효과", f"{total_val: .4f}")
                c2.metric("1. 직접 유발효과", f"{direct_val: .4f}")
                c3.metric("1. 간접 유발효과", f"{indirect_val: .4f}")

                st.markdown("---")

                c_col, c_row = st.columns(2)
                with c_col:
                    st.markdown(f"### 1. {name_new}이 타 산업에 미치는 영향(Top 10)")
                    st.caption(f"{name_new}의 투자가 **어떤 산업**을 가장 많이 자극하는가?")

                    impact_col = matrix[:, idx_new]
                    impact_df = pd.DataFrame({"산업명": names, "유발액": impact_col})
                    impact_df_filtered = impact_df[impact_df["산업명"] != name_new]
                    top10_col = impact_df_filtered.sort_values("유발액", ascending=False).head(10)

                    top10_col = top10_col.reset_index(drop=True)
                    top10_col.index = top10_col.index + 1
                    st.dataframe(top10_col.style.bar(subset=["유발액"], color="#d65f5f"))

                with c_row:
                    st.markdown(f"### 2. {name_new}이 타 산업으로부터 받는 영향(Top 10)")
                    st.caption(f"**타 산업**이 성장할 때 **{name_new}**이 얼마나 영향을 받는가?")

                    impact_row = matrix[idx_new, :]
                    impact_row_df = pd.DataFrame({"산업명": names, "유발액": impact_row})
                    impact_row_filtered = impact_row_df[impact_row_df["산업명"] != name_new]
                    top10_row = impact_row_filtered.sort_values("유발액", ascending=False).head(10)

                    top10_row = top10_row.reset_index(drop=True)
                    top10_row.index = top10_row.index + 1
                    st.dataframe(top10_row.style.bar(subset=["유발액"], color="#4682B4"))

                st.markdown("---")
                st.markdown(f"### 전체 산업 내 위상 (Ranking)")
                st.caption(f"전체 산업 중 **{name_new}의 파급력**은 몇 위인가?")
                rank_df = pd.DataFrame({"산업명": names, "총 유발효과": total_vec})
                rank_df = rank_df.sort_values("총 유발효과", ascending=False).reset_index(drop=True)
                rank_df.index += 1

                try:
                    my_row = rank_df[rank_df["산업명"] == name_new]
                    if not my_row.empty:
                        my_rank = my_row.index[0]
                        st.info(f"**{name_new}**은(는) 전체 **{n}개 산업 중 {my_rank}위** 입니다.")

                        if my_rank > 10:
                            display_df = pd.concat([rank_df.head(10), my_row])
                        else:
                            display_df = rank_df.head(10)
                        st.dataframe(display_df.style.apply(lambda x: ["background-color: lightyellow" if v == name_new else "" for v in x], axis=1))
                    else:
                        st.warning("신설 산업을 찾을 수 없습니다.")
                except:
                    st.info("순위 집계 중 오류")
                        
        with tab1:
            st.subheader("신설 산업 vs 전산업 평균")
        
            lf_sum = Lf.sum(axis=0)
            lv_sum = Lv.sum(axis=0)
            le_sum = Le.sum(axis=0)
            lw_sum = Lw.sum(axis=0)
        
            summary_df = pd.DataFrame({"구분": ["생산유발", "부가가치유발", "취업유발", "고용유발"],
                                       name_new: [lf_sum[idx_new], lv_sum[idx_new], le_sum[idx_new], lw_sum[idx_new]],
                                       "전산업 평균": [lf_sum.mean(), lv_sum.mean(), le_sum.mean(), lw_sum.mean()]})
        
            fig = px.bar(summary_df.melt(id_vars="구분", var_name="산업", value_name="계수"),
                         x="구분", y="계수", color="산업",
                         barmode="group", text_auto=".3f",
                         title=f"{name_new} 산업의 경제적 파급효과 비교")
            st.plotly_chart(fig, use_container_width=True)
        
            st.markdown("---")
        
            st.subheader("전·후방 연쇄효과 분석")
        
            scatter_df = pd.DataFrame({"산업명": names,
                                       "영향력계수(BL)": Lf.sum(axis=0) / Lf.sum(axis=0).mean(),
                                       "감응도계수(FL)": Lf.sum(axis=1) / Lf.sum(axis=1).mean(),
                                       "구분": ["신설산업" if i == idx_new else "기타산업" for i in range(n)]})
            fig_scatter = px.scatter(scatter_df,
                                     x="감응도계수(FL)", y="영향력계수(BL)",
                                     color="구분", text="산업명", hover_data=["산업명"],
                                     title="산업간 연쇄효과 분석(Impact vs Sensitivity)",
                                     color_discrete_map={"신설산업": "red", "기타산업": "blue"})
            fig_scatter.add_hline(y=1, line_dash="dash", line_color="gray")
            fig_scatter.add_vline(x=1, line_dash="dash", line_color="gray")
            fig_scatter.update_traces(textposition="top center")
            st.plotly_chart(fig_scatter, use_container_width=True)
        
        with tab2:
            st.subheader(f"생산 유발계수 행렬 (Lf)")
            Lf_df = pd.DataFrame(Lf, index=names, columns=names)
            st.dataframe(Lf_df.style.background_gradient(cmap="Blues"))

            show_detail_tab("생산 유발효과", Lf, is_production=True)
    
        with tab3:
            st.subheader("부가가치 유발계수 행렬 (Lv)")
            Lv_df = pd.DataFrame(Lv, index=Lf_df.index, columns=Lf_df.columns)
            st.dataframe(Lv_df.style.background_gradient(cmap="Greens"))

            show_detail_tab("부가가치 유발효과", Lv, is_production=True)
        
        with tab4:
            st.subheader("취업 유발계수 행렬 (Le)")
            Le_df = pd.DataFrame(Le, index=Lf_df.index, columns=Lf_df.columns)
            st.dataframe(Le_df.style.background_gradient(cmap="Oranges"))

            show_detail_tab("취업 유발효과", Le, is_production=True)
        
        with tab5:
            st.subheader("고용 유발계수 행렬 (Lw)")
            Lw_df = pd.DataFrame(Lw, index=Lf_df.index, columns=Lf_df.columns)
            st.dataframe(Lw_df.style.background_gradient(cmap="Oranges"))
        
            show_detail_tab("고용 유발효과", Lw, is_production=True)
    
        with tab6:
            st.subheader("분석 결과 다운로드")
            def convert_df(df):
                return df.to_csv(encoding="utf-8-sig").encode('utf-8-sig')

            cl, c2, c3, c4 = st.columns(4)
            st.download_button("생산 유발계수(Lf)", convert_df(pd.DataFrame(Lf, index=names, columns=names)), "Lf_생산 유발계수.csv", "text/csv")
            st.download_button("부가가치 유발계수(Lv)", convert_df(pd.DataFrame(Lv, index=names, columns=names)), "Lv_부가가치 유발계수.csv", "text/csv")
            st.download_button("취업 유발계수(Le)", convert_df(pd.DataFrame(Le, index=names, columns=names)), "Le_취업 유발계수.csv", "text/csv")

            st.download_button("고용 유발계수(Lw)", convert_df(pd.DataFrame(Lw, index=names, columns=names)), "Lw_고용 유발계수.csv", "text/csv")
