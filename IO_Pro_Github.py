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
    with col1:
        new_sector_name = st.text_input("신설할 대분류 명칭", value=st.session_state.get("new_sector_name",""))
        if new_sector_name.strip() == "":
            display_name = "신설 산업"
        else:
            display_name = new_sector_name
        st.session_state.new_sector_name = display_name
        
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
                st.session_state["Lf_bwd"] = Lf_bwd
                st.session_state["Lf_fwd"] = Lf_fwd
                
                st.success("분석 완료!, '4. 분석결과 및 시각화' 탭에서 확인하세요.")

#4. 분석 결과 및 시각화
elif choice == "4. 분석 결과 및 시각화":
    st.title("Step 4. 분석 결과 및 시각화")

    if "Lf" not in st.session_state:
        st.warning("분석 결과가 없습니다. Step 3에서 분석을 실행해주세요.")
        st.stop()

    Lf = st.session_state["Lf"]
    Lv = st.session_state["Lv"]
    Le = st.session_state["Le"]
    Lw = st.session_state["Lw"]
    names = st.session_state["sector_names"]
    Lf_bwd = st.session_state["Lf"].sum(axis=0) / st.session_state["Lf"].sum(axis=0).mean()
    Lf_fwd = st.session_state["Lf"].sum(axis=1) / st.session_state["Lf"].sum(axis=1).mean()

    st.header("산업연관분석 결과")
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["생산 유발계수", "부가가치 유발계수", "고용 유발계수", "시각화 분석", "다운로드"])
    with tab1:
        st.subheader(f"생산 유발계수 행렬 (Lf)")
        Lf_df = pd.DataFrame(Lf, index=names, columns=names)
        st.dataframe(Lf_df.style.background_gradient(cmap="Blues"))

        st.markdown("#### 영향력 계수(Backward linkage)")
        bwd_df = pd.DataFrame({"영향력 계수": Lf_bwd}, index=Lf_df.columns)
        st.dataframe(bwd_df.sort_values("영향력 계수", ascending=False))

        st.markdown("#### 감응도 계수(Forward linkage)")
        fwd_df = pd.DataFrame({"감응도 계수": Lf_fwd}, index=Lf_df.index)
        st.dataframe(fwd_df.sort_values("감응도 계수", ascending=False))

    with tab2:
        st.subheader("부가가치 유발계수 행렬 (Lv)")
        Lv_df = pd.DataFrame(Lv, index=Lf_df.index, columns=Lf_df.columns)
        st.dataframe(Lv_df.style.background_gradient(cmap="Greens"))
    
        st.markdown("#### 부가가치 유발계수 합계")
        st.dataframe(pd.DataFrame(Lv_df.sum(axis=0),columns=["부가가치 유발계수 합"]))
    
    with tab3:
        st.subheader("고용 유발계수 행렬 (Le)")
        Le_df = pd.DataFrame(Le, index=Lf_df.index, columns=Lf_df.columns)
        st.dataframe(Le_df.style.background_gradient(cmap="Oranges"))
    
        st.markdown("#### 고용 유발계수 합계(10억원 단위)")
        st.dataframe(pd.DataFrame(Le_df.sum(axis=0),columns=["고용 유발계수 합"]))

    with tab4:
        st.subheader("신설 산업 vs 전산업 평균 비교")
    
        names = st.session_state.sector_names
        idx_new = len(names) - 1
        name_new = names[idx_new]
    
        lf_sum = Lf.sum(axis=0)
        lv_sum = Lv.sum(axis=0)
        le_sum = Le.sum(axis=0)
    
        summary_df = pd.DataFrame({"구분": ["생산유발", "부가가치유발", "고용유발"],
                                   name_new: [lf_sum[idx_new], lv_sum[idx_new], le_sum[idx_new]],
                                   "전산업 평균": [lf_sum.mean(), lv_sum.mean(), le_sum.mean()]})
    
        fig = px.bar(summary_df.melt(id_vars="구분", var_name="산업", value_name="계수"),
                     x="구분", y="계수", color="산업",
                     barmode="group", text_auto=".3f",
                     title=f"{name_new} 산업의 경제적 파급효과 비교")
        st.plotly_chart(fig, use_container_width=True)
    
        st.markdown("---")
    
        st.subheader("전·후방 연쇄효과 분석")
    
        scatter_df = pd.DataFrame({"산업명": names,
                                   "영향력계수(BL)": Lf.sum(axis=0) / Lf.sum(axis=0).mean(),
                                   "감응도계수(FL)": Lf.sum(axis=1) / Lf.sum(axis=1).mean()})
    
        fig_scatter = px.scatter(scatter_df,
                                 x="감응도계수(FL)", y="영향력계수(BL)",
                                 text="산업명",
                                 title="산업간 연쇄효과 분석")
        fig_scatter.add_hline(y=1, line_dash="dash")
        fig_scatter.add_vline(x=1, line_dash="dash")
        fig_scatter.update_traces(textposition="top center")
    
        st.plotly_chart(fig_scatter, use_container_width=True)

    with tab5:
        st.subheader("분석 결과 다운로드")
        st.download_button(label="생산 유발계수(Lf) 다운로드", data=Lf_df.to_csv(encoding="utf-8-sig"), file_name="Lf_생산 유발계수.csv", mime="text/csv")
    
        st.download_button(label="부가가치 유발계수(Lv) 다운로드", data=Lv_df.to_csv(encoding="utf-8-sig"), file_name="Lv_부가가치 유발계수.csv", mime="text/csv")
    
        st.download_button(label="고용 유발계수(Le) 다운로드", data=Le_df.to_csv(encoding="utf-8-sig"), file_name="Le_고용 유발계수.csv", mime="text/csv")