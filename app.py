
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="LinkedIn Job Market Dashboard",layout="wide",initial_sidebar_state="expanded")

post=pd.read_csv("postings.csv")
jobsal=pd.read_csv("salaries.csv")
comp=pd.read_csv("companies.csv")
emp=pd.read_csv("employee_counts.csv")

st.markdown("""
<style>
body{background-color:#0e1117}
.block-container{padding-top:1.5rem}
h1,h2,h3,h4{color:white}
[data-testid=stSidebar]{background-color:#111827}

.section{
    background:radial-gradient(circle at top left,#0f172a,#020617);
    border-radius:22px;
    padding:34px;
    margin-bottom:36px;
    box-shadow:0 20px 40px rgba(0,0,0,0.55);
    border:1px solid rgba(255,255,255,0.06);
}
.kpi-grid{
    display:grid;
    grid-template-columns:repeat(4,1fr);
    gap:26px;
    margin-top:26px;
}
.kpi-card{
    background:linear-gradient(135deg,#1e293b,#020617);
    padding:26px;
    border-radius:18px;
    transition:all .25s ease;
}
.kpi-card:hover{
    transform:translateY(-6px);
    box-shadow:0 14px 30px rgba(0,0,0,0.6);
}
.kpi-title{
    font-size:14px;
    color:#94a3b8;
    letter-spacing:.4px;
}
.kpi-value{
    font-size:36px;
    font-weight:800;
    color:white;
    margin-top:8px;
}
</style>
""",unsafe_allow_html=True)

post=post.merge(jobsal,on="job_id",how="left",suffixes=("","_s"))
post=post.merge(comp,on="company_id",how="left")
post=post.merge(
    emp.groupby("company_id")[["employee_count","follower_count"]]
    .max()
    .reset_index(),
    on="company_id",
    how="left"
)

st.sidebar.markdown("##  Filters")

loc=st.sidebar.selectbox("Location",["All"]+sorted(post["location"].dropna().unique()))
wrk=st.sidebar.selectbox("Work Type",["All"]+sorted(post["formatted_work_type"].dropna().unique()))
exp=st.sidebar.selectbox("Experience Level",["All"]+sorted(post["formatted_experience_level"].dropna().unique()))

df=post.copy()
if loc!="All":
    df=df[df["location"]==loc]
if wrk!="All":
    df=df[df["formatted_work_type"]==wrk]
if exp!="All":
    df=df[df["formatted_experience_level"]==exp]

avg_salary=int(df["normalized_salary"].dropna().mean()) if df["normalized_salary"].notna().any() else 0

st.markdown(f"""
<div class="section">
    <h1> LinkedIn Job Market Dashboard</h1>
    <p style="color:#cbd5f5;font-size:18px;margin-top:-8px">
        Hiring trends, salaries & workforce insights
    </p>
    <div class="kpi-grid">
        <div class="kpi-card">
            <div class="kpi-title">Total Jobs</div>
            <div class="kpi-value">{len(df):,}</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-title">Companies</div>
            <div class="kpi-value">{df["company_id"].nunique():,}</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-title">Locations</div>
            <div class="kpi-value">{df["location"].nunique():,}</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-title">Avg Salary</div>
            <div class="kpi-value">${avg_salary:,}</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("## Hiring Demand Overview")

l1,l2=st.columns(2)

jr=df["title"].value_counts().head(12)
fig1=px.bar(
    x=jr.values,
    y=jr.index,
    orientation="h",
    color=jr.values,
    color_continuous_scale="blues",
    title="Top Job Roles"
)
l1.plotly_chart(fig1,use_container_width=True)

lc=df["location"].value_counts().head(12)
fig2=px.bar(
    x=lc.values,
    y=lc.index,
    orientation="h",
    color=lc.values,
    color_continuous_scale="teal",
    title="Top Hiring Locations"
)
l2.plotly_chart(fig2,use_container_width=True)

st.markdown("---")
st.markdown("## Work & Experience Trends")

w1,w2=st.columns(2)

with w1:
    wt=df["formatted_work_type"].value_counts().reset_index()
    wt.columns=["type","count"]
    fig3=px.pie(
        wt,
        names="type",
        values="count",
        hole=0.6,
        title="Work Type Distribution",
        color_discrete_sequence=[
            "#1f1fd1",
            "#6a0dad",
            "#9b4dcc",
            "#c77dff",
            "#ff6fb1",
            "#ff9f80",
            "#ffa07a"
        ]
    )
    fig3.update_traces(textposition="outside",textinfo="percent")
    st.plotly_chart(fig3,use_container_width=True)

with w2:
    ex=df["formatted_experience_level"].value_counts().reset_index()
    ex.columns=["level","count"]
    fig4=px.bar(
        ex,
        x="level",
        y="count",
        color="count",
        title="Experience Level Demand",
        color_continuous_scale=[
            "#fef3c7",
            "#fde68a",
            "#fcd34d",
            "#fbbf24",
            "#f59e0b",
            "#d97706"
        ]
    )
    fig4.update_layout(xaxis_title="",yaxis_title="Jobs",xaxis_tickangle=-30)
    st.plotly_chart(fig4,use_container_width=True)

st.markdown("---")
st.markdown("## Job Activity")

a1,a2=st.columns(2)

ra=df["remote_allowed"].value_counts().reset_index()
ra.columns=["remote","count"]
ra["remote"]=ra["remote"].map({0:"Not Remote",1:"Remote"})
fig5=px.pie(
    ra,
    names="remote",
    values="count",
    hole=0.5,
    title="Remote vs Onsite Jobs",
    color_discrete_sequence=["#636EFA","#00CC96"]
)
a1.plotly_chart(fig5,use_container_width=True)

df["listed_time"]=pd.to_datetime(df["listed_time"],errors="coerce")
ts=df.groupby(df["listed_time"].dt.to_period("M")).size()
fig6=go.Figure(go.Scatter(x=ts.index.astype(str),y=ts.values,mode="lines+markers"))
fig6.update_layout(title="Job Posting Trend Over Time",xaxis_title="Month",yaxis_title="Jobs")
a2.plotly_chart(fig6,use_container_width=True)


st.markdown("---")
st.markdown("## Skills & Industry Insights")

jobskill=pd.read_csv("job_skills.csv")
skills=pd.read_csv("skills.csv")
jobind=pd.read_csv("job_industries.csv")
inds=pd.read_csv("industries.csv")

sk=jobskill.merge(skills,on="skill_abr",how="left")
sk=sk[sk["job_id"].isin(df["job_id"])]

top_sk=sk["skill_name"].value_counts().head(20).reset_index()
top_sk.columns=["skill","count"]

fig7=px.treemap(
    top_sk,
    path=["skill"],
    values="count",
    color="count",
    color_continuous_scale="viridis",
    title="Top In-Demand Skills"
)
st.plotly_chart(fig7,use_container_width=True)

ji=jobind.merge(inds,on="industry_id",how="left")
ji=ji[ji["job_id"].isin(df["job_id"])]

top_ind=ji["industry_name"].value_counts().head(15).reset_index()
top_ind.columns=["industry","count"]

fig8=px.bar(
    top_ind,
    x="count",
    y="industry",
    orientation="h",
    color="count",
    color_continuous_scale="plasma",
    title="Top Hiring Industries"
)
st.plotly_chart(fig8,use_container_width=True)
st.markdown("---")
st.markdown("## Salary Analysis")

sal=df.dropna(subset=["normalized_salary"])
sal=sal[sal["normalized_salary"]<300000]

s1,s2=st.columns(2)

fig9=px.box(
    sal,
    x="formatted_experience_level",
    y="normalized_salary",
    color="formatted_experience_level",
    title="Salary vs Experience Level"
)
s1.plotly_chart(fig9,use_container_width=True)

fig10=px.violin(
    sal,
    x="formatted_work_type",
    y="normalized_salary",
    color="formatted_work_type",
    box=True,
    points="outliers",
    title="Salary Distribution by Work Type"
)
s2.plotly_chart(fig10,use_container_width=True)

st.markdown("---")
st.markdown("## Skills vs Salary Relationship")

skill_cnt=sk.groupby("job_id").size().reset_index(name="skill_count")
df2=df.merge(skill_cnt,on="job_id",how="left")

fig11=px.scatter(
    df2.dropna(subset=["normalized_salary","skill_count"]),
    x="skill_count",
    y="normalized_salary",
    size="skill_count",
    color="formatted_experience_level",
    title="Skill Count vs Salary",
    labels={
        "skill_count":"Number of Skills",
        "normalized_salary":"Salary"
    }
)
st.plotly_chart(fig11,use_container_width=True)

st.markdown("---")
st.markdown("## Machine Learning: Salary Prediction")

ml=df2[
    [
        "normalized_salary",
        "skill_count",
        "remote_allowed",
        "formatted_experience_level",
        "formatted_work_type"
    ]
].dropna()

ml=ml[ml["normalized_salary"]<300000]

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

y=np.log1p(ml["normalized_salary"])
X=pd.get_dummies(ml.drop("normalized_salary",axis=1))

Xtr,Xte,ytr,yte=train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

rf=RandomForestRegressor(
    n_estimators=200,
    max_depth=12,
    random_state=42
)
rf.fit(Xtr,ytr)
yp=rf.predict(Xte)

m1,m2=st.columns(2)
m1.metric("Model R² Score",round(r2_score(yte,yp),3))
m2.metric("Training Samples",len(Xtr))

st.markdown("### Predict Salary")

e=st.selectbox(
    "Experience Level",
    [c for c in X.columns if "formatted_experience_level" in c]
)
w=st.selectbox(
    "Work Type",
    [c for c in X.columns if "formatted_work_type" in c]
)
s=st.slider("Skill Count",1,50,10)
r=st.selectbox("Remote Allowed",[0,1])

ip=pd.DataFrame(0,index=[0],columns=X.columns)
ip[e]=1
ip[w]=1
ip["skill_count"]=s
ip["remote_allowed"]=r

pred=int(np.expm1(rf.predict(ip))[0])
st.success(f"Predicted Salary: ${pred:,}")
